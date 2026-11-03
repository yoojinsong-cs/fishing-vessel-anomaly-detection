"""
V-Pass 원본 → 이상탐지 입력 변환
------------------------------------------------
원본 컬럼: part_dt(YYYYMMDD), rfid_id(선박), rfid_time(시 0~23), geom(MULTILINESTRING), fishing_code(업종)
  - 원본 1행 = 선박 × 날짜 × 1시간, geom 안에 그 시간의 위치점들이 (경도 위도) 순서로 들어 있음
  - 점별 시각·속력·침로가 없으므로, 한 시간 안의 점들은 '그 시간 동안 고르게 찍혔다'고 가정해 근사 시각을 부여

분석 단위: 선박 × 날짜 × 6시간 시간대 (새벽 00-06 / 오전 06-12 / 오후 12-18 / 야간 18-24)

※ 원래 전처리·특징 계산은 1번(데이터) 담당 파트. 1번 결과가 나오면 그 특징 테이블로 바꿔 끼우면 되고,
  이 모듈은 그 전까지 3번 파트를 실제 데이터로 돌려보기 위한 변환기다.
"""
import re
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd

BAND_LABELS = {0: "새벽(00-06)", 1: "오전(06-12)", 2: "오후(12-18)", 3: "야간(18-24)"}
SEQ_NAMES = ["speed_kn", "sin_course", "cos_course", "dx_km", "dy_km", "log_dt_min"]


@dataclass
class RawConfig:
    band_hours: int = 6
    bbox: tuple = (120.0, 135.0, 30.0, 40.0)   # 경도 min,max / 위도 min,max (한반도 주변 해역)
    spike_km: float = 3.0         # 앞뒤 점 모두에서 이만큼 튀었다가 돌아오면 GPS 튐으로 제거
    max_step_kn: float = 50.0     # 근사 속력이 이보다 크면 비정상 이동으로 제거 (점 시각이 근사값이라 여유 있게)
    drift_kn: tuple = (0.2, 2.0)  # 표류 의심 시간의 속력 범위
    drift_straight: float = 0.6   # 표류 의심 시간의 직진성 하한
    stop_kn: float = 0.5          # 정지 판정 속력
    move_step_km: float = 0.02    # 침로 변화 계산에 쓸 최소 이동 거리 (GPS 잔떨림 제외)
    gap_min: float = 30.0         # 이보다 긴 시간 간격 = 신호 단절
    sharp_turn_deg: float = 60.0
    min_points: int = 10
    min_move_km: float = 1.0      # 시간대 내 총 이동거리가 이보다 작으면 '항내 정박'으로 보고 제외
    seq_len: int = 64
    # 검증용 인위적 이상 궤적
    synth_rate: float = 0.015     # 조건 맞는 궤적 중 이 비율만큼 표류/신호단절 각각 생성
    seed: int = 42


# ---------------------------------------------------------------------------
# geom 파싱
# ---------------------------------------------------------------------------
_num = re.compile(r"[-\d.]+")


def parse_multilinestring(s: str) -> List[np.ndarray]:
    """'MULTILINESTRING ((x y, x y), (x y))' → [array(n,2) [lon, lat], ...]"""
    body = s[s.find("((") + 2: s.rfind("))")]
    segs = []
    for part in body.split("), ("):
        v = np.array(_num.findall(part), dtype=float)
        if len(v) >= 2:
            segs.append(v[: len(v) // 2 * 2].reshape(-1, 2))
    return segs


def _hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ---------------------------------------------------------------------------
# 한 분석 단위(선박×날짜×시간대)의 점 배열 만들기
# ---------------------------------------------------------------------------
def unit_points(rows, rc: RawConfig, band: int):
    """rows: [(hour, geom), ...] → lon, lat, t(분, 시간대 시작 기준), n_seg"""
    lons, lats, ts = [], [], []
    n_seg = 0
    for hour, geom in sorted(rows, key=lambda r: r[0]):
        segs = parse_multilinestring(geom)
        n_seg += len(segs)
        if not segs:
            continue
        P = np.vstack(segs)
        n = len(P)
        h0 = (hour - band * rc.band_hours) * 60.0
        lons.append(P[:, 0])
        lats.append(P[:, 1])
        ts.append(h0 + (np.arange(n) + 0.5) * 60.0 / n)
    if not lons:
        return None
    return np.concatenate(lons), np.concatenate(lats), np.concatenate(ts), n_seg


def clean_points(lon, lat, t, rc: RawConfig):
    """해역 밖 좌표 제거 → GPS 튐(스파이크) 제거 → 비정상 고속 이동 제거"""
    x0, x1, y0, y1 = rc.bbox
    ok = (lon > x0) & (lon < x1) & (lat > y0) & (lat < y1)
    lon, lat, t = lon[ok], lat[ok], t[ok]
    n_removed = int((~ok).sum())
    if len(lon) >= 3:
        d_prev = _hav(lat[1:-1], lon[1:-1], lat[:-2], lon[:-2])
        d_next = _hav(lat[1:-1], lon[1:-1], lat[2:], lon[2:])
        d_skip = _hav(lat[:-2], lon[:-2], lat[2:], lon[2:])
        spike = (d_prev > rc.spike_km) & (d_next > rc.spike_km) & (d_skip < rc.spike_km)
        keep = np.r_[True, ~spike, True]
        n_removed += int(spike.sum())
        lon, lat, t = lon[keep], lat[keep], t[keep]
    # 근사 속력 기준 비정상 점 반복 제거 (최대 3회)
    for _ in range(3):
        if len(lon) < 2:
            break
        d = _hav(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dt = np.diff(t)
        v = d / 1.852 / np.clip(dt / 60.0, 1e-3, None)
        bad = np.r_[False, (v > rc.max_step_kn) & (dt < rc.gap_min)]
        if not bad.any():
            break
        n_removed += int(bad.sum())
        lon, lat, t = lon[~bad], lat[~bad], t[~bad]
    return lon, lat, t, n_removed


# ---------------------------------------------------------------------------
# 특징 + 시퀀스
# ---------------------------------------------------------------------------
def _resample(arr, T):
    L = len(arr)
    if L == 1:
        return np.repeat(arr, T, axis=0)
    src, dst = np.linspace(0, 1, L), np.linspace(0, 1, T)
    return np.stack([np.interp(dst, src, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def unit_features(lon, lat, t, n_seg, rc: RawConfig):
    d = _hav(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt = np.diff(t)
    cont = dt <= rc.gap_min                               # 신호가 이어진 구간
    v = d / 1.852 / np.clip(dt / 60.0, 1e-3, None)        # 근사 속력(kn)
    brg = _bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])

    # 시간별 속력 = 그 시간 동안 이동거리 / 1시간
    hour_idx = np.floor(t[1:] / 60.0).astype(int)
    hours = np.unique(np.floor(t / 60.0).astype(int))
    hd = np.array([d[(hour_idx == h) & cont].sum() for h in hours]) / 1.852
    # 표류 징후: 저속(0.2~2kn)인데 한 방향으로 꾸준히 밀려가는 시간 (조업 중 정지는 제자리, 표류는 직선 이동)
    pt_hour = np.floor(t / 60.0).astype(int)
    h_straight = np.zeros(len(hours))
    for i, h in enumerate(hours):
        m = pt_hour == h
        if m.sum() >= 3 and hd[i] > 0:
            h_disp = _hav(lat[m][0], lon[m][0], lat[m][-1], lon[m][-1]) / 1.852
            h_straight[i] = min(h_disp / hd[i], 1.0)
    drift_like = (hd >= rc.drift_kn[0]) & (hd <= rc.drift_kn[1]) & (h_straight >= rc.drift_straight)
    drift_run = run_h = 0
    prev = None
    for h, dl in zip(hours, drift_like):
        run_h = (run_h + 1 if (prev is not None and h == prev + 1 and run_h > 0) else 1) if dl else 0
        drift_run = max(drift_run, run_h)
        prev = h
    dist = d[cont].sum()
    disp = _hav(lat[0], lon[0], lat[-1], lon[-1])

    mv = (d >= rc.move_step_km) & cont
    b = brg[mv]
    turn = np.abs((np.diff(b) + 180) % 360 - 180) if len(b) > 1 else np.zeros(1)

    stop = (v < rc.stop_kn) & cont
    best = run = 0.0
    for s, dd in zip(stop, dt):
        run = run + dd if s else 0.0
        best = max(best, run)

    clat, clon = lat.mean(), lon.mean()
    feats = {
        "speed_mean": hd.mean(), "speed_max": hd.max(), "speed_std": hd.std(),
        "distance_km": dist, "straightness": disp / dist if dist > 1e-6 else 0.0,
        "course_change_mean": turn.mean(), "sharp_turn_cnt": int((turn > rc.sharp_turn_deg).sum()),
        "stop_ratio": stop[cont].mean() if cont.any() else 1.0, "max_stop_min": best,
        "radius_km": _hav(lat, lon, clat, clon).max(),
        "n_points": len(lon), "n_segments": n_seg, "active_hours": len(hours),
        "max_gap_min": dt.max() if len(dt) else 0.0,
        "drift_hours": drift_run, "drift_hour_ratio": drift_like.mean(),
        "center_lat": clat, "center_lon": clon,
        "_hours": hours, "_hour_speed": hd,
    }
    # 시퀀스: [근사속력, sin/cos 침로, dx, dy, log(시간간격)]
    x = (lon - lon[0]) * 111.32 * np.cos(np.radians(clat))
    y = (lat - lat[0]) * 110.57
    seq = np.stack([np.r_[0, np.clip(v, 0, 30)],
                    np.r_[0, np.sin(np.radians(brg))], np.r_[0, np.cos(np.radians(brg))],
                    np.r_[0, np.diff(x)], np.r_[0, np.diff(y)],
                    np.r_[0, np.log1p(np.clip(dt, 0, None))]], axis=1)
    return feats, _resample(seq, rc.seq_len).astype(np.float32)


# ---------------------------------------------------------------------------
# 인위적 이상 궤적 (검증용)
# ---------------------------------------------------------------------------
def make_drift(lon, lat, t, rng):
    """중간 이후 기관 고장 → 0.3~1.0kn로 조류 방향 표류"""
    lon, lat = lon.copy(), lat.copy()
    k = int(len(lon) * rng.uniform(0.35, 0.55))
    cur = rng.uniform(0, 360)
    for i in range(k + 1, len(lon)):
        s = abs(rng.normal(0.6, 0.25))
        c = np.radians(cur + rng.normal(0, 25))
        km = s * 1.852 * (t[i] - t[i - 1]) / 60.0
        lat[i] = lat[i - 1] + km * np.cos(c) / 110.57
        lon[i] = lon[i - 1] + km * np.sin(c) / (111.32 * np.cos(np.radians(lat[i - 1])))
    return lon, lat, t


def make_signal_loss(lon, lat, t, rng):
    """중간 2~3시간 신호 단절"""
    hours = np.unique(np.floor(t / 60.0).astype(int))
    inner = hours[1:-1]
    L = int(rng.integers(2, 4))
    if len(inner) < L:
        return None
    s = int(rng.integers(0, len(inner) - L + 1))
    drop = np.isin(np.floor(t / 60.0).astype(int), inner[s:s + L])
    return lon[~drop], lat[~drop], t[~drop]


# ---------------------------------------------------------------------------
# 병렬 처리
# ---------------------------------------------------------------------------
def _process_chunk(args):
    units, rc, seed = args
    rng = np.random.default_rng(seed)
    F, S, E, Q = [], [], [], {"too_few": 0, "in_port": 0, "removed_pts": 0}
    for key, rows in units:
        vid, dt_, band, ftype = key
        p = unit_points(rows, rc, band)
        if p is None:
            Q["too_few"] += 1
            continue
        lon, lat, t, nseg = p
        lon, lat, t, nrem = clean_points(lon, lat, t, rc)
        Q["removed_pts"] += nrem
        if len(lon) < rc.min_points:
            Q["too_few"] += 1
            continue
        f, s = unit_features(lon, lat, t, nseg, rc)
        if f["distance_km"] < rc.min_move_km:
            Q["in_port"] += 1
            continue
        hrs, hsp = f.pop("_hours"), f.pop("_hour_speed")
        E.extend(zip([ftype] * len(hrs), (band * rc.band_hours + hrs).tolist(), hsp.tolist()))
        base = {"vessel_id": vid, "date": dt_, "band": band, "reg_type": ftype,
                "trip_id": f"{vid}_{dt_}_{band}", "is_synthetic": 0, "anomaly_type": "normal",
                "removed_pts": nrem}
        F.append({**base, **f})
        S.append(s)
        # 인위적 이상
        if f["active_hours"] >= 3 and rng.random() < rc.synth_rate:
            dl = make_drift(lon, lat, t, rng)
            f2, s2 = unit_features(*dl, nseg, rc)
            f2.pop("_hours"); f2.pop("_hour_speed")
            F.append({**base, **f2, "trip_id": "SYN_drift_" + base["trip_id"], "is_synthetic": 1,
                      "anomaly_type": "drift"})
            S.append(s2)
        if f["active_hours"] >= 4 and rng.random() < rc.synth_rate:
            sl = make_signal_loss(lon, lat, t, rng)
            if sl is not None and len(sl[0]) >= rc.min_points:
                f2, s2 = unit_features(*sl, nseg, rc)
                f2.pop("_hours"); f2.pop("_hour_speed")
                F.append({**base, **f2, "trip_id": "SYN_signal_loss_" + base["trip_id"], "is_synthetic": 1,
                          "anomaly_type": "signal_loss"})
                S.append(s2)
    S = np.stack(S) if S else np.zeros((0, rc.seq_len, len(SEQ_NAMES)), np.float32)
    return pd.DataFrame(F), S, Q, pd.DataFrame(E, columns=["reg_type", "hour", "speed_kn"])


def hourly_summary(E: pd.DataFrame, stop_kn: float = 0.5) -> pd.DataFrame:
    """업종 × 시(0~23)별 정지 비율·활동 비중 (조업 궤적의 시간별 근사 속력 기준)"""
    E = E.assign(is_stop=(E["speed_kn"] < stop_kn).astype(int))
    out = E.groupby(["reg_type", "hour"]).agg(n_points=("is_stop", "size"), stop_ratio=("is_stop", "mean"),
                                                speed_mean=("speed_kn", "mean")).reset_index()
    out["move_ratio"] = 1 - out["stop_ratio"]
    out["activity_share"] = out["n_points"] / out.groupby("reg_type")["n_points"].transform("sum")
    return out


def build_from_raw(df: pd.DataFrame, rc: RawConfig = RawConfig(), n_jobs: int = 2):
    """원본 DataFrame → (특징 테이블, 시퀀스 X, ids, 품질 요약, 시간별 속력 기록)"""
    df = df.copy()
    df["part_dt"] = df["part_dt"].astype(str)
    df["rfid_id"] = df["rfid_id"].astype(str)
    df["band"] = (df["rfid_time"] // rc.band_hours).astype(int)
    groups = list(df.groupby(["rfid_id", "part_dt", "band", "fishing_code"], sort=False))
    units = [(k, list(zip(g["rfid_time"].values, g["geom"].values))) for k, g in groups]
    del groups
    n = max(1, n_jobs)
    chunks = [units[i::n * 8] for i in range(n * 8)]
    args = [(ch, rc, rc.seed + i) for i, ch in enumerate(chunks)]
    if n > 1:
        with Pool(n) as pool:
            res = pool.map(_process_chunk, args)
    else:
        res = [_process_chunk(a) for a in args]
    F = pd.concat([r[0] for r in res], ignore_index=True)
    X = np.concatenate([r[1] for r in res])
    Q = {k: sum(r[2][k] for r in res) for k in res[0][2]}
    E = pd.concat([r[3] for r in res], ignore_index=True)
    Q["units_total"] = len(units)
    F["date"] = pd.to_datetime(F["date"], format="%Y%m%d")
    F["time_band"] = F["band"].map(BAND_LABELS)
    F["start_time"] = F["date"] + pd.to_timedelta(F["band"] * rc.band_hours, unit="h")
    return F, X, F["trip_id"].to_numpy(), Q, E
