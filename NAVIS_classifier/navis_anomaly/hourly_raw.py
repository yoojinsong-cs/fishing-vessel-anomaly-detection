"""
V-Pass 원본 → '시간 단위' 이상점수용 데이터 + 분류모델용 시간별 통계
------------------------------------------------
제출 형식(testB)이 '선박 × 시각(YYYYMMDD HH)별 이상항적 점수'라서,
각 시간 h마다 주변 5시간 [h-2, h+2] 창의 항적으로 특징을 계산해 점수를 매긴다.
  - 시간 하나만 보면 점이 적고, 표류·신호단절은 앞뒤 시간을 같이 봐야 드러나기 때문
  - 창은 선박 단위로 이어 붙인 시간축에서 잡으므로 자정을 넘어가도 끊기지 않음

같은 원본 파싱 결과로 '선박별 시간 통계'도 만들어 분류모델(testA) 특징으로 쓴다.
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .vpass_raw import (BAND_LABELS, RawConfig, _bearing, _hav, clean_points, parse_multilinestring,
                        unit_features)

HALF = 2              # 창 = [h-2, h+2] (5시간)
GAP_CAP = 360.0       # 신호 단절 시간 상한(분)


def _day_index(part_dt: np.ndarray) -> np.ndarray:
    d = pd.to_datetime(pd.Series(part_dt).astype(str), format="%Y%m%d").to_numpy()
    return (d - np.datetime64("2000-01-01")).astype("timedelta64[D]").astype(int)


def _hour_label(H: int) -> tuple:
    day = np.datetime64("2000-01-01") + np.timedelta64(int(H // 24), "D")
    return str(day).replace("-", ""), int(H % 24)


# ---------------------------------------------------------------------------
# 선박 하나 처리
# ---------------------------------------------------------------------------
def _vessel_points(Hs, geoms, rc):
    lons, lats, ts, nseg = [], [], [], {}
    for H, g in zip(Hs, geoms):
        segs = parse_multilinestring(g)
        nseg[H] = nseg.get(H, 0) + len(segs)
        if not segs:
            continue
        P = np.vstack(segs)
        n = len(P)
        lons.append(P[:, 0]); lats.append(P[:, 1])
        ts.append(H * 60.0 + (np.arange(n) + 0.5) * 60.0 / n)
    if not lons:
        return None
    lon, lat, t = np.concatenate(lons), np.concatenate(lats), np.concatenate(ts)
    o = np.argsort(t, kind="stable")
    lon, lat, t, nrem = clean_points(lon[o], lat[o], t[o], rc)
    return lon, lat, t, nseg, nrem


def _hour_stats(lon, lat, t, nseg, rc):
    """시간별 통계 (분류 특징·EDA·창 중심 특징에 사용)"""
    Hp = np.floor(t / 60.0).astype(int)
    hours, start = np.unique(Hp, return_index=True)
    end = np.r_[start[1:], len(t)]
    rows = {}
    for H, s, e in zip(hours, start, end):
        la, lo, tt = lat[s:e], lon[s:e], t[s:e]
        n = e - s
        if n >= 2:
            d = _hav(la[:-1], lo[:-1], la[1:], lo[1:])
            dt = np.diff(tt)
            v = d / 1.852 / np.clip(dt / 60.0, 1e-3, None)
            dist = d.sum()
            disp = _hav(la[0], lo[0], la[-1], lo[-1])
            mv = d >= rc.move_step_km
            b = _bearing(la[:-1], lo[:-1], la[1:], lo[1:])[mv]
            turn = np.abs((np.diff(b) + 180) % 360 - 180).mean() if len(b) > 1 else 0.0
            stop = (v < rc.stop_kn).mean()
        else:
            dist = disp = turn = 0.0
            stop = 1.0
        rows[H] = dict(h_speed=dist / 1.852, h_straight=min(disp / dist, 1.0) if dist > 1e-6 else 0.0,
                       h_turn=turn, h_stop=stop, h_npts=n, h_nseg=nseg.get(H, 0),
                       h_lat=la.mean(), h_lon=lo.mean(), first=tt[0], last=tt[-1])
    return rows


def _home(hs):
    """선박의 모항(정박 위치) 추정: 거의 안 움직인 시간들의 위치 중앙값"""
    st = [(r["h_lat"], r["h_lon"]) for r in hs.values() if r["h_speed"] < 0.3]
    if not st:
        st = [(r["h_lat"], r["h_lon"]) for r in hs.values()]
    a = np.array(st)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1]))


def _drift_like(r, rc):
    return rc.drift_kn[0] <= r["h_speed"] <= rc.drift_kn[1] and r["h_straight"] >= rc.drift_straight


def _windows(lon, lat, t, nseg, hs, centers, rc, base, home, extra=None):
    """centers 각 시간을 중심으로 5시간 창 특징 계산"""
    out, seqs = [], []
    for h in centers:
        if h not in hs:
            continue
        s = np.searchsorted(t, (h - HALF) * 60.0)
        e = np.searchsorted(t, (h + HALF + 1) * 60.0)
        if e - s < 3:
            continue
        ns = sum(nseg.get(x, 0) for x in range(h - HALF, h + HALF + 1))
        f, sq = unit_features(lon[s:e], lat[s:e], t[s:e], ns, rc)
        f.pop("_hours"); f.pop("_hour_speed")
        c = hs[h]
        prev = hs.get(h - 1)
        # 중심 시간 이후/이전 신호 공백 (다음 점까지 시간)
        nxt = np.searchsorted(t, c["last"], side="right")
        gap_after = min(t[nxt] - c["last"], GAP_CAP) if nxt < len(t) else GAP_CAP
        prv = np.searchsorted(t, c["first"], side="left") - 1
        gap_before = min(c["first"] - t[prv], GAP_CAP) if prv >= 0 else GAP_CAP
        # 사고 '시작 시점'을 짚기 위한 전후 대비 특징
        bef = [hs[x] for x in (h - 2, h - 1) if x in hs]
        aft = [hs[x] for x in (h + 1, h + 2) if x in hs]
        spd_drop = (np.mean([r["h_speed"] for r in bef]) - np.mean([r["h_speed"] for r in [c] + aft])) if bef else 0.0
        drift_onset = sum(_drift_like(r, rc) for r in [c] + aft) - sum(_drift_like(r, rc) for r in bef)
        home_dist = float(_hav(c["h_lat"], c["h_lon"], home[0], home[1]))
        dstr, hh = _hour_label(h)
        row = {**base, "date": dstr, "hour": hh, "H": h,
               "trip_id": f"{base['trip_prefix']}{base['vessel_id']}_{dstr}_{hh:02d}",
               **f,
               "c_speed": c["h_speed"], "c_stop": c["h_stop"], "c_npts": c["h_npts"],
               "d_speed_prev": c["h_speed"] - prev["h_speed"] if prev else 0.0,
               "gap_after_min": gap_after, "gap_before_min": gap_before,
               "spd_drop": spd_drop, "drift_onset": drift_onset, "home_dist_km": home_dist,
               # 모항에서 멀리 떨어진 바다에서 신호가 끊긴 정도 (정상 입항 후 전원 끔과 구분)
               "gap_at_sea": gap_after if home_dist > 3.0 else 0.0}
        if extra:
            row.update(extra(h))
        row.pop("trip_prefix")
        out.append(row)
        seqs.append(sq)
    return out, seqs


def _synthetic_cases(vid, ftype, lon, lat, t, nseg, hs, rc, rng, home):
    """
    검증용 인위적 사고 (사고 시각 k를 알고 있는 궤적)
      drift       : k시 이후 기관 고장 → 0.3~1.0kn로 한 방향 표류
      signal_loss : k시를 마지막으로 신호가 끊김 (이후 기록 없음)
    조건: k-4 ~ k+3 시간이 모두 있고, 사고 직전 이동 중(1.5kn 이상)
    """
    hset = set(hs)
    cand = [k for k in hs if all((k + d) in hset for d in range(-4, 4))
            and hs[k - 1]["h_speed"] >= 1.5 and hs[k]["h_speed"] >= 1.5]
    if not cand:
        return [], []
    out, seqs = [], []
    for kind in ("drift", "signal_loss"):
        if rng.random() >= rc.synth_rate:
            continue
        k = int(rng.choice(cand))
        m = (t >= (k - 6) * 60) & (t < (k + 6) * 60) if kind == "drift" else (t >= (k - 6) * 60) & (t < (k + 1) * 60)
        lo, la, tt = lon[m].copy(), lat[m].copy(), t[m]
        if kind == "drift":
            st = np.searchsorted(tt, k * 60 + 15)
            cur = rng.uniform(0, 360)
            for i in range(max(st, 1), len(tt)):
                sp = abs(rng.normal(0.6, 0.25))
                cc = np.radians(cur + rng.normal(0, 25))
                km = sp * 1.852 * (tt[i] - tt[i - 1]) / 60.0
                la[i] = la[i - 1] + km * np.cos(cc) / 110.57
                lo[i] = lo[i - 1] + km * np.sin(cc) / (111.32 * np.cos(np.radians(la[i - 1])))
            centers = range(k - 4, k + 4)
        else:
            centers = range(k - 4, k + 1)
        hs2 = _hour_stats(lo, la, tt, nseg, rc)
        base = {"vessel_id": vid, "reg_type": ftype, "is_synthetic": 1, "anomaly_type": kind,
                "case_id": f"{kind}_{vid}_{k}", "case_k": k, "trip_prefix": f"SYN_{kind}_"}
        o, s = _windows(lo, la, tt, nseg, hs2, centers, rc, base, home)
        out += o
        seqs += s
    return out, seqs


def _process_chunk(args):
    vessels, rc, mode, rate, seed, want_seq = args
    rng = np.random.default_rng(seed)
    W, S, HS = [], [], []
    for vid, ftype, Hs, geoms in vessels:
        p = _vessel_points(Hs, geoms, rc)
        if p is None:
            continue
        lon, lat, t, nseg, _ = p
        if len(t) < 3:
            continue
        hs = _hour_stats(lon, lat, t, nseg, rc)
        for H, r in hs.items():
            HS.append((vid, ftype, H, r["h_speed"], r["h_straight"], r["h_turn"], r["h_stop"],
                       r["h_npts"], r["h_nseg"], r["h_lat"], r["h_lon"]))
        hours = np.array(sorted(hs))
        centers = hours if mode == "test" else hours[rng.random(len(hours)) < rate]
        base = {"vessel_id": vid, "reg_type": ftype, "is_synthetic": 0, "anomaly_type": "normal",
                "case_id": "", "case_k": -1, "trip_prefix": ""}
        home = _home(hs)
        o, s = _windows(lon, lat, t, nseg, hs, centers, rc, base, home)
        W += o
        S += s
        if mode == "train" and rc.synth_rate > 0:
            o, s = _synthetic_cases(vid, ftype, lon, lat, t, nseg, hs, rc, rng, home)
            W += o
            S += s
    S = (np.stack(S) if S else np.zeros((0, rc.seq_len, 6), np.float32)) if want_seq else None
    HS = pd.DataFrame(HS, columns=["vessel_id", "reg_type", "H", "h_speed", "h_straight", "h_turn", "h_stop",
                                   "h_npts", "h_nseg", "h_lat", "h_lon"])
    return pd.DataFrame(W), S, HS


def build_hourly(df: pd.DataFrame, rc: RawConfig, mode: str = "train", train_hour_rate: float = 0.2,
                 n_jobs: int = 2, want_seq: bool = False, seed: int = 0):
    """
    df: 원본 (part_dt, rfid_id, rfid_time, geom[, fishing_code])
    mode='train': 정상 창은 시간의 train_hour_rate 비율만 샘플링 + 인위적 사고 생성
    mode='test' : 모든 시간에 대해 창 계산
    반환: 창 특징 DataFrame, 시퀀스(옵션), 시간별 통계 DataFrame
    """
    df = df.copy()
    if "fishing_code" not in df.columns:
        df["fishing_code"] = "미상"
    df["fishing_code"] = df["fishing_code"].fillna("미상").astype(str)
    df["rfid_id"] = df["rfid_id"].astype(str)
    df["H"] = _day_index(df["part_dt"].values) * 24 + df["rfid_time"].astype(int).values
    vessels = []
    for vid, g in df.groupby("rfid_id", sort=False):
        g = g.sort_values("H")
        vessels.append((vid, g["fishing_code"].iloc[0], g["H"].to_numpy(), g["geom"].to_numpy()))
    del df
    n = max(1, n_jobs) * 8
    chunks = [vessels[i::n] for i in range(n)]
    args = [(ch, rc, mode, train_hour_rate, seed * 1000 + i, want_seq) for i, ch in enumerate(chunks)]
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            res = pool.map(_process_chunk, args)
    else:
        res = [_process_chunk(a) for a in args]
    W = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True) if any(len(r[0]) for r in res) else pd.DataFrame()
    S = np.concatenate([r[1] for r in res]) if want_seq else None
    HS = pd.concat([r[2] for r in res if len(r[2])], ignore_index=True) if any(len(r[2]) for r in res) else res[0][2]
    num = ["H", "h_speed", "h_straight", "h_turn", "h_stop", "h_npts", "h_nseg", "h_lat", "h_lon"]
    HS[num] = HS[num].apply(pd.to_numeric)
    HS["H"] = HS["H"].astype("int64")
    if len(W):
        W["date"] = pd.to_datetime(W["date"], format="%Y%m%d")
        W["band"] = W["hour"] // rc.band_hours
        W["time_band"] = W["band"].map(BAND_LABELS)
        W["start_time"] = W["date"] + pd.to_timedelta(W["hour"], unit="h")
    HS["hour"] = HS["H"] % 24
    HS["day"] = HS["H"] // 24
    return W, S, HS
