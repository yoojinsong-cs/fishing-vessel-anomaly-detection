"""
사고 목록(발생일시·위경도·어업면허) ↔ V-Pass 선박 매칭
------------------------------------------------
테스트 엑셀에는 rfid_id가 없으므로, 사고마다
  발생 시각 -2시간 ~ +1시간 사이 V-Pass 위치점 중 사고 위치와 가장 가까웠던 선박을 찾는다.
  - 어업면허(업종)가 같은 선박을 우선 (반경 max_km 안에 있으면)
  - 같은 업종이 없으면 가장 가까운 선박
  - 반경 max_km 안에 아무도 없으면 '매칭 실패'
"""
import numpy as np
import pandas as pd

from .vpass_raw import _hav, parse_multilinestring

EPOCH = np.datetime64("2000-01-01")


def _abs_hour_from_ts(ts: pd.Series) -> np.ndarray:
    d = ts.dt.normalize().to_numpy()
    days = (d - EPOCH).astype("timedelta64[D]").astype(int)
    return days * 24 + ts.dt.hour.to_numpy()


def accident_hours(acc: pd.DataFrame) -> pd.DataFrame:
    acc = acc.copy()
    acc["acc_H"] = _abs_hour_from_ts(acc["acc_time"])
    return acc


def candidates_in_month(df: pd.DataFrame, acc: pd.DataFrame, before_h=2, after_h=1, max_km=10.0,
                        time_penalty_km=0.5) -> pd.DataFrame:
    """
    df : 한 달 원본 (part_dt, rfid_id, rfid_time, geom, fishing_code)
    acc: acc_id, acc_time, acc_H, lat, lon, license
    반환: 사고별 후보 선박 (acc_id, rfid_id, fishing_code, dist_km = 거리 + 시간차 벌점)
    """
    days = pd.to_datetime(df["part_dt"].astype(str), format="%Y%m%d")
    H = ((days.to_numpy() - EPOCH).astype("timedelta64[D]").astype(int)) * 24 + df["rfid_time"].astype(int).to_numpy()
    lo, hi = H.min(), H.max()
    out = []
    for a in acc[(acc["acc_H"] >= lo - after_h) & (acc["acc_H"] <= hi + before_h)].itertuples():
        m = (H >= a.acc_H - before_h) & (H <= a.acc_H + after_h)
        if not m.any():
            continue
        sub = df.loc[m, ["rfid_id", "geom"] + (["fishing_code"] if "fishing_code" in df.columns else [])]
        offs = H[m] - a.acc_H
        best = {}
        for r, off in zip(sub.itertuples(), offs):
            segs = parse_multilinestring(r.geom)
            if not segs:
                continue
            P = np.vstack(segs)
            d = float(_hav(P[:, 1], P[:, 0], a.lat, a.lon).min())
            # 사고 시각에서 먼 시간의 위치일수록 불리하게 (1시간당 time_penalty_km)
            score = d + time_penalty_km * abs(int(off))
            if d <= max_km and score < best.get(r.rfid_id, (np.inf,))[0]:
                best[r.rfid_id] = (score, getattr(r, "fishing_code", ""))
        for vid, (d, fc) in best.items():
            out.append((a.acc_id, str(vid), str(fc), d))
    return pd.DataFrame(out, columns=["acc_id", "rfid_id", "fishing_code", "dist_km"])


def choose_matches(acc: pd.DataFrame, cand: pd.DataFrame, prefer_km=5.0) -> pd.DataFrame:
    """사고별 최종 선박 선택"""
    rows = []
    for a in acc.itertuples():
        c = cand[cand["acc_id"] == a.acc_id]
        c = c.groupby(["rfid_id", "fishing_code"], as_index=False)["dist_km"].min().sort_values("dist_km")
        if c.empty:
            rows.append((a.acc_id, None, None, np.nan, False, 0, 0))
            continue
        same = c[(c["fishing_code"] == str(a.license)) & (c["dist_km"] <= prefer_km)]
        pick = same.iloc[0] if len(same) else c.iloc[0]
        rows.append((a.acc_id, pick["rfid_id"], pick["fishing_code"], float(pick["dist_km"]),
                     bool(pick["fishing_code"] == str(a.license)),
                     int((c["dist_km"] <= 1.0).sum()), int(len(c))))
    m = pd.DataFrame(rows, columns=["acc_id", "rfid_id", "vpass_fishing_code", "match_dist_km", "license_match",
                                    "n_within_1km", "n_candidates"])
    return acc.merge(m, on="acc_id", how="left")
