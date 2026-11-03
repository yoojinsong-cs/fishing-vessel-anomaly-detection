"""
최소 특징 계산기 (데모·대체용)
------------------------------------------------
원래 특징 계산은 1번(데이터) 담당 파트. 이 파일은
  - 데모 데이터 검증
  - 인위적 이상 궤적을 직접 만들어야 할 때
처럼 점 단위 데이터에서 특징을 빠르게 다시 계산해야 할 때만 쓴다.
"""
import numpy as np
import pandas as pd

from .config import Config


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_features(points: pd.DataFrame, cfg: Config, gap_min: float = 30.0,
                     sharp_turn_deg: float = 60.0) -> pd.DataFrame:
    c = cfg.cols
    rows = []
    pts = points.sort_values([c.unit_id, c.seq_time])
    for uid, g in pts.groupby(c.unit_id, sort=False):
        lat, lon = g[c.lat].to_numpy(float), g[c.lon].to_numpy(float)
        t = pd.to_datetime(g[c.seq_time]).to_numpy()
        spd = g[c.speed].to_numpy(float)
        crs = g[c.course].to_numpy(float)
        if len(g) < 2:
            continue
        d = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dt = np.diff(t).astype("timedelta64[s]").astype(float) / 60.0
        dist = d.sum()
        disp = _haversine_km(lat[0], lon[0], lat[-1], lon[-1])
        dc = np.abs((np.diff(crs) + 180) % 360 - 180)
        stop = spd < cfg.stop_speed_kn
        # 최장 연속 정지 시간
        run, best = 0.0, 0.0
        for s, dd in zip(stop[1:], dt):
            run = run + dd if s else 0.0
            best = max(best, run)
        clat, clon = lat.mean(), lon.mean()
        rad = _haversine_km(lat, lon, clat, clon).max()
        start = pd.Timestamp(t[0])
        rows.append({
            c.unit_id: uid, c.vessel_id: g[c.vessel_id].iloc[0], c.reg_type: g[c.reg_type].iloc[0],
            c.date: start.normalize(), c.start_time: start,
            "speed_mean": spd.mean(), "speed_max": spd.max(), "speed_std": spd.std(),
            "distance_km": dist, "straightness": disp / dist if dist > 1e-6 else 0.0,
            "course_change_mean": dc.mean(), "sharp_turn_cnt": int((dc > sharp_turn_deg).sum()),
            "stop_ratio": stop.mean(), "max_stop_min": best, "radius_km": rad,
            "n_points": len(g), "n_segments": int((dt > gap_min).sum()) + 1,
            "max_gap_min": dt.max(),
        })
    return pd.DataFrame(rows)
