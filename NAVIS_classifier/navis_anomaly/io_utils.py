"""데이터 불러오기 / 기본 정리"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        for enc in ("utf-8-sig", "cp949", "euc-kr"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"인코딩을 알 수 없음: {path}")
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"지원하지 않는 형식: {path}")


# ---------------------------------------------------------------------------
# 특징 테이블
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """특징 테이블 기본 정리: 분석 단위 ID, 시간대, 인위적 이상 표시, 특징 목록 확정"""
    c = cfg.cols
    df = df.copy()

    if c.vessel_id not in df.columns:
        raise KeyError(f"선박 ID 컬럼 '{c.vessel_id}' 없음. config.Columns.vessel_id 수정 필요")
    if c.reg_type not in df.columns:
        raise KeyError(f"등록 업종 컬럼 '{c.reg_type}' 없음. config.Columns.reg_type 수정 필요")

    # 분석 단위 ID
    if c.unit_id not in df.columns:
        if c.date and c.date in df.columns:
            df[c.unit_id] = df[c.vessel_id].astype(str) + "_" + df[c.date].astype(str)
        else:
            df[c.unit_id] = df[c.vessel_id].astype(str) + "_" + df.groupby(c.vessel_id).cumcount().astype(str)
    df[c.unit_id] = df[c.unit_id].astype(str)

    # 시간대(주간/야간)
    if c.time_band not in df.columns:
        if c.start_time and c.start_time in df.columns:
            hour = pd.to_datetime(df[c.start_time], errors="coerce").dt.hour
            is_day = (hour >= cfg.day_start_hour) & (hour < cfg.day_end_hour)
            df[c.time_band] = np.where(is_day, cfg.day_label, cfg.night_label)
        else:
            df[c.time_band] = "전체"
    df[c.time_band] = df[c.time_band].astype(str)
    df[c.reg_type] = df[c.reg_type].astype(str)

    # 날짜
    if c.date and c.date in df.columns:
        df[c.date] = pd.to_datetime(df[c.date], errors="coerce")
    elif c.start_time and c.start_time in df.columns:
        df[c.date or "date"] = pd.to_datetime(df[c.start_time], errors="coerce").dt.normalize()
        c.date = c.date or "date"

    # 인위적 이상 표시
    if c.is_synthetic not in df.columns:
        df[c.is_synthetic] = 0
    df[c.is_synthetic] = df[c.is_synthetic].fillna(0).astype(int)
    if c.anomaly_type not in df.columns:
        df[c.anomaly_type] = np.where(df[c.is_synthetic] == 1, "synthetic", "normal")
    df[c.anomaly_type] = df[c.anomaly_type].fillna("normal").astype(str)
    df.loc[df[c.is_synthetic] == 0, c.anomaly_type] = "normal"

    # 사용할 특징 확정 (없는 컬럼 제외, 숫자형만)
    feats = [f for f in cfg.features if f in df.columns]
    missing = [f for f in cfg.features if f not in df.columns]
    if missing:
        print(f"[알림] 특징 테이블에 없는 변수 제외: {missing}")
    if not feats:
        # 설정한 이름이 하나도 없으면 숫자형 컬럼 전부 사용
        exclude = {c.is_synthetic}
        feats = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude]
        print(f"[알림] 설정된 특징이 없어 숫자형 컬럼 전체 사용: {feats}")
    for f in feats:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    cfg.features = feats
    return df


# ---------------------------------------------------------------------------
# 시퀀스 데이터 → (N, T, F) 고정 길이 배열
# ---------------------------------------------------------------------------
def _local_xy_km(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """첫 점 기준 국지 평면 좌표(km). 위치 자체가 아니라 '움직임 모양'만 학습하게 하려는 목적"""
    lat0 = np.nanmean(lat)
    x = (lon - lon[0]) * 111.32 * np.cos(np.radians(lat0))
    y = (lat - lat[0]) * 110.57
    return x, y


def _resample(arr: np.ndarray, T: int) -> np.ndarray:
    """(L, F) → (T, F) 순서 기준 선형 보간"""
    L = arr.shape[0]
    if L == 1:
        return np.repeat(arr, T, axis=0)
    src = np.linspace(0, 1, L)
    dst = np.linspace(0, 1, T)
    return np.stack([np.interp(dst, src, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def long_to_sequences(points: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    점 단위 long format (unit_id, seq_idx, lat, lon, speed, course[, timestamp])
    → X: (N, T, F), ids: (N,), feature_names

    시퀀스 특징 (점마다):
      speed, sin(course), cos(course), 이동 dx·dy(km, 점 간), [log1p(시간 간격 분)]
    """
    c = cfg.cols
    T = cfg.seq_len
    has_time = c.seq_time and c.seq_time in points.columns
    pts = points.copy()
    if has_time:
        pts[c.seq_time] = pd.to_datetime(pts[c.seq_time], errors="coerce")
    sort_cols = [c.unit_id] + ([c.seq_order] if c.seq_order in pts.columns else []) + ([c.seq_time] if has_time else [])
    pts = pts.sort_values(sort_cols)

    names = ["speed", "sin_course", "cos_course", "dx_km", "dy_km"] + (["log_dt_min"] if has_time else [])
    X_list, ids = [], []
    for uid, g in pts.groupby(c.unit_id, sort=False):
        lat = g[c.lat].to_numpy(float)
        lon = g[c.lon].to_numpy(float)
        ok = ~(np.isnan(lat) | np.isnan(lon))
        if ok.sum() < 2:
            continue
        g = g[ok]
        lat, lon = lat[ok], lon[ok]
        x, y = _local_xy_km(lat, lon)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        spd = g[c.speed].to_numpy(float) if c.speed in g.columns else np.hypot(dx, dy)
        crs = np.radians(g[c.course].to_numpy(float)) if c.course in g.columns else np.arctan2(dx, dy)
        cols = [np.nan_to_num(spd), np.sin(np.nan_to_num(crs)), np.cos(np.nan_to_num(crs)), dx, dy]
        if has_time:
            dt = g[c.seq_time].diff().dt.total_seconds().fillna(0).to_numpy() / 60.0
            cols.append(np.log1p(np.clip(dt, 0, None)))
        arr = np.stack(cols, axis=1)
        X_list.append(_resample(arr, T))
        ids.append(str(uid))
    X = np.stack(X_list).astype(np.float32) if X_list else np.zeros((0, T, len(names)), np.float32)
    return X, np.array(ids), names


def load_sequences(path, cfg: Config) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    지원 형식
      1) .npz : X (N,T,F) 또는 (N,L,F), ids (N,), [feature_names]
      2) .csv/.parquet : 점 단위 long format
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=True)
        X = z["X"].astype(np.float32)
        ids = np.array([str(i) for i in z["ids"]])
        names = list(z["feature_names"]) if "feature_names" in z.files else [f"f{i}" for i in range(X.shape[2])]
        if X.shape[1] != cfg.seq_len:
            X = np.stack([_resample(np.nan_to_num(x), cfg.seq_len) for x in X]).astype(np.float32)
        return np.nan_to_num(X), ids, names
    pts = read_table(path)
    return long_to_sequences(pts, cfg)


# ---------------------------------------------------------------------------
# 분류모델 예측 결과
# ---------------------------------------------------------------------------
def load_predictions(path, cfg: Config) -> pd.DataFrame:
    c = cfg.cols
    p = read_table(path)
    key = c.unit_id if c.unit_id in p.columns else c.vessel_id
    keep = [key, c.pred_type] + ([c.pred_prob] if c.pred_prob in p.columns else [])
    p = p[keep].copy()
    p[key] = p[key].astype(str)
    p[c.pred_type] = p[c.pred_type].astype(str)
    if c.pred_prob not in p.columns:
        p[c.pred_prob] = 1.0
    return p.drop_duplicates(key), key
