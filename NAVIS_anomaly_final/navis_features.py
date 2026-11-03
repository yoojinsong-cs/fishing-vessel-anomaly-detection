# -*- coding: utf-8 -*-
"""
NAVIS 조업유형 분류 - 1. 데이터 전처리 및 레코드(행) 단위 특징 추출

[실행 방법]  VS Code 터미널에서 가상환경을 켠 뒤
    pip install pandas pyarrow shapely pyproj scikit-learn scipy joblib
    python navis_features.py

[처리 흐름]
    1.0 미터 좌표로 변환 (EPSG:4326 -> EPSG:5179)
        + GPS 튐(앞뒤 점과 멀리 떨어진 점) 제거, 비정상적으로 긴 스텝에서 조각 분리
    1.1 점간 거리(속력) 시퀀스 생성     - 중복 좌표는 지우지 않음(정지 시간 정보)
    1.2 거리 재샘플링 위치 시퀀스 생성   - 정지 구간을 하나의 점으로 합친 뒤 Δs 간격으로 interpolate
    1.3 속력 시퀀스 지표                 - 점 수가 최빈값 ±MODE_TOL 안인 행에서만 계산, 나머지는 NaN
    1.4 위치 시퀀스 지표                 - 스케일별 / 스케일 간 / 스케일 무관 지표

[2-pass 구조]
    pass 1 : 전체 데이터에서 표본을 뽑아 점 수 최빈값, 정지/고속 경계(GMM), 급가감속 기준을 정하고
             thresholds.json 에 저장 (테스트 데이터에도 같은 임계값을 써야 하므로 반드시 보관)
    pass 2 : 월별 CSV를 청크 단위로 읽어 특징을 계산하고 월별 parquet 로 저장한 뒤 하나로 합침
             (이미 끝난 달의 parquet 이 있으면 건너뛰므로 중간에 멈춰도 이어서 실행 가능)

[출력]  OUT_DIR
    features_<원본파일명>.parquet  월별 특징 테이블
    features.parquet               6개월 통합 특징 테이블 (원본의 geom 외 모든 컬럼 + 특징)
    thresholds.json                pass 1 에서 정한 임계값
"""

import os
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from scipy.ndimage import median_filter
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed


# =====================================================================
# 설정
# =====================================================================
# 경로는 navis_paths.py 에서 (환경 변수 > navis_paths.json > 레포 기본 구조). --data / --out-dir 로도 바꿀 수 있음
import navis_paths as _P
DATA_DIR = _P.RAW_DIR
DATA_PATHS = _P.raw_files(DATA_DIR) or [os.path.join(DATA_DIR, f"vpass_{m}월 데이터.csv") for m in range(1, 7)]

OUT_DIR = _P.FEAT_DIR
OUT_NAME = "features.parquet"
THRESH_NAME = "thresholds.json"

GEOM_COL = 'geom'           # MULTILINESTRING이 들어 있는 컬럼 (WKT, EWKT, WKB hex 모두 가능)
ID_COL = 'rfid_id'          # 선박 ID 컬럼 (선박 단위 집계·GroupKFold용, 특징으로 쓰지 않음)
LABEL_COL = 'fishing_code'  # 업종 라벨 컬럼 (테스트 데이터처럼 없으면 경고만 하고 진행)
KEEP_COLS = None            # 결과에 남길 원본 컬럼. None이면 geom을 뺀 모든 컬럼을 그대로 남김
                            # (모두 문자열로 저장됨. 일자·시간대는 이후 단계에서 숫자로 변환)

CFG = dict(
    # ----- 1.0 좌표 변환 -----
    SRC_CRS='EPSG:4326',
    DST_CRS='EPSG:5179',
    SWAP_LATLON='auto',      # 좌표가 (위도, 경도) 순서면 True. 'auto'는 값 범위로 판단
    JUMP_M=3000.0,           # 앞뒤 점 모두와 이보다 멀고, 앞뒤 점끼리는 가까우면 GPS 튐으로 보고 제거
    MAX_STEP_M=10000.0,      # 연속 두 점이 이보다 멀면 신호 단절로 보고 조각을 나눔

    # ----- 1.2 재샘플링 -----
    SCALES=[25, 100, 400],   # 재샘플링 간격 Δs [m]
    MIN_GAP_M=5.0,           # 직전에 남긴 점과 이보다 가까운 점은 합침 (거의 정지한 구간의 회전각 노이즈 제거)
    MIN_PIECE_LEN_M=50.0,    # 이보다 짧은 조각은 위치 지표 계산에서 제외

    # ----- 1.3 속력 지표 -----
    MODE_TOL=0.10,           # 점 수가 최빈값 ±10% 안인 행에서만 속력 지표 계산 (등간격 가정)
    STOP_THR=None,           # 정지/저속 경계 [m/스텝]. None이면 GMM으로 추정
    FAST_THR=None,           # 저속/고속 경계 [m/스텝]. None이면 GMM으로 추정
    ACC_THR=None,            # 급가속·급감속 기준 |Δd|. None이면 전체 |Δd|의 p90
    STATE_SMOOTH=3,          # 정지·조업·이동 상태를 정할 때 쓰는 이동 중앙값 창 크기
    ACF_LAG_K=5,             # lag-k 자기상관의 k
    ZERO_EPS_M=0.01,         # 이 거리 이하는 '거리 0'(중복 좌표)으로 봄

    # ----- 1.4 위치 지표 -----
    UTURN_WIN_M=300.0,       # U턴 판정 창 길이 [m]
    UTURN_DEG=150.0,         # 창 안 누적 회전이 이 값 이상이면 U턴 (≈180°)
    SIGN_EPS_DEG=1.0,        # 이보다 작은 회전각은 부호(좌/우) 판정에서 제외
    LOCAL_M=5,               # 국소 직진성 슬라이딩 창의 점 수 m
    DP_EPS_K=0.5,            # Douglas-Peucker ε = DP_EPS_K × Δs
    GRID_M=200.0,            # 격자 재방문율의 격자 크기 [m]
    HEAD_BINS=12,            # 방향 엔트로피 계산용 방향 구간 수 (30°씩)
    STOP_RADIUS_M=30.0,      # 정지 지점: 이 반경 안에 머문 연속 점들을 한 지점으로 합침
    STOP_MIN_PTS=3,          # 정지 지점으로 인정할 최소 점 수
    SELFX_MAX_SEG=3000,      # 자기교차 계산 시 선분이 이보다 많으면 더 큰 스케일의 재샘플링 점 사용

    # ----- pass 1 (임계값 추정) -----
    PASS1_FRAC=0.05,         # CSV 청크마다 이 비율만큼 표본 추출
    PASS1_MAX_RECORDS=100_000,  # 임계값 추정에 쓸 최대 레코드 수
    PASS1_CHUNKS_PER_FILE=None, # 파일당 읽을 최대 청크 수. None이면 전체 (빠르게 하려면 3 정도)
    REUSE_THRESH=False,      # True면 OUT_DIR의 thresholds.json을 읽어 그대로 사용 (테스트 데이터 처리 시)

    # ----- 실행 -----
    CSV_CHUNK_ROWS=100_000,  # CSV를 한 번에 읽는 행 수 (메모리가 부족하면 줄이기)
    N_JOBS=6,                # 병렬 작업 수. RAM 16GB 기준 6부터 시작해 작업 관리자 보며 조절
    CHUNK=300,               # 한 작업이 처리할 레코드 수
)


# =====================================================================
# 특징 이름 (모든 행이 같은 컬럼을 갖도록 미리 정의)
# =====================================================================
STATE_NAMES = ['s', 'f', 'm']  # stop(정지), fish(조업/저속), move(이동/고속)

SPEED_KEYS = (
    [f'spd_d_{s}' for s in ['mean', 'median', 'std', 'cv', 'p10', 'p25', 'p75', 'p90', 'iqr', 'max']]
    + ['spd_bin_stop', 'spd_bin_slow', 'spd_bin_fast']
    + ['spd_st_stop', 'spd_st_fish', 'spd_st_move']
    + [f'spd_tr_{a}{b}' for a in STATE_NAMES for b in STATE_NAMES] + ['spd_tr_rate']
    + ['spd_run_stop_n', 'spd_run_stop_mean', 'spd_run_stop_max', 'spd_run_fish_mean', 'spd_run_move_mean']
    + ['spd_dd_abs_mean', 'spd_dd_abs_std', 'spd_n_acc', 'spd_n_dec']
    + ['spd_acf_lag1', 'spd_acf_lagk']
    + ['spd_fft_peak_freq', 'spd_fft_peak_share']
    + ['spd_trend_slope', 'spd_half_ratio']
    + ['spd_zero_ratio', 'spd_zero_run_max']
)

SCALE_KEYS = [
    'th_mean', 'th_median', 'th_p90', 'th_std',
    'th_b0_15', 'th_b15_45', 'th_b45_90', 'th_b90p',
    'kappa_p50', 'kappa_p90',
    'sharp_per_km', 'uturn_per_km',
    'sign_change_rate', 'same_sign_max_m',
    'net_turn_per_km', 'loops',
    'R1', 'R2', 'R2_minus_R1', 'head_entropy',
    'loc_str_mean', 'loc_str_min', 'loc_str_p10',
    'dp_vert_per_km',
]

CROSS_KEYS = ['frac_dim', 'turn_ratio_fine_coarse', 'jitter_index']

GEO_KEYS = [
    'L_total', 'D_net', 'straightness', 'Rg',
    'hull_area', 'hull_perim', 'compactness', 'mrr_long', 'mrr_short', 'mrr_ratio', 'pca_linearity',
    'selfx_per_km', 'grid_revisit',
    'stop_n', 'stop_gap_mean', 'stop_gap_cv', 'stop_hull_area',
]

QUALITY_KEYS = ['n_points_raw', 'n_parts_raw', 'n_spikes', 'n_splits',
                'n_pieces_short', 'valid_len_ratio', 'spd_valid']


def sfx(ds):
    ds = float(ds)
    return f"_{int(ds)}" if ds.is_integer() else f"_{ds:g}"


def feature_names(cfg):
    scale_cols = [k + sfx(ds) for ds in sorted(cfg['SCALES']) for k in SCALE_KEYS]
    return QUALITY_KEYS + SPEED_KEYS + scale_cols + CROSS_KEYS + GEO_KEYS


# =====================================================================
# 기본 기하 함수
# =====================================================================
def seglen(p):
    """연속 점 사이 거리 배열 (len = n-1)"""
    if len(p) < 2:
        return np.zeros(0)
    v = np.diff(p, axis=0)
    return np.hypot(v[:, 0], v[:, 1])


def pathlen(p):
    return float(seglen(p).sum())


def wrap_angle(a):
    """라디안 각도를 -π ~ π 로"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_lengths(a):
    """같은 값이 이어지는 구간의 (값, 길이)"""
    a = np.asarray(a)
    if len(a) == 0:
        return a, np.zeros(0, dtype=int)
    cut = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.concatenate([[0], cut])
    lens = np.diff(np.concatenate([starts, [len(a)]]))
    return a[starts], lens


def safe_div(a, b):
    return a / b if (b is not None and np.isfinite(b) and b > 0) else np.nan


# =====================================================================
# 1.0 데이터 읽기 · 좌표 변환 · 정제
# =====================================================================
def detect_encoding(path):
    """CSV 앞부분으로 utf-8 / cp949 판별"""
    with open(path, 'rb') as f:
        raw = f.read(1 << 20)
    nl = raw.rfind(b'\n')
    if nl > 0:
        raw = raw[:nl + 1]
    for enc in ('utf-8-sig', 'cp949'):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            pass
    return 'cp949'


def read_chunks(path, enc, usecols=None):
    return pd.read_csv(path, chunksize=CFG['CSV_CHUNK_ROWS'], encoding=enc,
                       dtype=str, usecols=usecols, low_memory=False)


def parse_geoms(s):
    """WKT / EWKT / WKB hex 문자열 시리즈 -> shapely 객체 배열 (읽기 실패는 None)"""
    arr = s.astype(object).where(s.notna(), '').astype(str).str.strip()
    arr = arr.str.replace(r'^SRID=\d+;', '', regex=True)
    out = np.full(len(arr), None, dtype=object)
    nonempty = (arr.str.len() > 0).to_numpy()
    is_hex = arr.str.fullmatch(r'[0-9A-Fa-f]+').fillna(False).to_numpy(dtype=bool) & nonempty
    is_wkt = nonempty & ~is_hex
    if is_wkt.any():
        out[is_wkt] = shapely.from_wkt(arr[is_wkt].to_numpy(dtype=object), on_invalid='ignore')
    if is_hex.any():
        out[is_hex] = shapely.from_wkb(arr[is_hex].to_numpy(dtype=object), on_invalid='ignore')
    return out


def records_to_parts(geoms, transformer, swap):
    """레코드별로 [조각1 (n,2) 미터 좌표, 조각2, ...] 리스트를 만든다 (좌표 변환은 한 번에 벡터 연산)"""
    n = len(geoms)
    out = [[] for _ in range(n)]
    ok = ~shapely.is_missing(geoms)
    idx_ok = np.flatnonzero(ok)
    if len(idx_ok) == 0:
        return out
    parts, pidx = shapely.get_parts(geoms[idx_ok], return_index=True)
    if len(parts) == 0:
        return out
    coords, cidx = shapely.get_coordinates(parts, return_index=True)
    if len(coords) == 0:
        return out
    if swap:
        coords = coords[:, ::-1]
    x, y = transformer.transform(coords[:, 0], coords[:, 1])
    xy = np.column_stack([x, y])
    cut = np.flatnonzero(np.diff(cidx)) + 1
    starts = np.concatenate([[0], cut])
    part_ids = cidx[starts]
    rec_ids = idx_ok[pidx[part_ids]]
    for r, piece in zip(rec_ids, np.split(xy, cut)):
        out[r].append(piece)
    return out


def clean_part(p, jump_m, max_step_m):
    """GPS 튐 제거 + 큰 스텝에서 조각 분리. 중복 좌표는 남긴다(정지 정보).
    반환: (조각 리스트, 제거한 튐 수, 분리 횟수)"""
    p = p[np.isfinite(p).all(axis=1)]
    n_spike = 0
    if len(p) >= 3:
        d_prev = np.hypot(*(p[1:-1] - p[:-2]).T)
        d_next = np.hypot(*(p[2:] - p[1:-1]).T)
        d_skip = np.hypot(*(p[2:] - p[:-2]).T)
        spike = (d_prev > jump_m) & (d_next > jump_m) & (d_skip < 0.5 * np.minimum(d_prev, d_next))
        if spike.any():
            keep = np.ones(len(p), dtype=bool)
            keep[1:-1] = ~spike
            n_spike = int(spike.sum())
            p = p[keep]
    if len(p) >= 2:
        cut = np.flatnonzero(seglen(p) > max_step_m) + 1
        pieces = np.split(p, cut) if len(cut) else [p]
        n_split = len(cut)
    else:
        pieces, n_split = [p], 0
    return [q for q in pieces if len(q) > 0], n_spike, n_split


def collapse(p, radius):
    """직전에 남긴 점(앵커)과 radius 미만인 점들을 하나의 군집으로 합쳐 중심점으로 바꾼다.
    반환: (중심점 배열, 군집별 점 수)"""
    if len(p) == 0:
        return p, np.zeros(0, dtype=int)
    cents, sizes = [], []
    anchor = p[0]
    sx, sy, cnt = p[0, 0], p[0, 1], 1
    for q in p[1:]:
        if np.hypot(q[0] - anchor[0], q[1] - anchor[1]) < radius:
            sx += q[0]; sy += q[1]; cnt += 1
        else:
            cents.append((sx / cnt, sy / cnt)); sizes.append(cnt)
            anchor = q
            sx, sy, cnt = q[0], q[1], 1
    cents.append((sx / cnt, sy / cnt)); sizes.append(cnt)
    return np.array(cents), np.array(sizes)


def resample(p, ds):
    """선을 따라 Δs 간격으로 점을 다시 찍는다 (0, Δs, 2Δs, ...). 길이가 Δs 미만이면 시작점 1개."""
    sl = seglen(p)
    keep = np.concatenate([[True], sl > 0])
    p = p[keep]
    if len(p) < 2:
        return p[:1]
    cum = np.concatenate([[0.0], np.cumsum(seglen(p))])
    n = int(cum[-1] // ds)
    t = np.arange(n + 1) * ds
    return np.column_stack([np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])])


# =====================================================================
# 1.3 속력 시퀀스 지표
# =====================================================================
def pooled_acf(steps, mu, var, k):
    """조각 경계를 넘지 않는 lag-k 자기상관"""
    num, cnt = 0.0, 0
    for s in steps:
        if len(s) > k:
            x = s - mu
            num += float(np.dot(x[:-k], x[k:]))
            cnt += len(s) - k
    return num / cnt / var if (cnt > 0 and var > 0) else np.nan


def speed_features(steps, thr, cfg):
    """steps: 조각별 점간 거리 배열 리스트"""
    f = {}
    steps = [s for s in steps if len(s) > 0]
    if not steps:
        return f
    d = np.concatenate(steps)
    if len(d) < 2:
        return f

    # 분포 요약
    p10, p25, p50, p75, p90 = np.percentile(d, [10, 25, 50, 75, 90])
    mu, sd = float(d.mean()), float(d.std())
    f.update(spd_d_mean=mu, spd_d_median=p50, spd_d_std=sd, spd_d_cv=safe_div(sd, mu),
             spd_d_p10=p10, spd_d_p25=p25, spd_d_p75=p75, spd_d_p90=p90,
             spd_d_iqr=p75 - p25, spd_d_max=float(d.max()))

    # 구간 비율 (원 시퀀스)
    bins = [thr['stop_thr'], thr['fast_thr']]
    raw_state = np.digitize(d, bins)
    f['spd_bin_stop'] = float(np.mean(raw_state == 0))
    f['spd_bin_slow'] = float(np.mean(raw_state == 1))
    f['spd_bin_fast'] = float(np.mean(raw_state == 2))

    # 상태 (이동 중앙값으로 평활한 시퀀스)
    w = int(cfg['STATE_SMOOTH'])
    states = []
    for s in steps:
        sm = median_filter(s, size=w, mode='nearest') if (w > 1 and len(s) > 1) else s
        states.append(np.digitize(sm, bins))
    sall = np.concatenate(states)
    f['spd_st_stop'] = float(np.mean(sall == 0))
    f['spd_st_fish'] = float(np.mean(sall == 1))
    f['spd_st_move'] = float(np.mean(sall == 2))

    # 상태 전이 (조각 안에서만)
    T = np.zeros((3, 3))
    for s in states:
        if len(s) >= 2:
            np.add.at(T, (s[:-1], s[1:]), 1)
    rows = T.sum(axis=1)
    for a in range(3):
        for b in range(3):
            f[f'spd_tr_{STATE_NAMES[a]}{STATE_NAMES[b]}'] = safe_div(T[a, b], rows[a])
    f['spd_tr_rate'] = (T.sum() - np.trace(T)) / len(d)

    # 연속 길이 (run-length, 단위: 스텝 수)
    run_v, run_l = [], []
    for s in states:
        v, l = run_lengths(s)
        run_v.append(v); run_l.append(l)
    run_v, run_l = np.concatenate(run_v), np.concatenate(run_l)
    stop_runs = run_l[run_v == 0]
    f['spd_run_stop_n'] = float(len(stop_runs))
    f['spd_run_stop_mean'] = float(stop_runs.mean()) if len(stop_runs) else np.nan
    f['spd_run_stop_max'] = float(stop_runs.max()) if len(stop_runs) else 0.0
    fish_runs, move_runs = run_l[run_v == 1], run_l[run_v == 2]
    f['spd_run_fish_mean'] = float(fish_runs.mean()) if len(fish_runs) else np.nan
    f['spd_run_move_mean'] = float(move_runs.mean()) if len(move_runs) else np.nan

    # 변화량 (조각 안에서만)
    dd = [np.diff(s) for s in steps if len(s) >= 2]
    if dd:
        dd = np.concatenate(dd)
        f['spd_dd_abs_mean'] = float(np.abs(dd).mean())
        f['spd_dd_abs_std'] = float(np.abs(dd).std())
        f['spd_n_acc'] = float(np.sum(dd > thr['acc_thr']))
        f['spd_n_dec'] = float(np.sum(dd < -thr['acc_thr']))

    # 시계열 구조
    var = float(d.var())
    f['spd_acf_lag1'] = pooled_acf(steps, mu, var, 1)
    f['spd_acf_lagk'] = pooled_acf(steps, mu, var, int(cfg['ACF_LAG_K']))
    longest = max(steps, key=len)
    if len(longest) >= 8:
        x = longest - longest.mean()
        power = np.abs(np.fft.rfft(x)) ** 2
        power = power[1:]  # 직류 성분 제외
        tot = power.sum()
        if tot > 0:
            k = int(np.argmax(power))
            f['spd_fft_peak_freq'] = (k + 1) / len(longest)  # 단위: 주기/스텝
            f['spd_fft_peak_share'] = float(power[k] / tot)

    # 추세
    if len(d) >= 3:
        f['spd_trend_slope'] = float(np.polyfit(np.arange(len(d)), d, 1)[0])
    h = len(d) // 2
    f['spd_half_ratio'] = safe_div(float(d[h:].mean()), float(d[:h].mean()))

    # 정지 관련 (거리 0 = 중복 좌표)
    eps = cfg['ZERO_EPS_M']
    f['spd_zero_ratio'] = float(np.mean(d <= eps))
    zmax = 0
    for s in steps:
        v, l = run_lengths(s <= eps)
        if np.any(v):
            zmax = max(zmax, int(l[v].max()))
    f['spd_zero_run_max'] = float(zmax)
    return f


# =====================================================================
# 1.4 위치 시퀀스 지표
# =====================================================================
def count_uturns(theta_deg, w, thr_deg):
    """w개 연속 회전각의 합이 thr 이상이면 U턴 1회 (겹치지 않게 셈)"""
    n = len(theta_deg)
    if n == 0:
        return 0
    w = min(w, n)
    cs = np.concatenate([[0.0], np.cumsum(theta_deg)])
    win = np.abs(cs[w:] - cs[:-w])
    i, c = 0, 0
    while i < len(win):
        if win[i] >= thr_deg:
            c += 1
            i += w
        else:
            i += 1
    return c


def scale_features(rs_list, ds, cfg):
    """rs_list: 재샘플링된 조각들 (각 3점 이상). 회전각은 조각 경계를 넘지 않는다."""
    f = {}
    s = sfx(ds)
    thetas, heads, locs = [], [], []
    L = 0.0
    n_ut = n_ch = n_pair = 0
    max_run_pts = 0
    net = 0.0
    dp_vert = 0
    w = max(1, int(round(cfg['UTURN_WIN_M'] / ds)))
    m = int(cfg['LOCAL_M'])
    for r in rs_list:
        v = np.diff(r, axis=0)
        sl = np.hypot(v[:, 0], v[:, 1])
        L += sl.sum()
        h = np.arctan2(v[:, 1], v[:, 0])
        heads.append(h)
        th = np.degrees(wrap_angle(np.diff(h)))
        thetas.append(th)
        n_ut += count_uturns(th, w, cfg['UTURN_DEG'])
        net += abs(th.sum())
        # 회전 방향성
        t = th[np.abs(th) >= cfg['SIGN_EPS_DEG']]
        if len(t) >= 1:
            sg = np.sign(t)
            n_ch += int(np.sum(sg[1:] != sg[:-1]))
            n_pair += len(t) - 1
            _, l = run_lengths(sg)
            max_run_pts = max(max_run_pts, int(l.max()))
        # 국소 직진성
        if len(r) >= m:
            cs = np.concatenate([[0.0], np.cumsum(sl)])
            Lw = cs[m - 1:] - cs[:len(cs) - m + 1]
            Dw = np.hypot(*(r[m - 1:] - r[:len(r) - m + 1]).T)
            locs.append(Dw[Lw > 0] / Lw[Lw > 0])
        # Douglas-Peucker 단순화
        simp = shapely.simplify(shapely.linestrings(r), cfg['DP_EPS_K'] * ds, preserve_topology=False)
        dp_vert += int(shapely.get_num_coordinates(simp))

    th = np.concatenate(thetas)
    a = np.abs(th)
    km = L / 1000.0
    if len(a) == 0 or km <= 0:
        return f
    f['th_mean' + s] = float(a.mean())
    f['th_median' + s] = float(np.median(a))
    f['th_p90' + s] = float(np.percentile(a, 90))
    f['th_std' + s] = float(a.std())
    f['th_b0_15' + s] = float(np.mean(a < 15))
    f['th_b15_45' + s] = float(np.mean((a >= 15) & (a < 45)))
    f['th_b45_90' + s] = float(np.mean((a >= 45) & (a < 90)))
    f['th_b90p' + s] = float(np.mean(a >= 90))
    kappa = np.radians(a) / ds  # [rad/m]
    f['kappa_p50' + s] = float(np.percentile(kappa, 50))
    f['kappa_p90' + s] = float(np.percentile(kappa, 90))
    f['sharp_per_km' + s] = float(np.sum(a > 90) / km)
    f['uturn_per_km' + s] = n_ut / km
    f['sign_change_rate' + s] = n_ch / n_pair if n_pair > 0 else np.nan
    f['same_sign_max_m' + s] = max_run_pts * ds
    f['net_turn_per_km' + s] = net / km  # [deg/km]
    f['loops' + s] = net / 360.0
    hh = np.concatenate(heads)
    R1 = float(np.hypot(np.cos(hh).mean(), np.sin(hh).mean()))
    R2 = float(np.hypot(np.cos(2 * hh).mean(), np.sin(2 * hh).mean()))
    f['R1' + s], f['R2' + s], f['R2_minus_R1' + s] = R1, R2, R2 - R1
    B = int(cfg['HEAD_BINS'])
    cnt, _ = np.histogram(np.mod(hh, 2 * np.pi), bins=B, range=(0, 2 * np.pi))
    p = cnt[cnt > 0] / cnt.sum()
    f['head_entropy' + s] = float(-(p * np.log(p)).sum() / np.log(B))
    if locs:
        loc = np.concatenate(locs)
        if len(loc):
            f['loc_str_mean' + s] = float(loc.mean())
            f['loc_str_min' + s] = float(loc.min())
            f['loc_str_p10' + s] = float(np.percentile(loc, 10))
    f['dp_vert_per_km' + s] = dp_vert / km
    return f


def count_self_intersections(parts, block=256):
    """여러 조각의 선분끼리 교차하는 쌍의 수 (같은 조각의 이웃 선분은 제외)"""
    segs, owner, pos = [], [], []
    for k, p in enumerate(parts):
        if len(p) < 2:
            continue
        segs.append(np.hstack([p[:-1], p[1:]]))
        owner.append(np.full(len(p) - 1, k))
        pos.append(np.arange(len(p) - 1))
    if not segs:
        return 0
    S = np.vstack(segs); owner = np.concatenate(owner); pos = np.concatenate(pos)
    n = len(S)
    if n < 2:
        return 0

    def orient(P, Q, R):
        return (Q[..., 0] - P[..., 0]) * (R[..., 1] - P[..., 1]) - (Q[..., 1] - P[..., 1]) * (R[..., 0] - P[..., 0])

    total = 0
    for st in range(0, n - 1, block):
        ii = np.arange(st, min(st + block, n - 1))
        jj = np.arange(st + 1, n)
        A = S[ii, None, :2]; Bp = S[ii, None, 2:]
        C = S[None, jj, :2]; D = S[None, jj, 2:]
        o1 = orient(A, Bp, C); o2 = orient(A, Bp, D)
        o3 = orient(C, D, A); o4 = orient(C, D, Bp)
        cross = (o1 * o2 < 0) & (o3 * o4 < 0)
        valid = jj[None, :] > ii[:, None]
        adj = (owner[ii][:, None] == owner[jj][None, :]) & (pos[jj][None, :] - pos[ii][:, None] == 1)
        total += int(np.sum(cross & valid & ~adj))
    return total


def position_features(cleaned, cfg):
    f = {}
    # 정지 지점 (정제된 원래 점 기준, 모든 조각)
    stops = []
    for p in cleaned:
        c, sz = collapse(p, cfg['STOP_RADIUS_M'])
        if len(c):
            stops.append(c[sz >= cfg['STOP_MIN_PTS']])
    stops = np.vstack(stops) if stops else np.zeros((0, 2))
    f['stop_n'] = float(len(stops))
    if len(stops) >= 2:
        g = seglen(stops)
        f['stop_gap_mean'] = float(g.mean())
        f['stop_gap_cv'] = safe_div(float(g.std()), float(g.mean()))
    if len(stops) >= 3:
        f['stop_hull_area'] = float(shapely.area(shapely.convex_hull(shapely.multipoints(stops))))
    else:
        f['stop_hull_area'] = 0.0

    # 1.2 정지 구간 합치기 → 유효 조각 선택
    collapsed = [collapse(p, cfg['MIN_GAP_M'])[0] for p in cleaned]
    lens = [pathlen(c) for c in collapsed]
    valid = [i for i, (c, L) in enumerate(zip(collapsed, lens))
             if len(c) >= 2 and L >= cfg['MIN_PIECE_LEN_M']]
    f['n_pieces_short'] = float(len(collapsed) - len(valid))
    total_len = sum(lens)
    vlen = sum(lens[i] for i in valid)
    f['valid_len_ratio'] = safe_div(vlen, total_len)
    if not valid:
        return f
    V = [collapsed[i] for i in valid]
    Vraw = [cleaned[i] for i in valid]

    # 스케일별 재샘플링 및 지표
    scales = sorted(float(x) for x in cfg['SCALES'])
    rs_good, Ldiv, mean_abs = {}, {}, {}
    for ds in scales:
        rs = [resample(v, ds) for v in V]
        # divider 길이: 재샘플링 점 + 끝점을 잇는 현(chord) 길이의 합
        Ldiv[ds] = sum(pathlen(np.vstack([r, v[-1:]])) for r, v in zip(rs, V))
        good = [r for r in rs if len(r) >= 3]  # 점 3개 미만이면 회전각 계산 불가 → NaN
        rs_good[ds] = good
        if good:
            sf = scale_features(good, ds, cfg)
            f.update(sf)
            mean_abs[ds] = sf.get('th_mean' + sfx(ds), np.nan)

    # 스케일 간 비교
    ok = [ds for ds in scales if Ldiv[ds] > 0]
    if len(ok) >= 2:
        slope = np.polyfit(np.log(ok), np.log([Ldiv[ds] for ds in ok]), 1)[0]
        f['frac_dim'] = float(1.0 - slope)
    lo, hi = scales[0], scales[-1]
    if lo in mean_abs and hi in mean_abs:
        f['turn_ratio_fine_coarse'] = safe_div(mean_abs[lo], mean_abs[hi])
    f['jitter_index'] = safe_div(sum(pathlen(p) for p in Vraw), Ldiv[lo])

    # 스케일 무관 지표
    f['L_total'] = vlen
    f['D_net'] = float(np.hypot(*(V[-1][-1] - V[0][0])))
    f['straightness'] = safe_div(f['D_net'], vlen)
    # 점 간격에 영향받지 않도록 가장 작은 스케일의 재샘플링 점 사용
    pts_fine = np.vstack(rs_good[lo]) if rs_good[lo] else np.vstack(V)
    allV = np.vstack(V)
    c = pts_fine.mean(axis=0)
    f['Rg'] = float(np.sqrt(np.mean(np.sum((pts_fine - c) ** 2, axis=1))))
    mp = shapely.multipoints(allV)
    hull = shapely.convex_hull(mp)
    A, P = float(shapely.area(hull)), float(shapely.length(hull))
    f['hull_area'] = A
    f['hull_perim'] = P
    f['compactness'] = safe_div(4 * np.pi * A, P ** 2) if A > 0 else np.nan
    rect = shapely.minimum_rotated_rectangle(mp)
    rc = shapely.get_coordinates(rect)
    if shapely.get_type_id(rect) == 3 and len(rc) >= 3:  # Polygon
        e1 = float(np.hypot(*(rc[1] - rc[0]))); e2 = float(np.hypot(*(rc[2] - rc[1])))
        f['mrr_long'], f['mrr_short'] = max(e1, e2), min(e1, e2)
    else:  # 일직선 또는 한 점
        f['mrr_long'], f['mrr_short'] = float(shapely.length(rect)), 0.0
    f['mrr_ratio'] = safe_div(f['mrr_short'], f['mrr_long'])
    if len(pts_fine) >= 3:
        ev = np.linalg.eigvalsh(np.cov(pts_fine.T))
        f['pca_linearity'] = safe_div(float(ev.max()), float(ev.sum()))

    # 자기교차
    sx_parts = V
    if sum(len(v) - 1 for v in V) > cfg['SELFX_MAX_SEG']:
        for ds in scales[1:]:
            sx_parts = rs_good[ds] if rs_good[ds] else sx_parts
            if sum(len(v) - 1 for v in sx_parts) <= cfg['SELFX_MAX_SEG']:
                break
    f['selfx_per_km'] = count_self_intersections(sx_parts) / (vlen / 1000.0)

    # 격자 재방문율 (재샘플링 점이 지나간 격자 순서에서, 이미 지나간 격자로 되돌아온 비율)
    G = cfg['GRID_M']
    gp = pts_fine
    cell = np.floor(gp / G).astype(np.int64)
    key = cell[:, 0] * 4_000_003 + cell[:, 1]
    visits, _ = run_lengths(key)
    if len(visits):
        _, first = np.unique(visits, return_index=True)
        f['grid_revisit'] = float((len(visits) - len(first)) / len(visits))
    return f


# =====================================================================
# 레코드 하나의 전체 특징
# =====================================================================
def record_features(parts, thr, cfg, names):
    f = dict.fromkeys(names, np.nan)
    n_raw = int(sum(len(p) for p in parts))
    f['n_points_raw'] = float(n_raw)
    f['n_parts_raw'] = float(len(parts))
    if n_raw == 0:
        f['spd_valid'] = 0.0
        return f
    cleaned, n_sp, n_sl = [], 0, 0
    for p in parts:
        pcs, a, b = clean_part(p, cfg['JUMP_M'], cfg['MAX_STEP_M'])
        cleaned += pcs; n_sp += a; n_sl += b
    f['n_spikes'], f['n_splits'] = float(n_sp), float(n_sl)

    # 1.3 속력 지표: 점 수가 최빈값 근처인 행만
    valid = thr['mode_lo'] <= n_raw <= thr['mode_hi']
    f['spd_valid'] = float(valid)
    if valid:
        steps = [seglen(p) for p in cleaned if len(p) >= 2]
        f.update(speed_features(steps, thr, cfg))

    # 1.4 위치 지표
    f.update(position_features(cleaned, cfg))
    return f


def features_batch(parts_batch, thr, cfg, names):
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        return [record_features(p, thr, cfg, names) for p in parts_batch]


# =====================================================================
# pass 1: 임계값 추정
# =====================================================================
def decide_swap(path, enc):
    if CFG['SWAP_LATLON'] != 'auto':
        return bool(CFG['SWAP_LATLON'])
    s = pd.read_csv(path, nrows=2000, encoding=enc, dtype=str, usecols=[GEOM_COL])[GEOM_COL]
    g = parse_geoms(s)
    g = g[~shapely.is_missing(g)]
    xy = shapely.get_coordinates(g)
    if len(xy) == 0:
        raise ValueError('geom 컬럼에서 좌표를 읽지 못했습니다. GEOM_COL 과 데이터 형식을 확인하세요.')
    m0, m1 = np.median(xy[:, 0]), np.median(xy[:, 1])
    if abs(m0) > 1000 or abs(m1) > 1000:
        raise ValueError(f'좌표 값이 위경도 범위가 아닙니다 (중앙값 {m0:.1f}, {m1:.1f}). '
                         'SRC_CRS 를 데이터의 실제 좌표계로 바꾸세요.')
    if 100 <= m0 <= 150 and 20 <= m1 <= 50:
        swap = False
    elif 20 <= m0 <= 50 and 100 <= m1 <= 150:
        swap = True
    else:
        print(f'  [주의] 좌표 범위로 경도/위도 순서를 판단하지 못했습니다 (중앙값 {m0:.3f}, {m1:.3f}). '
              '(경도, 위도) 순서로 가정합니다.')
        swap = False
    print(f'  좌표 순서: {"(위도, 경도) → 뒤집어서 사용" if swap else "(경도, 위도)"}  '
          f'[중앙값 {m0:.3f}, {m1:.3f}]')
    return swap


def gmm_thresholds(d, seed=0):
    """log(1+d)에 3성분 GMM → 인접 성분이 바뀌는 지점을 정지/저속, 저속/고속 경계로"""
    x = np.log1p(d).reshape(-1, 1)
    gm = GaussianMixture(n_components=3, random_state=seed, n_init=3, reg_covar=1e-4).fit(x)
    order = np.argsort(gm.means_.ravel())
    rank = np.empty(3, dtype=int); rank[order] = np.arange(3)
    lo, hi = float(np.min(gm.means_)), float(np.max(gm.means_))
    grid = np.linspace(lo, hi, 4000).reshape(-1, 1)
    r = rank[gm.predict(grid)]
    if not (r >= 1).any() or not (r >= 2).any():
        raise RuntimeError('GMM 경계를 찾지 못함')
    t1 = float(grid[np.argmax(r >= 1), 0]); t2 = float(grid[np.argmax(r >= 2), 0])
    if t2 <= t1:
        raise RuntimeError('GMM 경계 순서 오류')
    return float(np.expm1(t1)), float(np.expm1(t2)), np.expm1(np.sort(gm.means_.ravel())).tolist()


def pass1(paths, encs, transformer, swap):
    rng = np.random.default_rng(0)
    counts, recs = [], []
    max_rec = CFG['PASS1_MAX_RECORDS']
    t0 = time.time()
    for path in paths:
        for ci, ch in enumerate(read_chunks(path, encs[path], usecols=[GEOM_COL])):
            if CFG['PASS1_CHUNKS_PER_FILE'] and ci >= CFG['PASS1_CHUNKS_PER_FILE']:
                break
            k = min(len(ch), max(1000, int(len(ch) * CFG['PASS1_FRAC'])))
            sub = ch.sample(n=k, random_state=int(rng.integers(1 << 31)))
            parts = records_to_parts(parse_geoms(sub[GEOM_COL]), transformer, swap)
            for p in parts:
                n = int(sum(len(q) for q in p))
                if n == 0:
                    continue
                counts.append(n)
                if len(recs) < max_rec:
                    cl = []
                    for q in p:
                        cl += clean_part(q, CFG['JUMP_M'], CFG['MAX_STEP_M'])[0]
                    recs.append((n, [seglen(q) for q in cl if len(q) >= 2]))
        print(f'  pass 1 | {os.path.basename(path)} 완료 (누적 표본 {len(counts):,}개, {time.time() - t0:.0f}초)')

    counts = np.array(counts)
    if len(counts) == 0:
        raise ValueError('pass 1 에서 유효한 레코드를 찾지 못했습니다.')
    mode = int(np.bincount(counts).argmax())
    lo, hi = mode * (1 - CFG['MODE_TOL']), mode * (1 + CFG['MODE_TOL'])

    band = [st for n, st in recs if lo <= n <= hi]
    d = np.concatenate([s for st in band for s in st]) if band else np.zeros(0)
    dd = np.concatenate([np.diff(s) for st in band for s in st if len(s) >= 2]) if band else np.zeros(0)
    if len(d) == 0:
        raise ValueError('최빈값 근처 레코드에서 점간 거리를 얻지 못했습니다. MODE_TOL 을 늘려 보세요.')
    # 거리 0(중복 좌표)은 정의상 정지이므로 GMM에서 제외.
    # 포함하면 성분 하나가 0 근처의 좁은 덩어리에 쓰여 정지 경계가 0에 붙어 버림
    d = d[d > CFG['ZERO_EPS_M']]
    if len(d) < 100:
        raise ValueError('거리가 0보다 큰 점간 거리가 너무 적어 임계값을 추정할 수 없습니다.')
    if len(d) > 500_000:
        d = rng.choice(d, 500_000, replace=False)

    gmm_means = None
    stop_thr, fast_thr = CFG['STOP_THR'], CFG['FAST_THR']
    if stop_thr is None or fast_thr is None:
        try:
            g1, g2, gmm_means = gmm_thresholds(d)
        except Exception as e:
            print(f'  [주의] GMM 경계 추정 실패({e}) → 분위수(33%, 67%)로 대체')
            g1, g2 = np.quantile(d, [1 / 3, 2 / 3]).tolist()
        stop_thr = g1 if stop_thr is None else stop_thr
        fast_thr = g2 if fast_thr is None else fast_thr
    acc_thr = CFG['ACC_THR'] if CFG['ACC_THR'] is not None else (
        float(np.percentile(np.abs(dd), 90)) if len(dd) else np.inf)

    share = float(np.mean((counts >= lo) & (counts <= hi)))
    thr = dict(mode_n=mode, mode_lo=lo, mode_hi=hi, mode_share=share,
               stop_thr=float(stop_thr), fast_thr=float(fast_thr), acc_thr=float(acc_thr),
               gmm_means=gmm_means, swap=bool(swap), n_sample_records=int(len(counts)))
    print(f'  점 수 최빈값 {mode}개 (속력 지표 계산 범위 {lo:.1f}~{hi:.1f}개, 전체의 {share:.1%})')
    print(f'  정지/저속 경계 {stop_thr:.1f} m/스텝, 저속/고속 경계 {fast_thr:.1f} m/스텝, '
          f'급가감속 기준 |Δd| > {acc_thr:.1f} m')
    return thr


# =====================================================================
# pass 2: 특징 추출 및 저장
# =====================================================================
FP_KEY = b'navis_fingerprint'


def fingerprint(thr, cfg):
    """임계값 + 특징 계산에 영향을 주는 설정. 이 값이 같을 때만 기존 월별 파일을 재사용"""
    t = {k: thr[k] for k in ('mode_lo', 'mode_hi', 'stop_thr', 'fast_thr', 'acc_thr', 'swap')}
    c = {k: v for k, v in cfg.items() if k not in ('N_JOBS', 'CHUNK', 'CSV_CHUNK_ROWS', 'REUSE_THRESH',
                                                   'PASS1_FRAC', 'PASS1_MAX_RECORDS', 'PASS1_CHUNKS_PER_FILE')}
    return json.dumps([t, c], sort_keys=True, default=str)


def build_schema(passthrough, names, fp):
    fields = [pa.field(c, pa.string()) for c in passthrough]
    fields += [pa.field('src_file', pa.string()), pa.field('row_in_file', pa.int64())]
    fields += [pa.field(c, pa.float32()) for c in names]
    return pa.schema(fields, metadata={FP_KEY: fp.encode('utf-8')})


def is_up_to_date(path, schema):
    try:
        s = pq.read_schema(path)
    except Exception:
        return False
    return (s.metadata or {}).get(FP_KEY) == schema.metadata[FP_KEY] and s.names == schema.names


def process_file(path, enc, out_file, passthrough, names, thr, transformer, schema, par):
    tmp = out_file + '.tmp'
    writer = pq.ParquetWriter(tmp, schema, compression='snappy')
    row0, t0 = 0, time.time()
    B = int(CFG['CHUNK'])
    try:
        for ci, ch in enumerate(read_chunks(path, enc)):
            parts = records_to_parts(parse_geoms(ch[GEOM_COL]), transformer, thr['swap'])
            batches = [parts[i:i + B] for i in range(0, len(parts), B)]
            res = par(delayed(features_batch)(b, thr, CFG, names) for b in batches)
            feats = pd.DataFrame([r for b in res for r in b], columns=names).astype('float32')
            meta = ch.reindex(columns=passthrough).reset_index(drop=True)
            meta['src_file'] = os.path.basename(path)
            meta['row_in_file'] = np.arange(row0, row0 + len(ch), dtype=np.int64)
            out = pd.concat([meta, feats], axis=1)
            writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))
            row0 += len(ch)
            print(f'  pass 2 | {os.path.basename(path)} | 청크 {ci + 1} 완료 '
                  f'(누적 {row0:,}행, {time.time() - t0:.0f}초)')
    finally:
        writer.close()
    os.replace(tmp, out_file)
    return row0


def combine(month_files, out_path, schema):
    tmp = out_path + '.tmp'
    with pq.ParquetWriter(tmp, schema, compression='snappy') as w:
        for mf in month_files:
            pf = pq.ParquetFile(mf)
            for i in range(pf.num_row_groups):
                w.write_table(pf.read_row_group(i).cast(schema))
    os.replace(tmp, out_path)


def main():
    global DATA_PATHS, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='*', help='CSV 경로 (지정하지 않으면 DATA_PATHS 사용)')
    ap.add_argument('--out-dir', help='출력 폴더 (지정하지 않으면 OUT_DIR 사용)')
    ap.add_argument('--n-jobs', type=int)
    args = ap.parse_args()
    if args.data:
        DATA_PATHS = args.data
    if args.out_dir:
        OUT_DIR = args.out_dir
    if args.n_jobs:
        CFG['N_JOBS'] = args.n_jobs

    t_all = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    missing = [p for p in DATA_PATHS if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('다음 파일을 찾을 수 없습니다:\n  ' + '\n  '.join(missing))

    print('[0] 파일 확인')
    encs = {}
    for p in DATA_PATHS:
        encs[p] = detect_encoding(p)
        print(f'  {os.path.getsize(p) / 1e9:6.2f} GB  인코딩 {encs[p]:9s}  {os.path.basename(p)}')

    header = list(pd.read_csv(DATA_PATHS[0], nrows=0, encoding=encs[DATA_PATHS[0]]).columns)
    print(f'  컬럼: {header}')
    if GEOM_COL not in header:
        raise KeyError(f'GEOM_COL "{GEOM_COL}" 컬럼이 없습니다. 위 컬럼 목록에서 골라 설정하세요.')
    for name, col in [('ID_COL', ID_COL), ('LABEL_COL', LABEL_COL)]:
        if col and col not in header:
            print(f'  [주의] {name} "{col}" 컬럼이 없습니다. 위 컬럼 목록을 보고 이름을 확인하세요.')
    passthrough = [c for c in (KEEP_COLS if KEEP_COLS else header) if c != GEOM_COL]

    transformer = Transformer.from_crs(CFG['SRC_CRS'], CFG['DST_CRS'], always_xy=True)
    thr_path = os.path.join(OUT_DIR, THRESH_NAME)

    print('[1] 임계값 (pass 1)')
    if CFG['REUSE_THRESH'] and os.path.exists(thr_path):
        with open(thr_path, encoding='utf-8') as fp:
            thr = json.load(fp)
        print(f'  {thr_path} 의 임계값을 그대로 사용합니다')
        print(f'  점 수 범위 {thr["mode_lo"]:.1f}~{thr["mode_hi"]:.1f}개, 정지/저속 {thr["stop_thr"]:.1f} m, '
              f'저속/고속 {thr["fast_thr"]:.1f} m, 급가감속 {thr["acc_thr"]:.1f} m')
    else:
        swap = decide_swap(DATA_PATHS[0], encs[DATA_PATHS[0]])
        thr = pass1(DATA_PATHS, encs, transformer, swap)
        with open(thr_path, 'w', encoding='utf-8') as fp:
            json.dump(dict(thr, cfg=CFG), fp, ensure_ascii=False, indent=2)
        print(f'  저장: {thr_path}')

    names = feature_names(CFG)
    schema = build_schema(passthrough, names, fingerprint(thr, CFG))
    print(f'[2] 특징 추출 (pass 2) - 특징 {len(names)}개, 병렬 작업 {CFG["N_JOBS"]}개')
    month_files = []
    with Parallel(n_jobs=CFG['N_JOBS'], backend='loky') as par:
        for p in DATA_PATHS:
            stem = os.path.splitext(os.path.basename(p))[0]
            mf = os.path.join(OUT_DIR, f'features_{stem}.parquet')
            month_files.append(mf)
            if os.path.exists(mf):
                if is_up_to_date(mf, schema):
                    print(f'  {os.path.basename(mf)} 이(가) 같은 설정으로 이미 만들어져 있어 건너뜁니다')
                    continue
                print(f'  {os.path.basename(mf)} 은(는) 임계값·설정이 달라 다시 만듭니다')
            process_file(p, encs[p], mf, passthrough, names, thr, transformer, schema, par)

    print('[3] 통합 저장')
    out_path = os.path.join(OUT_DIR, OUT_NAME)
    combine(month_files, out_path, schema)
    meta = pq.ParquetFile(out_path).metadata
    print(f'  {out_path}  ({meta.num_rows:,}행 × {meta.num_columns}열)')
    print(f'완료 ({(time.time() - t_all) / 60:.1f}분)')


if __name__ == '__main__':
    main()
