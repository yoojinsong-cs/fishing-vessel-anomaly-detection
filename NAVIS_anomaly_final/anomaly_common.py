# -*- coding: utf-8 -*-
"""NAVIS 이상탐지 - 공통 함수 (v2.5): 로딩, 시간 맥락, 그룹·선박 기준, 상태 희소성, 점수 결합, 이상 사건"""
import os
import glob
import json
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import anomaly_config as C

KEYS = ['src_file', 'row_in_file']
STATES = ['active', 'stationary']


def out(*names):
    os.makedirs(C.OUT_DIR, exist_ok=True)
    return os.path.join(C.OUT_DIR, *names)


def feature_files():
    files = sorted(glob.glob(os.path.join(C.FEAT_DIR, C.FEAT_GLOB)))
    if not files:
        raise FileNotFoundError(f'특징 파일이 없습니다: {os.path.join(C.FEAT_DIR, C.FEAT_GLOB)}')
    return files


def raw_files():
    import navis_paths
    files = navis_paths.raw_files(C.RAW_DIR)
    if not files:
        raise FileNotFoundError(f'원본 CSV가 없습니다: {os.path.join(C.RAW_DIR, C.RAW_GLOB)}')
    return files


def raw_feature_names():
    """특징 테이블에서 읽어야 하는 컬럼 (맥락 특징은 여기서 계산하므로 제외)"""
    ctx = set(C.CONTEXT_ACTIVE + C.CONTEXT_STATIONARY + ['gap_before_h', 'gap_after_h'])
    need = set(C.ACTIVE_FEATURES + C.STATIONARY_FEATURES + ['L_total', 'straightness', 'stop_n'])
    return sorted(need - ctx)


# ---------------------------------------------------------------------
# 특징 테이블 읽기 (월별로 필요한 컬럼만)
# ---------------------------------------------------------------------
def load_features(vessels=None):
    meta = [C.COL_DATE, C.COL_VESSEL, C.COL_HOUR, C.COL_LABEL] + KEYS
    frames = []
    for f in feature_files():
        names = pq.read_schema(f).names
        cols = meta + [c for c in raw_feature_names() if c in names]
        t = pq.read_table(f, columns=cols)
        t = t.filter(pc.is_in(t[C.COL_LABEL], value_set=pa.array(C.CLASSES)))   # 깨진 행 제거
        if vessels is not None:
            t = t.filter(pc.is_in(t[C.COL_VESSEL], value_set=pa.array(sorted(vessels))))
        d = t.to_pandas()
        for c in d.columns:
            if d[c].dtype.kind == 'f':
                d[c] = d[c].astype('float32')
        frames.append(d)
        print(f'  읽음: {os.path.basename(f)} ({len(d):,}행)', flush=True)
    df = pd.concat(frames, ignore_index=True)
    if os.path.exists(C.ENDPOINTS):                          # [D] 0단계의 시작·끝 좌표
        ep = pd.read_parquet(C.ENDPOINTS)
        df = df.merge(ep, on=KEYS, how='left')
    else:
        print('  [주의] endpoints.parquet 이 없어 모항 거리 특징을 쓰지 않습니다 (step0_endpoints.py 실행 필요)')
        for c in ['xs', 'ys', 'xe', 'ye']:
            df[c] = np.float32(np.nan)
    df['hour'] = pd.to_numeric(df[C.COL_HOUR], errors='coerce').astype('int16')
    df['date'] = pd.to_datetime(df[C.COL_DATE], format='%Y%m%d', errors='coerce')
    df['month'] = df['date'].dt.month.astype('int16')
    df['code'] = df[C.COL_LABEL].astype(str)
    df['vessel'] = df[C.COL_VESSEL].astype(str)
    df = df.drop(columns=[C.COL_HOUR, C.COL_DATE, C.COL_LABEL, C.COL_VESSEL])
    df['state'] = np.where(df['L_total'].notna(), 'active', 'stationary')
    df['t'] = (df['date'].values.astype('datetime64[h]').astype('int64') + df['hour'].values).astype('int64')
    return df


def group_key(df):
    g = df['code'].astype(str) + '|' + df['hour'].astype(str).str.zfill(2) + '|' + df['state'].astype(str)
    if C.GROUP_BY_MONTH:
        g = g + '|' + df['month'].astype(str).str.zfill(2)
    return g


# ---------------------------------------------------------------------
# [개선 1·5] 시간 맥락: 공백, 항해(연속 레코드), 연속 활동, 최근 3시간, 이동 중 끊김
# ---------------------------------------------------------------------
def _shift_in(arr, k, same):
    o = np.full(len(arr), np.nan, dtype='float64')
    if k < len(arr):
        o[k:] = arr[:-k]
    o[~same] = np.nan
    return o


def add_time_context(df):
    """선박·시간 순으로 정렬한 df를 돌려준다 (행 순서가 바뀜, 키 컬럼으로 다시 연결할 것)"""
    df = df.sort_values(['vessel', 't'], kind='mergesort').reset_index(drop=True)
    v = df['vessel'].values
    t = df['t'].values.astype('int64')
    same_prev = np.r_[False, v[1:] == v[:-1]]
    same_next = np.r_[v[:-1] == v[1:], False]
    # [개선 1] 선박의 첫·마지막 레코드는 공백을 알 수 없으므로 NaN (z 변환에서 중립값 0)
    gb = np.where(same_prev, np.r_[np.nan, np.diff(t)], np.nan)
    ga = np.where(same_next, np.r_[np.diff(t), np.nan], np.nan)
    df['gap_before_h'] = np.minimum(gb, C.GAP_CAP_H).astype('float32')
    df['gap_after_h'] = np.minimum(ga, C.GAP_CAP_H).astype('float32')

    # 항해: 같은 선박의 레코드가 TRIP_GAP_H 이하 간격으로 이어진 구간
    new_trip = ~(same_prev & (gb <= C.TRIP_GAP_H))
    trip = np.cumsum(new_trip)
    df['trip'] = trip.astype('int64')
    in_prev = ~new_trip                                  # 직전 행이 같은 항해
    in_next = np.r_[in_prev[1:], False]                  # 다음 행이 같은 항해
    df['trip_hour_idx'] = df.groupby('trip').cumcount().astype('float32')
    df['trip_len_h'] = df.groupby('trip')['trip'].transform('size').astype('float32')

    active = (df['state'].values == 'active')
    # 연속 활동/정박 시간 (항해 안에서, 같은 상태가 이어진 길이)
    block = np.cumsum(new_trip | np.r_[True, active[1:] != active[:-1]])
    run = df.groupby(block).cumcount().values + 1
    df['run_active_h'] = np.where(active, run, 0).astype('float32')
    df['run_stationary_h'] = np.where(~active, run, 0).astype('float32')
    df['prev_active'] = np.where(in_prev, np.r_[0, active[:-1]], 0).astype('float32')
    df['next_active'] = np.where(in_next, np.r_[active[1:], 0], 0).astype('float32')

    # 최근 3시간(현재 포함, 같은 항해 안) 평균·합
    for col, how, name in [('L_total', 'mean', 'roll3_L_total'), ('straightness', 'mean', 'roll3_straightness'),
                           ('stop_n', 'sum', 'roll3_stop_n')]:
        x0 = df[col].values.astype('float64')
        if col == 'L_total':
            x0 = np.nan_to_num(x0, nan=0.0)                # 정박 시간 이동 거리 = 0
        same1 = in_prev
        same2 = in_prev & np.r_[False, in_prev[:-1]]
        stack = np.vstack([x0, _shift_in(x0, 1, same1), _shift_in(x0, 2, same2)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            val = np.nanmean(stack, 0) if how == 'mean' else np.nansum(stack, 0)
        df[name] = val.astype('float32')

    # 움직이던 중 끊김: 항해의 마지막 행이 활동 상태이고 뒤에 공백이 있으면 그 공백 시간
    last_of_trip = ~in_next
    moving_end = last_of_trip & active
    # [v2.5-1] 선박의 마지막 기록은 다음 기록이 없어 공백을 알 수 없다 → 0 (v2.4까지는 상한 336시간으로 채워 가짜 단절 경보)
    df['gap_after_moving_h'] = np.where(moving_end, np.nan_to_num(df['gap_after_h'].values, nan=0.0), 0).astype('float32')
    return df



# ---------------------------------------------------------------------
# [D] 모항 위치 맥락: 선박별로 항해가 자주 시작·끝나는 지점 = 모항
# ---------------------------------------------------------------------
def fit_ports(df):
    """선박별 모항 (선박당 최대 2 × PORT_MAX_PER_VESSEL 곳). 두 가지 근거를 합친다.
      (a) 출항·귀항 지점: 항해 첫 행의 시작점과 마지막 행의 끝점. 출항과 귀항이 모두 있었고
          합쳐서 PORT_MIN_TRIPS 번 이상인 1km 격자 (끝점만 쌓이는 곳 = 늘 신호가 끊기는 어장일 수 있어 제외)
      (b) [v2.3-1] 오래 정박한 곳: 연속 정박이 PORT_MIN_RUN_H 시간에 이른 지점이 PORT_MIN_RUNS 번 이상인 격자
          정박 중에도 신호를 보내는 선박은 항구에서 기록이 끊기지 않아 (a)로는 항구를 못 찾는다 (4차 보고서 3.1).
          조업 중 정지는 몇 시간씩이라 8시간 연속 정박이 같은 곳에서 반복되는 경우는 항구로 본다.
    반환: DataFrame(vessel, x, y, source)"""
    G = C.PORT_GRID_M
    out = []
    first = df.groupby('trip').head(1)[['vessel', 'xs', 'ys']].rename(columns={'xs': 'x', 'ys': 'y'})
    last = df.groupby('trip').tail(1)[['vessel', 'xe', 'ye']].rename(columns={'xe': 'x', 'ye': 'y'})
    first['is_start'] = 1; last['is_start'] = 0
    pts = pd.concat([first, last]).dropna()
    if len(pts):
        pts['cx'] = np.floor(pts['x'] / G).astype('int64'); pts['cy'] = np.floor(pts['y'] / G).astype('int64')
        cnt = pts.groupby(['vessel', 'cx', 'cy']).agg(n=('x', 'size'), starts=('is_start', 'sum'),
                                                      x=('x', 'mean'), y=('y', 'mean')).reset_index()
        cnt['ends'] = cnt['n'] - cnt['starts']
        cnt = cnt[(cnt['n'] >= C.PORT_MIN_TRIPS) & (cnt['starts'] > 0) & (cnt['ends'] > 0)]
        cnt = cnt.sort_values(['vessel', 'n'], ascending=[True, False]).groupby('vessel').head(C.PORT_MAX_PER_VESSEL)
        out.append(cnt.assign(source='trip')[['vessel', 'cx', 'cy', 'x', 'y', 'source']])
    if 'run_stationary_h' in df.columns:
        st = df[df['run_stationary_h'].values == C.PORT_MIN_RUN_H]          # 긴 정박마다 한 번씩
        x = st['xs'].fillna(st['xe']).values; y = st['ys'].fillna(st['ye']).values
        st = pd.DataFrame({'vessel': st['vessel'].values, 'x': x, 'y': y}).dropna()
        if len(st):
            st['cx'] = np.floor(st['x'] / G).astype('int64'); st['cy'] = np.floor(st['y'] / G).astype('int64')
            cs = st.groupby(['vessel', 'cx', 'cy']).agg(n=('x', 'size'), x=('x', 'mean'), y=('y', 'mean')).reset_index()
            cs = cs[cs['n'] >= C.PORT_MIN_RUNS]
            cs = cs.sort_values(['vessel', 'n'], ascending=[True, False]).groupby('vessel').head(C.PORT_MAX_PER_VESSEL)
            out.append(cs.assign(source='stay')[['vessel', 'cx', 'cy', 'x', 'y', 'source']])
    if not out:
        return pd.DataFrame(columns=['vessel', 'x', 'y', 'source'])
    P = pd.concat(out, ignore_index=True)
    P = P.groupby(['vessel', 'cx', 'cy'], as_index=False).agg(x=('x', 'mean'), y=('y', 'mean'),
                                                              source=('source', lambda s: '+'.join(sorted(set(s)))))
    return P[['vessel', 'x', 'y', 'source']]


def port_distance_km(vessels, x, y, ports):
    """[v2.2] 각 행에서 '그 선박의' 가장 가까운 모항까지 거리(km).
    모항을 찾지 못한 선박은 모든 선박의 모항 중 가장 가까운 곳으로 대신한다."""
    from scipy.spatial import cKDTree
    n = len(x)
    dist = np.full(n, np.nan)
    ok = np.isfinite(x) & np.isfinite(y)
    if ports is None or len(ports) == 0 or not ok.any():
        return dist
    if not isinstance(ports, pd.DataFrame):                      # v2.1 형식(좌표 배열) 호환
        d, _ = cKDTree(np.asarray(ports)).query(np.c_[x[ok], y[ok]])
        dist[ok] = d / 1000.0
        return dist
    P = ports.copy()
    P['k'] = P.groupby('vessel').cumcount()
    K = int(P['k'].max()) + 1
    PX = P.pivot(index='vessel', columns='k', values='x').reindex(columns=range(K))
    PY = P.pivot(index='vessel', columns='k', values='y').reindex(columns=range(K))
    pos = PX.index.get_indexer(vessels)                          # -1 = 모항 없는 선박
    own = ok & (pos >= 0)
    if own.any():
        px, py = PX.to_numpy()[pos[own]], PY.to_numpy()[pos[own]]
        d = np.sqrt((px - x[own, None]) ** 2 + (py - y[own, None]) ** 2)
        dist[own] = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1) / 1000.0
    other = ok & (pos < 0)
    if other.any():
        d, _ = cKDTree(P[['x', 'y']].to_numpy()).query(np.c_[x[other], y[other]])
        dist[other] = d / 1000.0
    # [v2.5-2] 공용 항구(여러 선박이 모항으로 쓰는 격자)까지의 거리와 비교해 가까운 쪽
    #   다른 지역 항구로 옮겨 조업하는 선박의 정박이 '바다 위'로 잡히지 않게 한다
    sh = shared_harbors(P)
    if len(sh) and ok.any():
        d, _ = cKDTree(sh).query(np.c_[x[ok], y[ok]])
        dist[ok] = np.fmin(dist[ok], d / 1000.0)
    return dist


def shared_harbors(ports):
    """[v2.5-2] PORT_SHARED_MIN 척 이상이 모항으로 쓰는 1km 격자의 중심 좌표"""
    if ports is None or not isinstance(ports, pd.DataFrame) or len(ports) == 0:
        return np.zeros((0, 2))
    G = C.PORT_GRID_M
    q = ports.assign(cx=np.floor(ports['x'] / G), cy=np.floor(ports['y'] / G))
    g = q.groupby(['cx', 'cy'])
    keep = g['vessel'].nunique() >= C.PORT_SHARED_MIN
    return g[['x', 'y']].mean()[keep].to_numpy()


def add_port_context(df, ports=None):
    """dist_port_km(이 시간 끝 지점→그 선박의 가장 가까운 모항), loss_at_sea_h, sea_context 추가. 반환: (df, ports)"""
    fitted = ports is None
    if fitted:
        ports = fit_ports(df)
    ex, ey = df['xe'].values.astype('float64'), df['ye'].values.astype('float64')
    sx, sy = df['xs'].values.astype('float64'), df['ys'].values.astype('float64')
    # 정박 행은 끝점이 없을 수 있어 시작점으로 보완
    ex = np.where(np.isfinite(ex), ex, sx); ey = np.where(np.isfinite(ey), ey, sy)
    dist = port_distance_km(df['vessel'].values, ex, ey, ports).astype('float32')
    df['dist_port_km'] = dist
    # [v2.2-3] 바다 위에서 멈춰 있음: 거의 움직이지 않은 시간(또는 정박)에만 모항 거리, 움직였으면 0
    idle = ~(df['L_total'].values >= C.IDLE_L_M)          # NaN(정박) 포함
    df['idle_at_sea_km'] = np.where(idle, np.nan_to_num(dist), 0).astype('float32')
    at_sea = dist > C.PORT_RADIUS_KM
    known = np.isfinite(dist)
    # 바다 위에서 끊김: 움직이던 중 끊겼고, 끊긴 지점이 모항에서 멂 (위치를 모르면 v2 방식 그대로)
    df['loss_at_sea_h'] = np.where(known, np.where(at_sea, df['gap_after_moving_h'].values, 0),
                                   df['gap_after_moving_h'].values).astype('float32')
    # [B] 상태 희소성을 적용할 행: 활동 행, 또는 바다 위 정박 행 (위치를 모르면 항해 도중의 정박)
    stat = df['state'].values == 'stationary'
    mid_trip = (df['prev_active'].values > 0) & (df['next_active'].values > 0)
    df['sea_context'] = (~stat) | (stat & np.where(known, at_sea, mid_trip))
    if fitted and len(ports):
        nv = ports['vessel'].nunique() if isinstance(ports, pd.DataFrame) else 0
        src = ports['source'].value_counts().to_dict() if 'source' in ports else {}
        print(f'  모항 {len(ports):,}곳 (모항을 찾은 선박 {nv:,}척 / {df["vessel"].nunique():,}척) '
              f'근거별 {src} [trip=출항·귀항, stay=오래 정박], 공용 항구 {len(shared_harbors(ports)):,}곳')
        print(f'  바다 위 정박 행 {float((stat & at_sea).sum()) / max(stat.sum(), 1):.1%} (정박 행 중)')
    return df, ports

# ---------------------------------------------------------------------
# 그룹별 기준과 robust z
# ---------------------------------------------------------------------
def tf(x, name):
    x = np.asarray(x, dtype='float32')
    return np.log1p(np.clip(x, 0, None)) if name in C.LOG_FEATURES else x


def feats_for(state):
    return C.ACTIVE_FEATURES if state == 'active' else C.STATIONARY_FEATURES


def compute_baseline(df):
    """그룹(업종×시간대×상태)별, 특징별 분위수. 변환된(log) 값 기준."""
    rows = []
    for state in STATES:
        sub = df[df['state'] == state]
        g = sub['grp']
        for f in feats_for(state):
            x = pd.Series(tf(sub[f].values, f), index=sub.index)
            q = x.groupby(g).quantile([C.RANGE_LO, 0.05, 0.25, 0.5, 0.75, 0.95, C.RANGE_HI]).unstack()
            q.columns = ['lo', 'p05', 'q25', 'med', 'q75', 'p95', 'hi']
            q['n'] = x.groupby(g).count()
            q['sd'] = x.groupby(g).std()
            q['feature'] = f
            rows.append(q.reset_index())
    b = pd.concat(rows, ignore_index=True)
    # [A] 사분위 범위가 0(값이 거의 일정)이면 표준편차를 쓰고, 하한을 둔다.
    #     v2까지는 하한이 0.001이라, 그런 특징이 조금만 달라도 z가 10으로 포화됐다.
    iqr = (b['q75'] - b['q25']) / 1.349
    b['scale'] = np.where(iqr > C.SCALE_FLOOR, iqr, np.fmax(b['sd'].fillna(0), C.SCALE_FLOOR))
    return b


def zscores(df, baseline, state):
    feats = feats_for(state)
    Z = np.zeros((len(df), len(feats)), dtype='float32')
    OUTR = np.zeros((len(df), len(feats)), dtype=bool)
    AV = np.zeros((len(df), len(feats)), dtype=bool)
    for j, f in enumerate(feats):
        b = baseline[baseline['feature'] == f].set_index('grp')
        med = df['grp'].map(b['med']).values.astype('float32')
        sc = df['grp'].map(b['scale']).values.astype('float32')
        lo = df['grp'].map(b['lo']).values.astype('float32')
        hi = df['grp'].map(b['hi']).values.astype('float32')
        x = tf(df[f].values, f)
        ok = np.isfinite(x) & np.isfinite(med)
        Z[:, j] = np.clip(np.where(ok, (x - med) / sc, 0.0), -C.Z_CLIP, C.Z_CLIP)
        OUTR[:, j] = ok & ((x < lo) | (x > hi))
        AV[:, j] = ok
    return Z, OUTR, AV, feats


# ---------------------------------------------------------------------
# [개선 4] 선박별 기준: 그 선박의 '평소 벗어남'을 빼고 나눔
# ---------------------------------------------------------------------
def fit_vessel_baseline(vessels, Z, feats):
    d = pd.DataFrame(Z, columns=feats)
    g = d.groupby(np.asarray(vessels))
    med = g.median()
    iqr = (g.quantile(0.75) - g.quantile(0.25)) / 1.349
    n = g.size()
    w = (n / (n + C.VESSEL_SHRINK_K)).values[:, None]
    off = med * w                                   # 레코드가 적으면 0(업종 기준 그대로)에 가까움
    # v2.1: 퍼짐은 1보다 작게 하지 않음. 평소 일정한 선박(특히 0/1 특징)에서 작은 차이가
    #       크게 증폭되어 같은 선박에 경보가 몰리던 문제 방지 → 선박 기준은 '늘 시끄러운 선박을 봐주는' 쪽으로만
    sc = (iqr * w + 1.0 * (1 - w)).clip(lower=1.0)
    return off.astype('float32'), sc.astype('float32'), n


def apply_vessel_baseline(vessels, Z, vb, feats):
    off, sc, _ = vb
    o = off.reindex(np.asarray(vessels))[feats].fillna(0).to_numpy('float32')
    s = sc.reindex(np.asarray(vessels))[feats].fillna(1).to_numpy('float32')
    return np.clip((Z - o) / s, -C.Z_CLIP, C.Z_CLIP)


# ---------------------------------------------------------------------
# [개선 2] 상태 희소성: -log P(상태 | 업종, 시간대, 그 선박의 습관)
# ---------------------------------------------------------------------
def fit_state_model(df):
    cls = df.groupby(['code', 'hour', 'state']).size().unstack(fill_value=0)
    cls = (cls + 1).div((cls + 1).sum(axis=1), axis=0)            # 업종·시간대별 상태 확률
    ves = df.groupby(['vessel', 'hour', 'state']).size().unstack(fill_value=0)
    for s in STATES:
        if s not in cls:
            cls[s] = 0.5
        if s not in ves:
            ves[s] = 0
    return dict(cls=cls[STATES], ves=ves[STATES])


def state_rarity(df, model):
    cls, ves = model['cls'], model['ves']
    key_c = pd.MultiIndex.from_arrays([df['code'].values, df['hour'].values.astype(cls.index.levels[1].dtype)])
    pc_ = cls.reindex(key_c)
    key_v = pd.MultiIndex.from_arrays([df['vessel'].values, df['hour'].values.astype(ves.index.levels[1].dtype)])
    nv = ves.reindex(key_v).fillna(0)
    st = df['state'].values
    p_cls = np.where(st == 'active', pc_['active'].values, pc_['stationary'].values)
    n_state = np.where(st == 'active', nv['active'].values, nv['stationary'].values)
    n_hour = nv.sum(axis=1).values
    k = C.STATE_SHRINK_K
    p = (n_state + k * np.nan_to_num(p_cls, nan=0.5)) / (n_hour + k)
    return (-np.log(np.clip(p, 1e-6, 1))).astype('float32')



# ---------------------------------------------------------------------
# [C] AE 점수의 선박별 기준: log(오차)를 그룹 기준 z로 만든 뒤 그 선박의 평소 수준을 뺌
# ---------------------------------------------------------------------
AE_CH = ['ae_err_dist', 'ae_err_turn', 'ae_err_stop']
AE_CH_LABEL = {'ae_err_dist': 'AE:거리 패턴', 'ae_err_turn': 'AE:회전 패턴', 'ae_err_stop': 'AE:정지 패턴'}


def fit_ae_baseline(df):
    la = np.log(df['ae_raw'].astype('float64').clip(lower=1e-8))
    g = la.groupby(df['grp'].values)
    med = g.median()
    sc = ((g.quantile(0.75) - g.quantile(0.25)) / 1.349).clip(lower=C.SCALE_FLOOR)
    z = (la - df['grp'].map(med)) / df['grp'].map(sc)
    vz = z.groupby(df['vessel'].values)
    n = vz.count()
    # 평소 오차가 높은(늘 시끄러운) 선박만 빼 준다. 평소 오차가 낮은 선박까지 빼면 점수 전체가 위로 밀려
    # 그 선박에 경보가 몰린다 (v2.1 점검에서 4,297시간 중 808시간이 경보인 선박 발생)
    off = (vz.median() * (n / (n + C.VESSEL_SHRINK_K))).clip(lower=0)
    ch_med = {c: float(df[c].median()) for c in AE_CH if c in df.columns}
    return dict(med=med, sc=sc, off=off, ch_med=ch_med)


def apply_ae_baseline(df, b):
    la = np.log(df['ae_raw'].astype('float64').clip(lower=1e-8))
    z = (la - df['grp'].map(b['med'])) / df['grp'].map(b['sc'])
    if C.AE_VESSEL_BASELINE:
        z = z - df['vessel'].map(b['off']).fillna(0)
    return z.values.astype('float32')


def ae_reason(df, b):
    """AE 채널별 오차 중 평소 대비 가장 큰 채널 이름 (3단계 v2.1을 실행한 경우만)"""
    cols = [c for c in AE_CH if c in df.columns and c in b.get('ch_med', {})]
    if not cols:
        return np.full(len(df), '', dtype=object)
    R = np.column_stack([df[c].values / max(b['ch_med'][c], 1e-8) for c in cols])
    j = np.nanargmax(np.nan_to_num(R, nan=-1), axis=1)
    return np.where(np.isfinite(R).any(axis=1), np.array([AE_CH_LABEL[c] for c in cols], dtype=object)[j], '')

def top_reasons(Z, feats, k=3):
    idx = np.argsort(-np.abs(Z), axis=1)[:, :k]
    names = np.array(feats, dtype=object)
    outc = {}
    for i in range(k):
        j = idx[:, i]
        zz = Z[np.arange(len(Z)), j]
        lab = np.where(zz >= 0, '↑', '↓')
        outc[f'reason{i + 1}'] = np.where(np.abs(zz) >= 2, names[j] + lab, '')
        outc[f'reason{i + 1}_z'] = zz.astype('float32')
    return outc


# ---------------------------------------------------------------------
# 점수 결합 (4단계와 6단계가 같은 함수를 씀). 백분위는 동점을 가운데 순위로.
# ---------------------------------------------------------------------
def _midrank(sorted_ref, v):
    lo = np.searchsorted(sorted_ref, v, side='left')
    hi = np.searchsorted(sorted_ref, v, side='right')
    return (lo + hi) / 2 / len(sorted_ref)


def sorted_by_group(values, groups):
    ref = pd.DataFrame({'g': np.asarray(groups), 'v': np.asarray(values, dtype='float64')}).dropna()
    return {g: np.sort(v.values) for g, v in ref.groupby('g')['v']}


def sorted_finite(values):
    r = np.asarray(values, dtype='float64')
    return np.sort(r[np.isfinite(r)])


def group_percentile(values, groups, sorted_ref):
    values = np.asarray(values, dtype='float64')
    out_ = np.full(len(values), np.nan)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        m = (groups == g) & np.isfinite(values)
        if m.any() and g in sorted_ref:
            out_[m] = _midrank(sorted_ref[g], values[m])
    return out_


def global_percentile(values, sorted_ref):
    if not isinstance(sorted_ref, np.ndarray) or sorted_ref.ndim != 1 or not np.all(np.diff(sorted_ref) >= 0):
        sorted_ref = sorted_finite(sorted_ref)
    v = np.asarray(values, dtype='float64')
    o = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    o[ok] = _midrank(sorted_ref, v[ok])
    return o


def make_ref(ref):
    """실제 데이터 점수 분포를 정렬해 둔 기준 (6단계에서 사례마다 다시 정렬하지 않도록)"""
    r = {}
    for col in ['if_raw', 'ae_raw', 'ae_adj']:
        if col in ref.columns and ref[col].notna().any():
            r[col] = sorted_by_group(ref[col].values, ref['grp'].values)
    if 'state_rarity' in ref.columns:
        r['state_rarity'] = sorted_finite(ref['state_rarity'].values)
    if 'combined' in ref.columns:
        r['combined'] = sorted_finite(ref['combined'].values)
    return r


def combine(df, refc):
    """IF·AE 원점수 → 그룹 내 백분위 → max(또는 평균) → 상태 희소성으로 보정. refc = make_ref(실제 데이터)"""
    num = np.zeros(len(df)); den = np.zeros(len(df)); mx = np.full(len(df), np.nan)
    res = {}
    ae_col = 'ae_adj' if ('ae_adj' in df.columns and 'ae_adj' in refc) else 'ae_raw'   # [C]
    for col, w in [('if_raw', C.W_IF), (ae_col, C.W_AE)]:
        if col not in df.columns or col not in refc:
            continue
        u = group_percentile(df[col].values, df['grp'].values, refc[col])
        res['u_' + col.split('_')[0]] = u
        ok = np.isfinite(u)
        num += np.where(ok, w * u, 0); den += np.where(ok, w, 0)
        mx = np.where(ok, np.fmax(mx, u), mx)
    base = mx if C.COMBINE == 'max' else np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    if 'state_rarity' in df.columns and 'state_rarity' in refc:
        us = global_percentile(df['state_rarity'].values, refc['state_rarity'])
        res['u_state'] = us
        g = np.clip((np.nan_to_num(us) - C.STATE_TAIL) / (1 - C.STATE_TAIL), 0, 1)
        if 'sea_context' in df.columns:                      # [B] 항구에서 쉬는 정박은 보정하지 않음
            g = g * df['sea_context'].values.astype(float)
        base = base + C.W_STATE * g * (1 - base)
    res['combined'] = base
    return res


# ---------------------------------------------------------------------
# [개선 3] 이상 사건: 같은 항해에서 연속된 주의 이상 시간들의 묶음
# ---------------------------------------------------------------------
def loss_hours(df, km=None):
    """[v2.2-2] 행별 '바다 위에서 움직이던 중 끊긴 시간'. 끊긴 지점이 모항에서 km 이내면 0.
    위치를 모르는 행(0단계 미실행, 모항을 못 찾은 선박 등)은 0 — 규칙 경보가 쏟아지지 않게 한다.
    (위치 없이 모든 이동 중 끊김을 세면 실제 데이터에서 하루 약 1,000건)"""
    km = C.LOSS_KM if km is None else km
    g = np.nan_to_num(df['gap_after_moving_h'].values.astype('float64'))
    if 'dist_port_km' in df.columns:
        d = df['dist_port_km'].values.astype('float64')
        at_sea = np.where(np.isfinite(d), d > km, False)
    else:
        at_sea = np.zeros(len(df), bool)
    return np.where(at_sea, g, 0.0)


def vessel_loss_habit(df, h=None, km=None):
    """[v2.2-2] 선박별 '바다 위 신호 단절로 끝난 항해 비율' (상습 단절 선박 판정용). h·km = 규칙 기준"""
    h = C.LOSS_H if h is None else h
    n_loss = pd.Series(loss_hours(df, km) >= h, index=df.index).groupby(df['vessel']).sum()
    n_trip = df.groupby('vessel')['trip'].nunique()
    return (n_loss / n_trip.clip(lower=1)).rename('loss_rate')


def stop_runs(df, km=None):
    """[v2.4-3] 바다 위 장시간 정지: 같은 항해에서 '움직이던 시간' 바로 뒤에 이어지는,
    모항에서 km 밖이면서 거의 움직이지 않은(시간당 IDLE_L_M 미만 또는 정박) 연속 시간.
    반환: (행별 정지 길이(시간, 해당 없으면 0), 행별 정지 번호(-1 = 없음))"""
    km = C.STOP_KM if km is None else km
    n = len(df)
    if n == 0 or 'dist_port_km' not in df.columns:
        return np.zeros(n), np.full(n, -1, dtype='int64')
    L = df['L_total'].values.astype('float64')
    d = df['dist_port_km'].values.astype('float64')
    trip = df['trip'].values
    idle = ~(L >= C.IDLE_L_M)
    m = idle & np.isfinite(d) & (d > km)
    same = np.r_[False, trip[1:] == trip[:-1]]
    moving_prev = np.r_[False, L[:-1] >= C.IDLE_L_M] & same
    start = m & ~(np.r_[False, m[:-1]] & same)
    rid = np.cumsum(start)
    rid = np.where(m, rid, 0)
    ok_start = pd.Series(moving_prev & start).groupby(rid).transform('max').values.astype(bool)   # 움직이다 멈춘 정지만
    keep = m & ok_start
    length = pd.Series(np.where(keep, 1, 0)).groupby(np.where(keep, rid, 0)).transform('sum').values
    return np.where(keep, length, 0).astype('float64'), np.where(keep, rid, -1).astype('int64')


def vessel_stop_habit(df, h=None, km=None):
    """[v2.4-3] 선박별 '바다 위 장시간 정지가 있었던 항해 비율' (상습 정지 선박 판정용)"""
    h = C.STOP_H if h is None else h
    length, rid = stop_runs(df, km)
    m = length >= h
    runs = pd.DataFrame({'vessel': df['vessel'].values[m], 'rid': rid[m]}).drop_duplicates()
    n_stop = runs.groupby('vessel').size()
    n_trip = df.groupby('vessel')['trip'].nunique()
    return (n_stop.reindex(n_trip.index).fillna(0) / n_trip.clip(lower=1)).rename('stop_rate')


def stop_frequency(df, days):
    """[v2.4-3] 바다 위 장시간 정지가 실제 데이터에 얼마나 흔한가: 정지 시간 × 모항 거리 기준별 하루 건수"""
    out = []
    n_v = df['vessel'].nunique()
    for km in C.STOP_DIAG_KM:
        length, rid = stop_runs(df, km)
        for h in C.STOP_DIAG_H:
            m = length >= h
            runs = pd.DataFrame({'vessel': df['vessel'].values[m], 'rid': rid[m]}).drop_duplicates()
            hab = vessel_stop_habit(df, h, km)
            hv = hab.reindex(runs['vessel']).fillna(0).values >= C.HABIT_STOP_RATE
            out.append(dict(min_stop_h=h, min_dist_km=km, cases=len(runs), per_day=round(len(runs) / days, 1),
                            per_day_alarm=round(float((~hv).sum()) / days, 1),
                            vessels=int(runs['vessel'].nunique()), vessel_share=round(runs['vessel'].nunique() / n_v, 4),
                            habitual_vessels=int((hab >= C.HABIT_STOP_RATE).sum())))
    return pd.DataFrame(out)


def detect_events(df, t1, t2, persist_h=None, bridge_h=None, t_mean=None, loss_rule=None, habit=None,
                  loss_h=None, loss_km=None, stop_rule=None, stop_h=None, stop_km=None, stop_habit=None):
    """df: vessel, code, trip, t, final_score, gap_after_moving_h, (dist_port_km), reason1 — 선박·시간 순 정렬
    사건 두 종류
      (1) 지속 이상: 같은 항해에서 주의 이상 시간의 묶음 (빈틈 bridge_h 시간까지 허용)
          주의 시간 ≥ persist_h 이고 (경보 수준 1회 이상 또는 묶음 평균 ≥ t_mean)
          등급: 경보 수준 시간이 있으면 '경보', 평균 기준만 충족하면 '주의'
      (2) 바다 위 신호 단절 (규칙): 바다 위에서 움직이던 중 LOSS_H 시간 이상 끊김. 점수와 무관
          등급: '경보', 단 상습 단절 선박(habit ≥ HABIT_LOSS_RATE)은 '주의'
          이미 지속 이상 사건의 마지막 시간이면 그 사건을 '지속 이상+단절'로 바꾸고 등급을 높임
      [v2.4-1] (1)은 바다 위(모항 SCORE_SEA_KM 밖) 시간만 쓴다 (위치를 모르는 행은 포함)
      [v2.4-3] (3) 바다 위 장시간 정지 (규칙): 움직이다가 모항 stop_km 밖에서 stop_h 시간 이상 거의 안 움직임
          등급: '경보', 단 상습 정지 선박(stop_habit ≥ HABIT_STOP_RATE)은 '주의'
          (1)과 겹치면 그 사건을 '지속 이상+정지'로 바꾸고 등급을 높임
    반환: (사건 표, 행별 사건 번호 배열(-1 = 없음))"""
    persist_h = C.PERSIST_H if persist_h is None else persist_h
    bridge_h = C.BRIDGE_H if bridge_h is None else bridge_h
    t_mean = np.inf if t_mean is None else t_mean
    loss_rule = C.LOSS_RULE if loss_rule is None else loss_rule
    loss_h = C.LOSS_H if loss_h is None else loss_h
    stop_rule = C.STOP_RULE if stop_rule is None else stop_rule
    stop_h = C.STOP_H if stop_h is None else stop_h
    s = df['final_score'].values.astype('float64')
    t = df['t'].values
    trip = df['trip'].values
    vessel = df['vessel'].values
    ev_id = np.full(len(df), -1, dtype='int64')
    cols = ['event_id', 'vessel', 'code', 't_start', 't_end', 'hours', 'max_score', 'mean_score',
            'alarm_hours', 'loss_h', 'stop_h', 'dist_port_km', 'top_reason', 'kind', 'level']
    tables = []
    lh = loss_hours(df, loss_km)
    dist = df['dist_port_km'].values if 'dist_port_km' in df.columns else np.full(len(df), np.nan)

    # (1) 지속 이상
    flag = s >= t1
    if C.SCORE_SEA_ONLY:                                    # [v2.4-1] 항구 안 시간은 점수 사건에서 제외
        flag &= np.where(np.isfinite(dist.astype('float64')), dist > C.SCORE_SEA_KM, True)
    idx = np.flatnonzero(flag)
    if len(idx):
        tt, tr = t[idx], trip[idx]
        cont = np.r_[False, (tr[1:] == tr[:-1]) & (tt[1:] - tt[:-1] <= bridge_h + 1)]
        run = np.cumsum(~cont)
        d = pd.DataFrame({'run': run, 'vessel': vessel[idx], 'code': df['code'].values[idx], 't': tt,
                          'score': s[idx], 'al': s[idx] >= t2, 'loss': lh[idx], 'dist': dist[idx],
                          'reason1': df['reason1'].values[idx]})
        g = d.groupby('run', sort=True)
        ev = pd.DataFrame({'vessel': g['vessel'].first(), 'code': g['code'].first(),
                           't_start': g['t'].min(), 't_end': g['t'].max(), 'hours': g.size(),
                           'max_score': g['score'].max(), 'mean_score': g['score'].mean(),
                           'alarm_hours': g['al'].sum().astype(int), 'loss_h': g['loss'].last(),
                           'dist_port_km': g['dist'].last()})
        rc = d[d['reason1'] != ''].groupby(['run', 'reason1']).size().reset_index(name='n')
        rc = rc.sort_values(['run', 'n'], ascending=[True, False]).drop_duplicates('run').set_index('run')['reason1']
        ev['top_reason'] = rc.reindex(ev.index).fillna('')
        keep = (ev['hours'] >= persist_h) & ((ev['max_score'] >= t2) | (ev['mean_score'] >= t_mean))
        ev = ev[keep]
        ev['kind'] = '지속 이상'
        ev['level'] = np.where(ev['alarm_hours'] > 0, '경보', '주의')
        ev['stop_h'] = 0.0
        ev_id[idx] = np.where(np.isin(run, ev.index.values), run, -1)
        ev = ev.reset_index().rename(columns={'run': 'event_id'})
        tables.append(ev)
        next_id = int(run.max()) + 1
    else:
        next_id = 1

    # (3) 바다 위 장시간 정지 (규칙) [v2.4-3]
    if stop_rule:
        slen, srid = stop_runs(df, stop_km)
        S = np.flatnonzero(slen >= stop_h)
        if len(S):
            sr = pd.DataFrame({'i': S, 'rid': srid[S], 'ev': ev_id[S]})
            hit = sr.loc[sr['ev'] >= 0, 'ev'].unique()
            if len(hit) and tables:                          # 지속 이상과 겹침 → 그 사건을 높임
                ev = tables[0].set_index('event_id')
                ev.loc[hit, 'kind'] = '지속 이상+정지'
                ev.loc[hit, 'level'] = '경보'
                ev.loc[hit, 'stop_h'] = sr[sr['ev'] >= 0].groupby('ev')['i'].apply(lambda x: float(slen[x].max()))
                tables[0] = ev.reset_index()
            covered = sr.groupby('rid')['ev'].transform('max') >= 0
            sr = sr[~covered]
            if len(sr):
                g = sr.groupby('rid')['i']
                first, last = g.min().values, g.max().values
                rows_by = g.apply(list)
                eid = np.arange(next_id, next_id + len(first))
                next_id += len(first)
                for e, ii in zip(eid, rows_by.values):
                    ev_id[np.asarray(ii)] = e
                ves = vessel[first]
                rate = stop_habit.reindex(ves).fillna(0).values if stop_habit is not None else np.zeros(len(first))
                mx = g.apply(lambda x: float(s[x].max())).values
                mn = g.apply(lambda x: float(s[x].mean())).values
                new = pd.DataFrame({'event_id': eid, 'vessel': ves, 'code': df['code'].values[first],
                                    't_start': t[first], 't_end': t[last], 'hours': slen[first],
                                    'max_score': mx, 'mean_score': mn,
                                    'alarm_hours': g.apply(lambda x: int((s[x] >= t2).sum())).values,
                                    'loss_h': 0.0, 'stop_h': slen[first],
                                    'dist_port_km': g.apply(lambda x: float(np.nanmean(dist[x]))).values,
                                    'top_reason': '바다 위 장시간 정지', 'kind': '바다 위 장시간 정지',
                                    'level': np.where(rate >= C.HABIT_STOP_RATE, '주의', '경보')})
                too_long = new['stop_h'] > C.STOP_MAX_H                 # [v2.5-3] 미확인 항구 정박 추정
                new.loc[too_long, 'kind'] = '바다 위 장기 정박(추정)'
                new.loc[too_long, 'level'] = '주의'
                tables.append(new)

    # (2) 바다 위 신호 단절 (규칙)
    if loss_rule:
        L = np.flatnonzero(lh >= loss_h)
        if len(L):
            in_ev = ev_id[L] >= 0
            if in_ev.any() and tables:                      # 이미 있는 사건 끝에서 끊김 → 그 사건에 합침
                lmax = pd.Series(lh[L[in_ev]], index=ev_id[L[in_ev]]).groupby(level=0).max()
                for k, tb in enumerate(tables):
                    tb = tb.set_index('event_id')
                    hit = lmax.index[lmax.index.isin(tb.index)]
                    if len(hit):
                        tb.loc[hit, 'kind'] = tb.loc[hit, 'kind'] + '+단절'
                        tb.loc[hit, 'level'] = '경보'
                        tb.loc[hit, 'loss_h'] = lmax.loc[hit].values
                    tables[k] = tb.reset_index()
            L = L[~in_ev]
            if len(L):
                rate = habit.reindex(vessel[L]).fillna(0).values if habit is not None else np.zeros(len(L))
                new = pd.DataFrame({'event_id': np.arange(next_id, next_id + len(L)), 'vessel': vessel[L],
                                    'code': df['code'].values[L], 't_start': t[L], 't_end': t[L], 'hours': 1,
                                    'max_score': s[L], 'mean_score': s[L], 'alarm_hours': (s[L] >= t2).astype(int),
                                    'loss_h': lh[L], 'stop_h': 0.0, 'dist_port_km': dist[L],
                                    'top_reason': '바다 위 신호 단절', 'kind': '바다 위 신호 단절',
                                    'level': np.where(rate >= C.HABIT_LOSS_RATE, '주의', '경보')})
                ev_id[L] = new['event_id'].values
                tables.append(new)
    if not tables or sum(len(x) for x in tables) == 0:
        return pd.DataFrame(columns=cols + ['start', 'end']), ev_id
    ev = pd.concat([x for x in tables if len(x)], ignore_index=True)[cols]
    ev['start'] = pd.to_datetime(ev['t_start'], unit='h')
    ev['end'] = pd.to_datetime(ev['t_end'], unit='h')
    return ev, ev_id


def loss_frequency(df, days):
    """[v2.2-1] 바다 위 신호 단절이 실제 데이터에 얼마나 흔한가: 끊긴 시간 × 모항 거리 기준별 하루 건수
    [v2.3] 상습 단절 판정도 같은 기준으로 계산 → per_day_alarm = '경보' 등급(상습 단절 선박 제외) 하루 건수"""
    out = []
    n_v = df['vessel'].nunique()
    for km in C.LOSS_DIAG_KM:
        lh = loss_hours(df, km)
        for h in C.LOSS_DIAG_H:
            m = lh >= h
            v = df['vessel'].values[m]
            row = dict(min_gap_h=h, min_dist_km=km, cases=int(m.sum()), per_day=round(m.sum() / days, 1),
                       vessels=int(len(np.unique(v))), vessel_share=round(len(np.unique(v)) / n_v, 4))
            hab = vessel_loss_habit(df, h, km)
            hv = hab.reindex(v).fillna(0).values >= C.HABIT_LOSS_RATE
            row['per_day_alarm'] = round(float((~hv).sum()) / days, 1)
            row['habitual_vessels'] = int((hab >= C.HABIT_LOSS_RATE).sum())
            out.append(row)
    return pd.DataFrame(out)


def setup_korean_font():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR']:
        if name in have:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore', message='Glyph')
    return plt


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2, default=str)
