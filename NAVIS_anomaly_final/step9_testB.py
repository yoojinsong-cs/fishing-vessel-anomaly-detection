# -*- coding: utf-8 -*-
"""
9단계 (테스트). V2.5 학습 결과로 테스트셋(testB 대상 선박)의 시간별 이상항적 점수를 매긴다.

학습(2026년 1~6월)에서 만든 것을 그대로 쓴다 — 다시 학습하지 않는다.
  normal_baseline.parquet  업종×시간대×상태 정상 기준
  model_if.joblib          상태별 Isolation Forest + 선박별 기준 + 상태 확률 + 모항
  model_ae.pt, model_ae_baseline.joblib   시퀀스 오토인코더와 선박별 AE 기준
  scores_final.parquet     학습 데이터 점수 분포(백분위 기준), 업종 불일치, 상습 단절·정지
  final_thresholds.json    최종 임계값과 규칙 기준

테스트셋에는 업종(fishing_code)이 없다.
  - 학습 데이터에 있던 선박: 학습 데이터의 등록 업종을 쓴다.
  - 없던 선박: 네 업종으로 각각 채점해 보고, 점수 중앙값이 가장 낮은(가장 '정상'으로 보이는) 업종을 쓴다.

입력: TEST_DIR/testB_raw_*.csv  (원본 CSV에서 testB 대상 선박만 뽑은 것: filter_testB.py)
출력: test_config.TEST_OUT (기본: 결과 폴더/test_scoring/out)
  testB_scores.parquet          선박·시간별 점수와 사건
  testB_events.csv              사건 목록
  (제출 파일은 step9b_submission.py 가 만든다)
실행 시간: 3~10분
"""
import os
import sys
import glob
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from pyproj import Transformer

import anomaly_config as C
import anomaly_common as U
import step3_seq_ae as AE
import step6_validate as V
from step0_endpoints import endpoints

NF = AE.NF
import test_config as T
TEST_DIR = T.TEST_DIR


def _feat_chunk(geoms, fthr, names, swap):
    tr = Transformer.from_crs(NF.CFG['SRC_CRS'], NF.CFG['DST_CRS'], always_xy=True)
    parts = NF.records_to_parts(NF.parse_geoms(pd.Series(geoms)), tr, swap)
    return NF.features_batch(parts, fthr, NF.CFG, names), parts


def load_test(fthr, files=None):
    files = files or sorted(glob.glob(os.path.join(TEST_DIR, 'testB_raw_*.csv')))
    if not files:
        raise FileNotFoundError(f'{TEST_DIR} 에 testB_raw_*.csv 가 없습니다 (filter_testB.py 먼저 실행)')
    names = NF.feature_names(NF.CFG)
    swap = bool(fthr.get('swap', False))
    tr = Transformer.from_crs(NF.CFG['SRC_CRS'], NF.CFG['DST_CRS'], always_xy=True)
    frames, all_parts = [], []
    for f in files:
        t0 = time.time()
        raw = pd.read_csv(f, dtype=str, encoding='utf-8-sig')
        raw = raw[raw[C.COL_VESSEL].notna() & raw[C.COL_GEOM].notna()].reset_index(drop=True)
        geoms = raw[C.COL_GEOM].tolist()
        step = 2000
        res = Parallel(n_jobs=max(1, (os.cpu_count() or 2)))(
            delayed(_feat_chunk)(geoms[i:i + step], fthr, names, swap) for i in range(0, len(geoms), step))
        feats = [x for r in res for x in r[0]]
        parts = [x for r in res for x in r[1]]
        F = pd.DataFrame(feats)
        e = endpoints(NF.parse_geoms(raw[C.COL_GEOM]), tr, swap)
        d = pd.DataFrame({
            'src_file': 'TEST_' + os.path.basename(f), 'row_in_file': np.arange(len(raw), dtype='int64'),
            'vessel': raw[C.COL_VESSEL].astype(str).values,
            'hour': pd.to_numeric(raw[C.COL_HOUR], errors='coerce').astype('int16').values,
            'date': pd.to_datetime(raw[C.COL_DATE], format='%Y%m%d', errors='coerce').values,
            'xs': e[:, 0].astype('float32'), 'ys': e[:, 1].astype('float32'),
            'xe': e[:, 2].astype('float32'), 'ye': e[:, 3].astype('float32')})
        for c in U.raw_feature_names():
            d[c] = F[c].astype('float32').values if c in F else np.float32(np.nan)
        frames.append(d)
        all_parts += parts
        print(f'  {os.path.basename(f)}: {len(d):,}행 ({time.time() - t0:.0f}초)', flush=True)
    df = pd.concat(frames, ignore_index=True)
    df['month'] = df['date'].dt.month.astype('int16')
    df['t'] = (df['date'].values.astype('datetime64[h]').astype('int64') + df['hour'].values).astype('int64')
    df['syn'] = np.arange(len(df))
    # 같은 선박·시각이 두 번 있으면 첫 행만
    keep = ~df.duplicated(['vessel', 't'])
    if (~keep).any():
        print(f'  중복 선박·시각 {(~keep).sum():,}행 제외')
    return df[keep].reset_index(drop=True), all_parts


class TestScorer(V.Scorer):
    """V2.5 6단계 채점기와 같은 순서. 학습 데이터의 AE 원점수는 쓰지 않는다(테스트 행은 모두 새로 계산)."""
    def __init__(self, ref):
        self.refc = U.make_ref(ref)
        self.base = pd.read_parquet(U.out('normal_baseline.parquet'))
        bundle = joblib.load(U.out('model_if.joblib'))
        self.models, self.state_model = bundle['models'], bundle['state_model']
        self.ports = bundle.get('ports')
        aeb_path = U.out('model_ae_baseline.joblib')
        self.aeb = joblib.load(aeb_path) if os.path.exists(aeb_path) else None
        self.ae = AE.load_model()
        self.ae_ref = pd.Series(dtype='float64', index=pd.MultiIndex.from_arrays([[], []], names=U.KEYS))
        self.mm = ref.groupby('vessel')['mismatch'].first()


def load_reference():
    """학습 결과(기준 분포·모델·임계값)를 읽는다. 여러 번 채점할 때 한 번만 읽도록 분리."""
    with open(C.FEAT_THRESH, encoding='utf-8') as fp:
        fthr = json.load(fp)
    with open(U.out('final_thresholds.json'), encoding='utf-8') as fp:
        FT = json.load(fp)
    print(f'[0] 최종 기준: 주의 ≥ {FT["caution"]:.4f}, 경보 ≥ {FT["alarm"]:.4f}, 지속 {FT["persist_h"]}h, '
          f'단절 {FT["loss_h"]}h·{FT["loss_km"]}km, 정지 {FT["stop_h"]}h·{FT["stop_km"]}km')
    print('[1] 학습 데이터 기준 분포 읽기', flush=True)
    cols = ['vessel', 'code', 't', 'trip', 'grp', 'if_raw', 'ae_raw', 'ae_adj', 'state_rarity', 'combined',
            'mismatch', 'gap_after_moving_h', 'dist_port_km', 'L_total']
    ref = pd.read_parquet(U.out('scores_final.parquet'), columns=cols).sort_values(['vessel', 't']).reset_index(drop=True)
    R = dict(fthr=fthr, FT=FT, vcode=ref.groupby('vessel')['code'].first(),
             habit=U.vessel_loss_habit(ref, FT['loss_h'], FT['loss_km']),
             shabit=U.vessel_stop_habit(ref, FT['stop_h'], FT['stop_km']), sc=TestScorer(ref))
    del ref
    return R


def score_files(R, files=None, tpred=None, t0=None):
    """원본 CSV(대상 선박만) → 선박·시간별 점수 S, 사건 E.
    tpred: {rfid: 업종} 학습 데이터에 없는 선박의 업종(분류 모델). None 이면 여기서 분류 모델을 부른다."""
    t0 = t0 or time.time()
    FT, vcode, habit, shabit, sc = R['FT'], R['vcode'], R['habit'], R['shabit'], R['sc']
    t1, t2, tm = FT['caution'], FT['alarm'], FT['event_mean']
    print('[2] 테스트 원본 → 특징', flush=True)
    df, parts = load_test(R['fthr'], files)
    print(f'  {len(df):,}행, 선박 {df.vessel.nunique()}척, {df.date.min():%Y-%m-%d} ~ {df.date.max():%Y-%m-%d}')

    print('[3] 선박별 채점', flush=True)
    out, ev_all, code_note = [], [], {}
    unknown = sorted(set(df['vessel']) - set(vcode.index))
    if tpred is None:
        tpred = {}
        # 학습 데이터에 없는 선박의 업종: 분류 모델(혜승님 패키지)을 import 해서 함수로 예측
        if unknown:
            try:
                import cls_bridge
                tp = cls_bridge.predict_vessels(files or sorted(glob.glob(os.path.join(TEST_DIR, 'testB_raw_*.csv'))))
                tpred = tp['pred_type'].reindex(unknown).dropna().to_dict()
                print(f'  학습 데이터에 없는 선박 {len(unknown)}척 → 분류 모델 업종 예측 {len(tpred)}척')
            except (ImportError, FileNotFoundError) as e:
                print(f'  [주의] 분류 모델을 부를 수 없어 업종을 점수 기준으로 추정합니다\n  {e}')
    n_all = df['vessel'].nunique()
    for i, (v, g) in enumerate(df.groupby('vessel', sort=True)):
        g = g.sort_values('t')
        syn_parts = {int(k): parts[int(k)] for k in g['syn'].values}
        if v in vcode.index:
            g = g.assign(code=vcode[v]); s = sc.score(g, syn_parts); code_note[v] = '학습 데이터 업종'
        elif tpred.get(v) in C.CLASSES:
            g = g.assign(code=tpred[v]); s = sc.score(g, syn_parts); code_note[v] = f'분류모델({tpred[v]})'
        else:
            best = None
            for c in C.CLASSES:
                s_c = sc.score(g.assign(code=c), syn_parts)
                med = float(np.nanmedian(s_c['final_score']))
                if best is None or med < best[0]:
                    best = (med, c, s_c)
            s = best[2]; code_note[v] = f'추정({best[1]})'
        s = s.sort_values('t').reset_index(drop=True)
        # 상습 여부: 학습 데이터에 있으면 그 값, 없으면 테스트 기간 값
        hb = habit if v in habit.index else U.vessel_loss_habit(s, FT['loss_h'], FT['loss_km'])
        shb = shabit if v in shabit.index else U.vessel_stop_habit(s, FT['stop_h'], FT['stop_km'])
        ev, ev_id = U.detect_events(s, t1, t2, persist_h=FT['persist_h'], bridge_h=FT['bridge_h'], t_mean=tm,
                                    habit=hb, loss_h=FT['loss_h'], loss_km=FT['loss_km'],
                                    stop_h=FT['stop_h'], stop_km=FT['stop_km'], stop_habit=shb)
        s['event_id'] = ev_id
        s['loss_h'] = U.loss_hours(s, FT['loss_km'])
        s['stop_h'] = U.stop_runs(s, FT['stop_km'])[0]
        if len(ev):
            ev_all.append(ev)
            s = s.merge(ev[['event_id', 'kind', 'level']].rename(columns={'kind': 'event_kind', 'level': 'event_level'}),
                        on='event_id', how='left')
        else:
            s['event_kind'] = np.nan; s['event_level'] = np.nan
        out.append(s)
        if (i + 1) % 100 == 0 or (i + 1) == n_all:
            print(f'  {i + 1}/{n_all}척 ({time.time() - t0:.0f}초)', flush=True)
    S = pd.concat(out, ignore_index=True)
    E = pd.concat(ev_all, ignore_index=True) if ev_all else pd.DataFrame()
    S['code_source'] = S['vessel'].map(code_note)
    S['row_level'] = np.where(S['final_score'] >= t2, '경보', np.where(S['final_score'] >= t1, '주의', '정상'))
    keep = ['vessel', 'code', 'code_source', 'date', 'hour', 't', 'trip', 'state', 'final_score', 'anomaly_score',
            'u_if', 'u_ae', 'u_state', 'mismatch', 'row_level', 'event_id', 'event_kind', 'event_level', 'reason1',
            'L_total', 'dist_port_km', 'idle_at_sea_km', 'loss_h', 'stop_h', 'xe', 'ye', 'src_file', 'row_in_file']
    return S[[c for c in keep if c in S]], E


def main():
    t0 = time.time()
    R = load_reference()
    S, E = score_files(R, t0=t0)
    od = T.TEST_OUT; os.makedirs(od, exist_ok=True)
    S.to_parquet(os.path.join(od, 'testB_scores.parquet'), index=False)
    if len(E):
        E.drop(columns=['t_start', 't_end'], errors='ignore').to_csv(os.path.join(od, 'testB_events.csv'),
                                                                   index=False, encoding='utf-8-sig')
    print(f'  저장: {od} ({time.time() - t0:.0f}초)')
    if len(E):
        print('  사건 종류×등급:'); print(pd.crosstab(E['kind'], E['level']).to_string())


if __name__ == '__main__':
    main()
