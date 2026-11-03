# -*- coding: utf-8 -*-
"""
2단계 (v2.3, v2.4 변경 없음). 정상 범위 + 선박별 기준 + Isolation Forest + 상태 희소성

방법
  - [개선 1·5] 시간 맥락 특징을 만든다 (첫·마지막 레코드 공백은 중립, 항해 단위 맥락).
  - 업종×시간대×상태 그룹의 중앙값·IQR로 정규화(robust z) → 업종·시간대 조건부 판단
  - [개선 4] 선박별 기준: z에서 그 선박의 평소 벗어남(중앙값)을 빼고 평소 퍼짐으로 나눈다.
    → 단말기 잡음이나 고유한 조업 습관처럼 '늘 그런 것'은 이상으로 보지 않는다.
  - Isolation Forest: 상태(정박/활동)별로 학습 → if_raw (클수록 이상)
  - [개선 2] 상태 희소성: 이 업종·이 선박이 이 시간대에 이 상태일 확률이 낮을수록 state_rarity가 큼
  - 설명용으로 |z|가 큰 특징 3개를 reason1~3에 남긴다.

출력 (OUT_DIR)
  scores_if.parquet       행(선박×시간)별 if_raw, state_rarity, range_score, reason1~3, 맥락 특징
  model_if.joblib         상태별 Isolation Forest + 선박별 기준 + 상태 확률 모델 (6단계에서 재사용)
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import anomaly_config as C
import anomaly_common as U

KEEP = U.KEYS + ['vessel', 'code', 'date', 'hour', 'month', 't', 'trip', 'state', 'grp', 'L_total',
                 'gap_before_h', 'gap_after_h', 'gap_after_moving_h', 'run_active_h', 'trip_hour_idx',
                 'dist_port_km', 'idle_at_sea_km', 'loss_at_sea_h', 'sea_context', 'xe', 'ye']


def main():
    t0 = time.time()
    print('[1] 특징 테이블 읽기 + 시간 맥락')
    df = U.add_time_context(U.load_features())
    df, ports = U.add_port_context(df)          # [v2.1 D]
    df['grp'] = U.group_key(df)
    try:
        base = pd.read_parquet(U.out('normal_baseline.parquet'))
        if not set(C.ACTIVE_FEATURES + C.STATIONARY_FEATURES) <= set(base['feature']) or 'sd' not in base.columns:
            raise FileNotFoundError
        print('  1단계의 정상 기준을 사용합니다')
    except FileNotFoundError:
        print('  1단계 결과가 지금 설정의 특징과 맞지 않아 정상 기준을 새로 계산합니다')
        base = U.compute_baseline(df)
        base.to_parquet(U.out('normal_baseline.parquet'), index=False)

    print('[2] 상태 희소성 모델')
    state_model = U.fit_state_model(df)
    df['state_rarity'] = U.state_rarity(df, state_model)

    rng = np.random.default_rng(C.SEED)
    models, parts = {}, []
    for state in U.STATES:
        idx = np.flatnonzero(df['state'].values == state)
        sub = df.iloc[idx]
        print(f'[3] {state}: {len(sub):,}행 → z 변환', flush=True)
        Z, OUTR, AV, feats = U.zscores(sub, base, state)
        range_score = OUTR.sum(1) / np.maximum(AV.sum(1), 1)
        vb = None
        if C.USE_VESSEL_BASELINE:
            vb = U.fit_vessel_baseline(sub['vessel'].values, Z, feats)
            Z = U.apply_vessel_baseline(sub['vessel'].values, Z, vb, feats)
            print(f'    선박별 기준 적용 (선박 {len(vb[2]):,}척)')
        n_fit = min(C.IF_SAMPLE, len(sub))
        fit_idx = rng.choice(len(sub), n_fit, replace=False)
        print(f'    Isolation Forest 학습 (표본 {n_fit:,}행, 특징 {len(feats)}개)', flush=True)
        m = IsolationForest(n_estimators=C.IF_TREES, max_samples=min(C.IF_MAX_SAMPLES, n_fit),
                            random_state=C.SEED, n_jobs=C.N_JOBS).fit(Z[fit_idx])
        if_raw = np.empty(len(sub), dtype='float32')
        for s in range(0, len(sub), 500_000):
            if_raw[s:s + 500_000] = -m.score_samples(Z[s:s + 500_000])
        models[state] = dict(model=m, features=feats, vessel_baseline=vb)

        res = sub[KEEP + ['state_rarity']].reset_index(drop=True)
        res['if_raw'] = if_raw
        res['range_score'] = range_score.astype('float32')
        for k, v in U.top_reasons(Z, feats).items():
            res[k] = v
        parts.append(res)
        del Z, OUTR, AV

    out = pd.concat(parts, ignore_index=True).sort_values(['vessel', 't']).reset_index(drop=True)
    out.to_parquet(U.out('scores_if.parquet'), index=False)
    joblib.dump(dict(models=models, state_model=state_model, ports=ports), U.out('model_if.joblib'))

    print('[4] 요약')
    top = out['if_raw'] >= out.groupby('grp')['if_raw'].transform(lambda s: s.quantile(0.99))
    print('  그룹별 상위 1% 행에서 자주 나온 이상 사유:')
    print(out.loc[top, 'reason1'].replace('', '(뚜렷한 단일 사유 없음)').value_counts().head(8).to_string())
    print(f'  저장: {U.out("scores_if.parquet")} ({len(out):,}행)')
    print(f'완료 ({time.time() - t0:.0f}초)')


if __name__ == '__main__':
    main()
