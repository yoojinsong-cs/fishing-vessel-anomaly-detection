# -*- coding: utf-8 -*-
"""
4단계 (v2.1). 두 점수 결합 + 상태 희소성 → 0~1 이상점수

  1) IF 원점수와 AE 원점수를 각각 '같은 그룹(업종×시간대×상태) 안에서의 백분위'로 바꾼다.
     → 두 점수의 단위가 같아지고, 업종·시간대 조건이 한 번 더 반영된다.
  2) 둘 중 큰 쪽을 택한다 (COMBINE='max'). AE 점수가 없는 행(점이 너무 적음)은 IF만 쓴다.
  3) [개선 2] 상태 희소성 백분위가 상위 10%에 들면 그만큼 점수를 끌어올린다 (W_STATE).
     [v2.1 B] 단, 활동 행과 바다 위 정박 행에만 적용한다.
  [v2.1 C] AE 점수는 log(오차)를 그룹 기준으로 정규화한 뒤 그 선박의 평소 수준을 빼서(ae_adj) 쓴다.
           3단계 v2.1(--score-only)을 실행했다면 채널별 오차로 AE 사유(거리·회전·정지 패턴)를 붙인다.
  4) 실제 데이터 전체에서의 백분위로 다시 바꿔 anomaly_score(0~1)로 만든다.
     → 0.99 = 전체 레코드 중 가장 이상한 1% 라는 뜻이 되어 임계값을 해석하기 쉽다.

출력: scores_combined.parquet
"""
import os
import time
import joblib
import numpy as np
import pandas as pd

import anomaly_config as C
import anomaly_common as U


def main():
    t0 = time.time()
    df = pd.read_parquet(U.out('scores_if.parquet'))
    ae_path = U.out('scores_ae.parquet')
    if os.path.exists(ae_path):
        ae = pd.read_parquet(ae_path)
        ae = ae[[c for c in ['src_file', 'row_in_file', 'ae_raw'] + U.AE_CH if c in ae.columns]]
        df = df.merge(ae, on=U.KEYS, how='left')
        print(f'  AE 점수 결합: {df.ae_raw.notna().mean():.1%} 행에 AE 점수 있음')
        aeb = U.fit_ae_baseline(df)                       # [v2.1 C]
        df['ae_adj'] = U.apply_ae_baseline(df, aeb)
        joblib.dump(aeb, U.out('model_ae_baseline.joblib'))
        if not any(c in df.columns for c in U.AE_CH):
            print('  [안내] 채널별 AE 오차가 없습니다. AE 사유를 보려면 python step3_seq_ae.py --score-only 를 실행하세요')
    else:
        print('  [주의] scores_ae.parquet 이 없어 IF 점수만으로 결합합니다 (3단계 실행 후 다시 실행하세요)')

    res = U.combine(df, U.make_ref(df))
    for k, v in res.items():
        df[k] = v.astype('float32')
    df['anomaly_score'] = U.global_percentile(df['combined'].values, U.sorted_finite(df['combined'].values)).astype('float32')
    if 'ae_raw' in df.columns:                           # [v2.1 C] AE가 주도한 행에 AE 사유
        ae_lab = U.ae_reason(df, aeb)
        ae_dom = (df['u_ae'].fillna(0) > df['u_if'].fillna(0)).values
        df['reason_ae'] = ae_lab
        df['reason1'] = np.where(ae_dom & (ae_lab != ''), ae_lab, df['reason1'].values)
    if 'u_state' in df:
        up = (df.u_state > C.STATE_TAIL) & (df['sea_context'] if 'sea_context' in df else True)
        print(f'  상태 희소성으로 점수가 올라간 행: {up.mean():.1%} (활동·바다 위 정박 행만)')
    df.to_parquet(U.out('scores_combined.parquet'), index=False)

    if 'u_if' in df and 'u_ae' in df:
        both = df[['u_if', 'u_ae']].dropna()
        print(f'  IF·AE 백분위 상관(스피어만): {both.corr(method="spearman").iloc[0, 1]:.3f}  '
              '(낮을수록 두 점수가 서로 다른 이상을 봄)')
    print(df.groupby('code')['anomaly_score'].describe(percentiles=[.5, .95, .99]).round(3).to_string())
    print(f'  저장: {U.out("scores_combined.parquet")}  ({time.time() - t0:.0f}초)')


if __name__ == '__main__':
    main()
