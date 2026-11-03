# -*- coding: utf-8 -*-
"""
5단계. '예측 업종 ≠ 등록 업종' 지표 추가

입력: 분류 담당(혜승님) 모델을 import 해서 함수로 부른다 — cls_bridge.train_vessel_oof()
  결과 파일을 주고받지 않는다. 학습 선박별 업종 확률은 선박 단위 교차검증(OOF)으로 계산
  (그 선박을 학습에 쓰지 않은 모델의 확률이어야 불일치 지표가 공정하다).
  분류 패키지 위치: navis_paths.json 의 cls_repo (기본: 레포/NAVIS_classifier)

지표 (선박 단위, 시간에 따라 변하지 않음)
  p_reg       등록 업종의 예측 확률
  pred_code   확률이 가장 높은 업종
  mismatch    max(0, (MISMATCH_TAU - p_reg) / MISMATCH_TAU)
              → 등록 업종 확률이 MISMATCH_TAU보다 낮을 때만 0~1로 켜진다.
              1차 분류 성능(macro-F1 0.51)에서는 자망↔연안복합 혼동이 커서,
              단순히 '예측 ≠ 등록'을 쓰면 정상 선박 다수가 걸리기 때문에 약하게 반영한다.
최종 점수
  final_score = anomaly_score + W_MISMATCH × mismatch × (1 − anomaly_score)
  → 불일치가 클수록 남은 여유만큼 점수를 끌어올리되, 1을 넘지 않는다.

출력: scores_final.parquet (선박·시간별 최종 이상점수), vessel_mismatch.csv
"""
import os
import numpy as np
import pandas as pd

import anomaly_config as C
import anomaly_common as U


def class_probs():
    """분류 모델 함수 호출 → 선박 × 업종 확률. 분류 패키지가 없으면 None (불일치 지표 0)"""
    try:
        import cls_bridge
        return cls_bridge.train_vessel_oof()
    except (ImportError, FileNotFoundError) as e:
        print(f'  [주의] 분류 모델을 부를 수 없어 불일치 지표를 0으로 둡니다\n  {e}')
        return None


def main():
    df = pd.read_parquet(U.out('scores_combined.parquet'))
    vessels = df.groupby('vessel')['code'].first()
    P = class_probs()
    if P is not None:
        v = pd.DataFrame({'code': vessels}).join(P, how='left')
        missing = v[C.CLASSES].isna().any(axis=1)
        if missing.any():
            print(f'  [주의] 분류 확률이 없는 선박 {int(missing.sum())}척 → 불일치 0으로 처리')
        p = v[C.CLASSES].to_numpy()
        reg_idx = v['code'].map({c: i for i, c in enumerate(C.CLASSES)}).to_numpy()
        v['p_reg'] = np.where(missing, np.nan, p[np.arange(len(v)), reg_idx])
        v['pred_code'] = np.where(missing, None, np.array(C.CLASSES, dtype=object)[np.nan_to_num(p, nan=-1).argmax(1)])
        v['mismatch'] = np.clip((C.MISMATCH_TAU - v['p_reg']) / C.MISMATCH_TAU, 0, 1).fillna(0)
        v[['code', 'pred_code', 'p_reg', 'mismatch']].to_csv(U.out('vessel_mismatch.csv'), encoding='utf-8-sig')
        hv = v[~missing]
        print(f'  확률이 있는 선박 {len(hv):,}척 중 예측 ≠ 등록: {(hv.pred_code != hv.code).mean():.1%} '
              f'| 실제 불일치 지표가 켜진 선박(p_reg < {C.MISMATCH_TAU}): {(v.mismatch > 0).mean():.1%}')
        print(v.groupby('code')['mismatch'].apply(lambda s: (s > 0).mean()).rename('불일치 선박 비율').round(3).to_string())
        df = df.merge(v[['p_reg', 'pred_code', 'mismatch']], left_on='vessel', right_index=True, how='left')
    else:
        df['p_reg'], df['pred_code'], df['mismatch'] = np.nan, None, 0.0
    df['mismatch'] = df['mismatch'].fillna(0).astype('float32')
    df['final_score'] = (df['anomaly_score'] + C.W_MISMATCH * df['mismatch'] * (1 - df['anomaly_score'])).astype('float32')
    df.to_parquet(U.out('scores_final.parquet'), index=False)
    print(f'  저장: {U.out("scores_final.parquet")}')


if __name__ == '__main__':
    main()
