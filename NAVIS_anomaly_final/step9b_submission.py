# -*- coding: utf-8 -*-
"""
9단계 (테스트) 후반.
[보정] test_config.CALIBRATE=1이면(기본 0) 마지막에 s^k 보정: 주의 기준 0.9506 → 0.5.
 testB_scores.parquet → 제출 형식 sheet2(테스트B)

이상항적 점수 (시간별, 0~1)
  기본은 최종 이상점수(final_score). 여기에 V2.5 규칙 사건을 반영한다.
  - '경보' 등급 사건(바다 위 신호 단절·장시간 정지·지속 이상)에 속한 시간:
        max(final_score, 0.999 + 0.001 × 심각도)     → 경보 임계값(0.9990) 이상, 1 이하
        심각도 = 끊긴 시간 / 72h, 정지 시간 / 24h, 지속 시간 / 24h (각각 최대 1)
  - '주의' 등급 사건에 속한 시간: max(final_score, 주의 임계값 0.9506)
  - 사건이 없는 모항 2km 안 시간: min(final_score, 0.9505) — 항구 안 회전·GPS 흔들림은 사고 대상이 아님
    (V2.4에서 점수 경보의 58%가 항구 안이었던 문제와 같은 처리. NAVIS_SEA_ONLY=0 이면 끔)
사고시각 = 이상항적 점수가 가장 높은 시간 (같으면 원래 최종 점수가 높은 시간, 그다음 이른 시간)
  신호 단절 사건은 끊기기 직전 마지막 기록 시간, 정지 사건은 정지가 시작된 시간이 된다.
시각1, 시각2, … = 사고시각 직전 TRACK_H 시간(기본 24시간) 안에 기록이 있는 시간, 시간 순서
"""
import os
import sys
import numpy as np
import pandas as pd
import openpyxl

import json
import anomaly_config as C
import anomaly_common as U
import test_config as T

TRACK_H = T.TRACK_H
SEA_ONLY = T.SEA_ONLY
CALIBRATE = getattr(T, "CALIBRATE", False)        # 모항 2km 안 시간은 주의 미만으로 (V2.4 SCORE_SEA_ONLY와 같은 이유)
with open(U.out('final_thresholds.json'), encoding='utf-8') as _fp:
    _FT = json.load(_fp)
CAUTION, ALARM = _FT['caution'], _FT['alarm']
SEA_KM = C.SCORE_SEA_KM


def report_score(S):
    s = S['final_score'].astype(float).clip(0, 1).values.copy()
    kind = S['event_kind'].fillna('').astype(str)
    lvl = S['event_level'].fillna('').astype(str).to_numpy()
    loss = S['loss_h'].fillna(0).values
    stop = S['stop_h'].fillna(0).values
    # 사건 길이(지속 이상용)
    hours = S.groupby('event_id')['t'].transform('size').where(S['event_id'] >= 0, 0).values
    sev = np.zeros(len(S))
    is_loss = kind.str.contains('단절').to_numpy()
    is_stop = kind.str.contains('정지').to_numpy()
    sev = np.where(is_loss, np.minimum(loss / 72, 1), sev)
    sev = np.maximum(sev, np.where(is_stop, np.minimum(stop / 24, 1), 0))
    sev = np.where(~is_loss & ~is_stop & (kind.to_numpy() != ''), np.minimum(hours / 24, 1), sev)
    al = lvl == '경보'
    ca = (lvl == '주의') | (lvl == '주의(반복)')
    s = np.where(al, np.maximum(s, ALARM + (1 - ALARM) * sev), s)
    s = np.where(ca, np.maximum(s, CAUTION), s)
    if SEA_ONLY:
        d = S['dist_port_km'].values.astype(float)
        port = np.isfinite(d) & (d <= SEA_KM) & ~al & ~ca
        s = np.where(port, np.minimum(s, CAUTION - 1e-4), s)
    s = np.minimum(s, 1.0)
    if CALIBRATE:                      # 주의 기준 → 0.5, 경보 기준 → 약 0.99, 정상 중앙값 → 0에 가까움 (순서 유지)
        k = np.log(0.5) / np.log(CAUTION)
        s = np.power(np.clip(s, 0, 1), k)
    return s


def fmt(t):
    ts = pd.Timestamp(int(t) * 3600, unit='s')
    return f'{ts:%Y%m%d} {ts:%H}'


def main(scores_path, sample_path, out_path):
    S = pd.read_parquet(scores_path).sort_values(['vessel', 't']).reset_index(drop=True)
    S['report'] = report_score(S)
    wb = openpyxl.load_workbook(sample_path)
    B = [str(r[0]) for r in wb['테스트B'].iter_rows(min_row=2, values_only=True) if r[0]]
    rows, detail = [], []
    for v in B:
        g = S[S['vessel'] == v]
        if len(g) == 0:
            rows.append([v]); detail.append(dict(rfid_id=v, note='테스트셋에 기록 없음')); continue
        # 이상항적 점수 최대, 같으면 원래 최종 점수가 높은 시간, 그것도 같으면 이른 시간
        o = np.lexsort((g['t'].values, -g['final_score'].values, -g['report'].values))
        i = int(o[0])
        acc = g.iloc[i]
        tr = g[(g['t'] >= acc['t'] - TRACK_H) & (g['t'] < acc['t'])]
        row = [v, fmt(acc['t']), round(float(acc['report']), 4)]
        for _, r in tr.iterrows():
            row += [fmt(r['t']), round(float(r['report']), 4)]
        rows.append(row)
        detail.append(dict(rfid_id=v, 사고시각=fmt(acc['t']), 이상항적점수=round(float(acc['report']), 4),
                           final_score=round(float(acc['final_score']), 4), 사건종류=acc['event_kind'],
                           사건등급=acc['event_level'], 주된사유=acc['reason1'], 끊긴시간_h=acc['loss_h'],
                           정지시간_h=acc['stop_h'], 모항거리_km=acc['dist_port_km'], 업종=acc['code'],
                           업종근거=acc['code_source'], 추적시간수=len(tr), 테스트기록수=len(g)))
    width = max(len(r) for r in rows)
    n_track = (width - 3) // 2
    header = ['rfid_id', '사고시각', '이상항적 점수']
    for k in range(1, n_track + 1):
        header += [f'시각{k}', f'시각{k} 이상항적 점수']
    out = openpyxl.Workbook()
    wa = out.active; wa.title = '테스트A'
    wa.append(['rfid_id', '예측 조업형태', '예측조업형태 확률'])
    for r in wb['테스트A'].iter_rows(min_row=2, values_only=True):
        if r[0]:
            wa.append([str(r[0]), None, None])           # 분류모델 담당이 채움
    wbB = out.create_sheet('테스트B')
    wbB.append(header)
    for r in rows:
        wbB.append(r)
    out.save(out_path)
    D = pd.DataFrame(detail)
    D.to_csv(os.path.splitext(out_path)[0] + '_근거.csv', index=False, encoding='utf-8-sig')
    print(f'저장: {out_path} (선박 {len(rows)}척, 추적 시각 최대 {n_track}개)')
    print(D['사건종류'].fillna('(사건 없음: 점수 최대 시간)').value_counts().to_string())


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        main(*sys.argv[1:4])
    else:
        main(os.path.join(T.TEST_OUT, 'testB_scores.parquet'), T.SUBMIT_SAMPLE,
             os.path.join(T.TEST_OUT, 'testB_submission.xlsx'))
