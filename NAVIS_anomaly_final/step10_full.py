# -*- coding: utf-8 -*-
"""
10단계 (테스트셋 전체). 테스트셋 csv에 있는 '모든' 선박에 대해
  테스트A: 조업형태 예측 (분류 모델 — cls_bridge.predict_vessels, 학습 원본 전체로 학습)
  테스트B: 사고시각·이상항적 점수·시각1~3 (V2.5 이상탐지 — step9 채점기 그대로)
를 만든다. 제출샘플.xlsx 의 선박 목록이 아니라 테스트셋 전체가 대상이다.

테스트셋(약 7GB)을 한 번에 올리면 메모리가 모자라서, 선박 번호로 N_BUCKETS 묶음으로 나눠 차례로 처리한다.
묶음마다 결과를 저장하므로 중간에 멈춰도 다시 실행하면 끝난 묶음은 건너뛴다.

입력: test_config.TEST_RAW_DIR/vpass_*월 데이터.csv, 0~7단계 결과(out/NAVIS_anomaly)
출력: test_config.TEST_DIR/full/
  full_b??_m?.csv            선박 묶음별 원본 (나누기 결과, 다시 실행할 때 재사용)
  b??_A.csv, b??_B.json      묶음별 테스트A 예측, 테스트B 행
  b??_S.parquet, b??_E.csv   묶음별 시간 위치(대시보드 지도용), 사건 목록
  full_testA.csv             전체 테스트A (rfid_id, 예측 조업형태, 예측조업형태 확률)
  full_testB.json            전체 테스트B 행 [rfid_id, 사고시각, 점수, 시각1, 점수1, ...]
  full_testB_근거.csv         선박별 사고시각 근거
  full_testB_scores.parquet  선박·시간별 위치(xe, ye)  ┐ 대시보드(viz/build_dashboard.py) 테스트B 화면용
  full_testB_events.csv      사건 목록                 ┘
→ 최종 제출 파일은 레포 루트의 make_submission.py 가 만든다.
실행 시간: 1~2시간 (선박 약 7,800척)
"""
import glob
import json
import os
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

import anomaly_config as C
import navis_paths as P
import test_config as T

N_BUCKETS = int(os.environ.get('NAVIS_FULL_BUCKETS', '16'))
FULL_DIR = os.path.join(T.TEST_DIR, 'full')


def test_files():
    files = sorted(glob.glob(os.path.join(T.TEST_RAW_DIR, 'vpass_*.csv')))
    if not files:                                       # 하위 폴더까지 (테스트셋 폴더가 없으면 NAVIS_DATA 원본)
        files = P.raw_files(T.TEST_RAW_DIR)
    if not files:
        raise FileNotFoundError(f'테스트셋 CSV가 없습니다: {T.TEST_RAW_DIR}')
    return files


def bucket_of(rfid):
    return int(rfid) % N_BUCKETS if rfid.isdigit() else sum(map(ord, rfid)) % N_BUCKETS


def split(files):
    """테스트셋 csv → 선박 묶음별 csv. 이미 같은 파일로 나눠 두었으면 건너뜀."""
    mark = os.path.join(FULL_DIR, 'split_done.json')
    fp = [[os.path.basename(f), os.path.getsize(f), int(os.path.getmtime(f))] for f in files] + [N_BUCKETS]
    if os.path.exists(mark) and json.load(open(mark, encoding='utf-8')).get('fingerprint') == fp:
        print(f'[1] 나누기: 이미 끝남 ({FULL_DIR})')
        return
    print(f'[1] 테스트셋 {len(files)}개 → 선박 {N_BUCKETS}묶음', flush=True)
    for src in files:
        name = unicodedata.normalize('NFC', os.path.basename(src))
        m = re.search(r'(\d+)월', name)
        tag = m.group(1) if m else os.path.splitext(name)[0]
        t0 = time.time(); n = 0
        with open(src, 'r', encoding='utf-8-sig', newline='') as f:
            head = f.readline()
            outs = []
            for b in range(N_BUCKETS):
                g = open(os.path.join(FULL_DIR, f'full_b{b:02d}_m{tag}.csv'), 'w', encoding='utf-8', newline='')
                g.write(head)
                outs.append(g)
            for line in f:
                a = line.find(','); c = line.find(',', a + 1)
                rid = line[a + 1:c].strip()
                if not rid:
                    continue
                outs[bucket_of(rid)].write(line); n += 1
            for g in outs:
                g.close()
        print(f'  {name}: {n:,}행 ({time.time() - t0:.0f}초)', flush=True)
    json.dump({'fingerprint': fp}, open(mark, 'w', encoding='utf-8'))


def sheet_rows(S, report_score, fmt, track_h):
    """step9b 와 같은 규칙: 사고시각 = 이상항적 점수 최대 시간, 시각1~ = 직전 track_h 시간 안의 기록(시간 순)"""
    S = S.sort_values(['vessel', 't']).reset_index(drop=True)
    S['report'] = report_score(S)
    rows, detail = {}, []
    for v, g in S.groupby('vessel', sort=False):
        o = np.lexsort((g['t'].values, -g['final_score'].values, -g['report'].values))
        acc = g.iloc[int(o[0])]
        tr = g[(g['t'] >= acc['t'] - track_h) & (g['t'] < acc['t'])]
        row = [v, fmt(acc['t']), round(float(acc['report']), 4)]
        for t, r in zip(tr['t'].values, tr['report'].values):
            row += [fmt(t), round(float(r), 4)]
        rows[v] = row
        detail.append(dict(rfid_id=v, 사고시각=fmt(acc['t']), 이상항적점수=round(float(acc['report']), 4),
                           final_score=round(float(acc['final_score']), 4),
                           사건종류=acc['event_kind'], 사건등급=acc['event_level'], 주된사유=acc['reason1'],
                           모항거리_km=acc['dist_port_km'], 업종=acc['code'], 업종근거=acc['code_source'],
                           추적시간수=len(tr), 테스트기록수=len(g)))
    return rows, detail


def first_seen(files):
    """선박별 첫 기록 시각 (점수를 못 매긴 선박의 사고시각 대체값)"""
    out = {}
    for f in files:
        d = pd.read_csv(f, usecols=[C.COL_DATE, C.COL_VESSEL, C.COL_HOUR], dtype=str, encoding='utf-8-sig')
        d = d.dropna()
        d['k'] = d[C.COL_DATE] + ' ' + d[C.COL_HOUR].str.zfill(2)
        for v, k in d.groupby(C.COL_VESSEL)['k'].min().items():
            out[v] = min(out.get(v, k), k)
    return out


def main():
    T0 = time.time()
    os.makedirs(FULL_DIR, exist_ok=True)
    split(test_files())

    import step9_testB as S9
    import step9b_submission as S9b
    import cls_bridge
    R = None
    for b in range(N_BUCKETS):
        fa, fb = os.path.join(FULL_DIR, f'b{b:02d}_A.csv'), os.path.join(FULL_DIR, f'b{b:02d}_B.json')
        fs, fe = os.path.join(FULL_DIR, f'b{b:02d}_S.parquet'), os.path.join(FULL_DIR, f'b{b:02d}_E.csv')
        if all(os.path.exists(f) for f in (fa, fb, fs, fe)):
            print(f'\n[묶음 {b + 1}/{N_BUCKETS}] 이미 끝남 → 건너뜀')
            continue
        files = sorted(glob.glob(os.path.join(FULL_DIR, f'full_b{b:02d}_m*.csv')))
        print(f'\n[묶음 {b + 1}/{N_BUCKETS}] ({(time.time() - T0) / 60:.0f}분 경과)', flush=True)
        if R is None:
            R = S9.load_reference()
        # 테스트A: 분류 모델로 이 묶음의 모든 선박 업종 예측 (학습 원본 전체로 학습한 모델, 한 번만 학습)
        tp = cls_bridge.predict_vessels(files)
        # 테스트B: 학습 데이터에 없는 선박의 업종도 같은 예측을 쓴다
        S, E = S9.score_files(R, files, tpred=tp['pred_type'].to_dict(), t0=time.time())
        rows, detail = sheet_rows(S, S9b.report_score, S9b.fmt, T.TRACK_H)
        S[['vessel', 'date', 'hour', 'xe', 'ye']].to_parquet(fs, index=False)
        (E.drop(columns=['t_start', 't_end'], errors='ignore') if len(E) else pd.DataFrame(
            columns=['vessel', 'event_id', 'start', 'end', 'level', 'kind'])).to_csv(fe, index=False, encoding='utf-8-sig')
        # 점수를 못 매긴 선박(유효한 항적 없음): 첫 기록 시각, 점수 0
        seen = first_seen(files)
        for v in sorted(set(seen) - set(rows)):
            rows[v] = [v, seen[v], 0.0]
            detail.append(dict(rfid_id=v, 사고시각=seen[v], 이상항적점수=0.0, 사건종류='유효한 항적 없음'))
        pd.DataFrame(detail).to_csv(os.path.join(FULL_DIR, f'b{b:02d}_근거.csv'), index=False, encoding='utf-8-sig')
        json.dump(rows, open(fb, 'w', encoding='utf-8'), ensure_ascii=False)
        A = tp[['pred_type', 'pred_prob']].reindex(sorted(seen)).rename_axis('rfid_id').reset_index()
        A.to_csv(fa, index=False, encoding='utf-8-sig')        # 마지막에 저장 → 이 파일이 있으면 묶음 끝
        print(f'  묶음 {b + 1}: 선박 {len(seen):,}척 (테스트A 예측 없음 {int(A.pred_type.isna().sum())}척)', flush=True)

    # ---- 합치기 ----
    A = pd.concat([pd.read_csv(os.path.join(FULL_DIR, f'b{b:02d}_A.csv'), dtype={'rfid_id': str})
                   for b in range(N_BUCKETS)], ignore_index=True)
    miss = A['pred_type'].isna()
    if miss.any():                         # 분류 특징을 못 만든 선박: 학습 데이터에서 가장 많은 업종
        HS = cls_bridge.train_hourly_stats()
        share = HS.groupby('vessel_id')['reg_type'].first().value_counts(normalize=True)
        A.loc[miss, 'pred_type'] = share.index[0]
        A.loc[miss, 'pred_prob'] = round(float(share.iloc[0]), 4)
        print(f'[알림] 테스트A 예측이 없는 선박 {int(miss.sum())}척 → 학습 데이터 최다 업종({share.index[0]})')
    A['pred_prob'] = A['pred_prob'].astype(float).round(4)
    A = A.sort_values('rfid_id', key=lambda s: s.str.zfill(20)).reset_index(drop=True)
    A.to_csv(os.path.join(FULL_DIR, 'full_testA.csv'), index=False, encoding='utf-8-sig')
    B = {}
    for b in range(N_BUCKETS):
        B.update(json.load(open(os.path.join(FULL_DIR, f'b{b:02d}_B.json'), encoding='utf-8')))
    rowsB = [B[k] for k in sorted(B, key=lambda s: s.zfill(20))]
    json.dump(rowsB, open(os.path.join(FULL_DIR, 'full_testB.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    pd.concat([pd.read_csv(os.path.join(FULL_DIR, f'b{b:02d}_근거.csv'), dtype={'rfid_id': str})
               for b in range(N_BUCKETS)]).to_csv(os.path.join(FULL_DIR, 'full_testB_근거.csv'),
                                                  index=False, encoding='utf-8-sig')
    pd.concat([pd.read_parquet(os.path.join(FULL_DIR, f'b{b:02d}_S.parquet')) for b in range(N_BUCKETS)],
              ignore_index=True).to_parquet(os.path.join(FULL_DIR, 'full_testB_scores.parquet'), index=False)
    pd.concat([pd.read_csv(os.path.join(FULL_DIR, f'b{b:02d}_E.csv'), dtype={'vessel': str}, encoding='utf-8-sig')
               for b in range(N_BUCKETS)], ignore_index=True).to_csv(
        os.path.join(FULL_DIR, 'full_testB_events.csv'), index=False, encoding='utf-8-sig')
    print(f'\n[완료] 테스트A {len(A):,}척 | 테스트B {len(rowsB):,}척 | {(time.time() - T0) / 60:.0f}분')
    print(f'  예측 업종 분포: {A["pred_type"].value_counts().to_dict()}')
    print(f'  저장: {FULL_DIR}')


if __name__ == '__main__':
    main()
