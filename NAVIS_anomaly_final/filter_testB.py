# -*- coding: utf-8 -*-
"""
9·10단계 준비. 테스트셋 원본 CSV(약 7GB)에서 대상 선박 100척의 행만 뽑는다.
  python filter_testB.py      → 테스트B 시트 선박 → testB_raw_1.csv ~ testB_raw_6.csv (9단계 이상탐지)
  python filter_testB.py A    → 테스트A 시트 선박 → testA_raw_1.csv ~ testA_raw_6.csv (10단계 조업형태 분류)
입력: test_config.TEST_RAW_DIR/vpass_*월 데이터.csv (테스트셋 폴더가 없으면 NAVIS_DATA 학습 원본), test_config.SUBMIT_SAMPLE
출력: test_config.TEST_DIR
실행 시간: 2~3분
"""
import os
import sys
import re
import glob
import time
import openpyxl
import unicodedata

import navis_paths as P

import test_config as T


def main(sheet='B'):
    wb = openpyxl.load_workbook(T.SUBMIT_SAMPLE, read_only=True)
    ids = {str(r[0]).strip() for r in wb[f'테스트{sheet}'].iter_rows(min_row=2, values_only=True) if r[0]}
    print(f'테스트{sheet} 대상 선박 {len(ids)}척')
    os.makedirs(T.TEST_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(T.TEST_RAW_DIR, 'vpass_*.csv')))
    if not files:                                       # 하위 폴더까지 (테스트셋 폴더가 없으면 NAVIS_DATA 학습 원본)
        files = P.raw_files(T.TEST_RAW_DIR)
    print(f'원본 {len(files)}개: {T.TEST_RAW_DIR}')
    if not files:
        raise FileNotFoundError(f'테스트셋 CSV가 없습니다: {T.TEST_RAW_DIR}')
    for src in files:
        name = unicodedata.normalize('NFC', os.path.basename(src))   # Mac zip 한글 파일명(NFD) 대비
        m = re.search(r'(\d+)월', name)
        tag = m.group(1) if m else os.path.splitext(name)[0]
        dst = os.path.join(T.TEST_DIR, f'test{sheet}_raw_{tag}.csv')
        t0 = time.time(); n = k = 0
        with open(src, 'r', encoding='utf-8-sig', newline='') as f, open(dst, 'w', encoding='utf-8', newline='') as g:
            g.write(f.readline())                       # 헤더
            for line in f:
                n += 1
                a = line.find(','); b = line.find(',', a + 1)
                if line[a + 1:b] in ids:                # 두 번째 컬럼 = rfid_id
                    g.write(line); k += 1
        print(f'  {os.path.basename(src)}: {n:,}행 중 {k:,}행 → {os.path.basename(dst)} ({time.time() - t0:.0f}초)', flush=True)


if __name__ == '__main__':
    main((sys.argv[1] if len(sys.argv) > 1 else 'B').upper())
