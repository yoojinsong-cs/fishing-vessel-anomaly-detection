# -*- coding: utf-8 -*-
"""테스트셋 채점(9단계) 설정. 폴더 경로는 navis_paths.py(환경 변수 > navis_paths.json > 레포 기본 구조)에서 가져온다."""
import os
import anomaly_config as C

import navis_paths as P
# 테스트셋 원본 CSV 폴더 (vpass_1월 데이터.csv ~ vpass_6월 데이터.csv) — 경로는 navis_paths.py 에서
TEST_RAW_DIR = P.TEST_RAW_DIR
# 제출 샘플 (테스트A·테스트B 시트의 rfid_id 목록을 여기서 읽음)
SUBMIT_SAMPLE = P.SUBMIT_SAMPLE
# 중간 파일·결과 폴더 (기본: 결과 폴더 안 test_scoring)
TEST_DIR = os.environ.get('NAVIS_TEST_DIR', os.path.join(C.OUT_DIR, 'test_scoring'))
TEST_OUT = os.environ.get('NAVIS_TEST_OUT', os.path.join(TEST_DIR, 'out'))
# 시각1, 시각2, … 에 넣을 사고시각 직전 시간 범위 (시간)
TRACK_H = int(os.environ.get('NAVIS_TRACK_H', '24'))
# 1이면 사건이 없는 모항 2km 안 시간은 주의 미만으로 둔다
SEA_ONLY = os.environ.get('NAVIS_SEA_ONLY', '1') == '1'
# 1이면 점수를 보정한다: 주의 기준(상위 5%)이 0.5가 되도록 s' = s^k (k = ln0.5 / ln 주의 기준)
#   V2.5 최종 점수는 백분위라 정상 시간도 0~1에 고르게 퍼진다(중앙값 약 0.57).
#   0.5 같은 기준으로 '사고 시간'을 판정하면 정상 시간의 절반 이상이 사고로 잡힌다.
#   순서는 그대로라 사고시각은 바뀌지 않고 값만 바뀐다.
#   테스트B 채점 방식이 확인되기 전까지는 0(끔)으로 둔다.
CALIBRATE = os.environ.get('NAVIS_CALIBRATE', '0') == '1'
