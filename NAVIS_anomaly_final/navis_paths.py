# -*- coding: utf-8 -*-
"""
경로 설정 (모든 단계가 이 파일 하나에서 경로를 가져간다)
------------------------------------------------
절대경로를 코드에 적지 않는다. 우선순위:
  1) 환경 변수            예) $env:NAVIS_RAW_DIR = "D:\\data\\vpass_1월_6월 데이터"
  2) navis_paths.json     레포 루트 또는 이 코드 폴더에 두는 '내 컴퓨터용' 설정 (깃에 올리지 않음)
                          → navis_paths.example.json 을 복사해서 필요한 항목만 적으면 된다
  3) 학습 원본만: 팀 공통 환경 변수 NAVIS_DATA (분류 담당 run_submission.py 와 같은 데이터 폴더)
  4) 기본값               레포 안의 상대경로 (아래 표)

기본 레포 구조 (ROOT = 이 코드 폴더의 상위 폴더 = 레포 루트)
  ROOT/
    NAVIS_anomaly_final/          ← 이 코드
    NAVIS_features/               ← navis_features.py 결과 (features_vpass_*.parquet, thresholds.json)
    NAVIS_classifier/navis_anomaly/ ← 분류 담당 패키지 (classifier.py 등)
    data/vpass_1월_6월 데이터/      ← 학습 원본 CSV
    data/테스트셋/                  ← 테스트 원본 CSV
    data/제출샘플.xlsx
    out/NAVIS_anomaly/            ← 이상탐지 결과 (scores_final.parquet, cache/, test_scoring/ ...)

json·환경 변수의 상대경로는 ROOT 기준으로 해석한다.
예전 방식(NAVIS_BASE 하나로 BASE/vpass_1월_6월 데이터, BASE/NAVIS_features, BASE/NAVIS_anomaly)도 그대로 동작한다.
"""
import json
import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.environ.get('NAVIS_ROOT', os.path.dirname(CODE_DIR)))

# key: (환경 변수, 기본값 = ROOT 기준 상대경로)
_DEFAULTS = {
    'raw_dir':       ('NAVIS_RAW_DIR', os.path.join('data', 'vpass_1월_6월 데이터')),
    'feat_dir':      ('NAVIS_FEAT_DIR', 'NAVIS_features'),
    'out_dir':       ('NAVIS_OUT_DIR', os.path.join('out', 'NAVIS_anomaly')),
    'test_raw_dir':  ('NAVIS_TEST_RAW_DIR', os.path.join('data', '테스트셋')),
    'submit_sample': ('NAVIS_SUBMIT_SAMPLE', os.path.join('data', '제출샘플.xlsx')),
    'cls_repo':      ('NAVIS_CLS_REPO', 'NAVIS_classifier'),
}
# 예전 방식: NAVIS_BASE (또는 json의 "base") 아래 폴더 이름
_BASE_LAYOUT = {'raw_dir': 'vpass_1월_6월 데이터', 'feat_dir': 'NAVIS_features', 'out_dir': 'NAVIS_anomaly'}


def _load_local():
    for d in (CODE_DIR, ROOT):
        p = os.path.join(d, 'navis_paths.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                return json.load(f), p
    return {}, None


LOCAL, LOCAL_FILE = _load_local()


def _abs(p):
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


def get(key):
    env, default = _DEFAULTS[key]
    if os.environ.get(env):
        return _abs(os.environ[env])
    if LOCAL.get(key):
        return _abs(LOCAL[key])
    base = os.environ.get('NAVIS_BASE') or LOCAL.get('base')
    if base and key in _BASE_LAYOUT:
        return _abs(os.path.join(base, _BASE_LAYOUT[key]))
    data = os.environ.get('NAVIS_DATA')
    if key == 'raw_dir' and data:          # 팀 공통 데이터 폴더
        return _abs(data)
    if key in ('submit_sample', 'test_raw_dir') and data and not os.path.exists(_abs(default)):
        found = _find_in_data(key, data)   # 레포 기본 위치에 없으면 NAVIS_DATA 폴더 안에서 찾음
        if found:
            return found
        if key == 'test_raw_dir':          # 테스트셋 폴더가 따로 없으면 학습 원본에서 테스트 선박의 행을 뽑는다
            return _abs(data)
    return _abs(default)


def _find_in_data(key, data):
    """NAVIS_DATA 아래(하위 폴더 포함)에서
       submit_sample: 이름에 '제출샘플'(또는 '제출 샘플')이 든 xlsx
       test_raw_dir : 경로에 '테스트'가 든 폴더 중 vpass_*.csv 가 바로 들어 있는 폴더"""
    import unicodedata
    nfc = lambda s: unicodedata.normalize('NFC', s)
    for d, dirs, files in sorted(os.walk(data)):
        dirs.sort()
        if key == 'submit_sample':
            for f in sorted(files):
                n = nfc(f)
                if n.lower().endswith('.xlsx') and not n.startswith('~$') and ('제출샘플' in n or '제출 샘플' in n):
                    return os.path.join(d, f)
        elif '테스트' in nfc(os.path.relpath(d, data)) and any(
                nfc(f).startswith('vpass_') and f.lower().endswith('.csv') for f in files):
            return d
    return None


RAW_DIR = get('raw_dir')
FEAT_DIR = get('feat_dir')
OUT_DIR = get('out_dir')
TEST_RAW_DIR = get('test_raw_dir')
SUBMIT_SAMPLE = get('submit_sample')
CLS_REPO = get('cls_repo')

RAW_PATTERN = 'vpass_*월 데이터.csv'


def raw_files(folder=None):
    """학습 원본 월별 CSV 목록. 폴더 바로 아래에 없으면 하위 폴더까지 찾는다('테스트'가 든 폴더는 제외)
    Mac에서 zip을 풀면 한글 파일명이 자모 분리형(NFD)으로 저장될 수 있어, 이름을 NFC로 맞춘 뒤 비교한다."""
    import fnmatch
    import glob
    import unicodedata
    nfc = lambda s: unicodedata.normalize('NFC', s)
    folder = folder or RAW_DIR

    def match(cands):
        return sorted((f for f in cands if fnmatch.fnmatch(nfc(os.path.basename(f)), RAW_PATTERN)),
                      key=lambda f: nfc(os.path.basename(f)))
    files = match(glob.glob(os.path.join(folder, '*.csv')))
    if not files:
        files = [f for f in match(glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True))
                 if '테스트' not in nfc(os.path.relpath(f, folder))]
    return files


def show():
    print(f'[경로] 설정 파일: {LOCAL_FILE or "(없음 → 레포 기본 구조)"}')
    for k in _DEFAULTS:
        p = get(k)
        print(f'  {k:<14} {"있음" if os.path.exists(p) else "없음"}  {p}')


if __name__ == '__main__':
    show()
