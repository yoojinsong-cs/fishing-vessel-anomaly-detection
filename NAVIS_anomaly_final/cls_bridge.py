# -*- coding: utf-8 -*-
"""
분류 모델(혜승님 패키지) 연결 — 결과 파일을 읽지 않고, 분류 패키지를 import 해서 함수로 부른다.
------------------------------------------------
이상탐지 쪽에서 쓰는 곳
  - 5단계 step5_mismatch.py : train_vessel_oof()     학습 선박별 업종 확률(교차검증 OOF) → 업종 불일치 지표
  - 9단계 step9_testB.py    : predict_vessels(paths) 테스트 선박별 업종 확률 → 학습 데이터에 없는 선박의 업종

분류 패키지에서 부르는 것 (공개 함수·클래스만 사용, 내부 구현에 의존하지 않음)
  navis_anomaly.raw_sources   iter_raw_sources, read_source     원본 CSV 읽기·정리
  navis_anomaly.hourly_raw    build_hourly                      시간별 통계 (분류 담당 전처리)
  navis_anomaly.vpass_raw     RawConfig
  navis_anomaly.classifier    vessel_features, VesselTypeClassifier(fit / predict)
→ 혜승님이 classifier.py 를 고도화하면(특징·모델·설정) 이상탐지 쪽은 코드 수정 없이 새 모델을 쓴다.

경로: navis_paths.py 의 cls_repo(분류 패키지의 상위 폴더), raw_dir(학습 원본). 절대경로 없음.
캐시: 학습 원본 → 시간별 통계(전처리 중간값)만 결과 폴더 cache/ 에 저장해 두 번째부터 빠르게 한다.
      분류 결과(확률)는 저장하지 않고 매번 함수로 계산한다. 끄기: 환경 변수 NAVIS_CLS_CACHE=0
      원본 파일이나 분류 패키지의 전처리 코드(hourly_raw.py, vpass_raw.py)가 바뀌면 캐시는 자동으로 다시 만든다.

단독 실행(점검): python cls_bridge.py → 선박 단위 교차검증 정확도(= 대회 지표 micro-F1)와 업종별 결과 출력
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import anomaly_config as C
import navis_paths as P

USE_CACHE = os.environ.get('NAVIS_CLS_CACHE', '1') == '1'
N_JOBS = int(os.environ.get('NAVIS_CLS_JOBS', '4'))
N_SPLITS = 5
SEED = 42
_MEM = {}          # 같은 실행 안에서 학습 원본을 두 번 읽지 않도록


# ---------------------------------------------------------------------
# 분류 패키지 import
# ---------------------------------------------------------------------
def cls_package():
    """분류 담당 패키지(navis_anomaly)를 import 해서 모듈들을 돌려준다."""
    if 'pkg' in _MEM:
        return _MEM['pkg']
    repo = P.CLS_REPO
    if not os.path.exists(os.path.join(repo, 'navis_anomaly', 'classifier.py')):
        raise ImportError(f'분류 패키지를 찾을 수 없습니다: {os.path.join(repo, "navis_anomaly")}\n'
                          '  navis_paths.json 의 cls_repo(또는 환경 변수 NAVIS_CLS_REPO)에 '
                          'navis_anomaly/classifier.py 가 들어 있는 폴더의 상위 폴더를 적어 주세요.')
    if repo not in sys.path:
        sys.path.insert(0, repo)          # 이상탐지 폴더 이름(NAVIS_anomaly)과 겹치지 않게 맨 앞에
    from navis_anomaly import classifier, hourly_raw, raw_sources, vpass_raw
    _MEM['pkg'] = (classifier, hourly_raw, raw_sources, vpass_raw)
    return _MEM['pkg']


# ---------------------------------------------------------------------
# 전처리: 분류 담당 코드 그대로 (원본 CSV → 시간별 통계 HS)
# ---------------------------------------------------------------------
def hourly_stats(paths, label=True):
    classifier, hourly_raw, raw_sources, vpass_raw = cls_package()
    rc = vpass_raw.RawConfig(synth_rate=0.0)            # 분류용: 인위적 이상 궤적은 만들지 않음
    parts = []
    for name, opener in raw_sources.iter_raw_sources(list(paths)):
        t0 = time.time()
        df = raw_sources.read_source(opener)
        if not label:
            df = df.drop(columns=['fishing_code'], errors='ignore')
        _, _, HS = hourly_raw.build_hourly(df, rc, mode='train', train_hour_rate=0.0, n_jobs=N_JOBS)
        del df
        parts.append(HS)
        print(f'  [분류 전처리] {name}: 시간 {len(HS):,}개, 선박 {HS["vessel_id"].nunique():,}척 '
              f'({time.time() - t0:.0f}초)', flush=True)
    HS = pd.concat(parts, ignore_index=True).drop_duplicates(['vessel_id', 'H'])
    HS['vessel_id'] = HS['vessel_id'].astype(str)
    HS['month'] = (pd.Timestamp('2000-01-01') + pd.to_timedelta(HS['day'], unit='D')).dt.strftime('%Y-%m')
    return HS


def _fingerprint(paths):
    _, hourly_raw, _, vpass_raw = cls_package()
    items = [(os.path.basename(p), os.path.getsize(p), int(os.path.getmtime(p))) for p in sorted(paths)]
    for mod in (hourly_raw, vpass_raw):                 # 분류 담당 전처리 코드가 바뀌면 캐시 무효
        f = mod.__file__
        items.append((os.path.basename(f), os.path.getsize(f), int(os.path.getmtime(f))))
    return hashlib.md5(json.dumps(items).encode()).hexdigest()


def train_hourly_stats():
    """학습 원본(raw_dir) → 시간별 통계. 캐시가 맞으면 캐시를 쓴다."""
    if 'train_hs' in _MEM:
        return _MEM['train_hs']
    paths = P.raw_files(C.RAW_DIR)
    if not paths:
        raise FileNotFoundError(f'학습 원본이 없습니다: {os.path.join(C.RAW_DIR, C.RAW_GLOB)}')
    cdir = os.path.join(C.OUT_DIR, 'cache')
    cpath, mpath = os.path.join(cdir, 'cls_hourly_train.parquet'), os.path.join(cdir, 'cls_hourly_train.json')
    fp = _fingerprint(paths)
    HS = None
    if USE_CACHE and os.path.exists(cpath) and os.path.exists(mpath):
        if json.load(open(mpath, encoding='utf-8')).get('fingerprint') == fp:
            HS = pd.read_parquet(cpath)
            print(f'  [분류 전처리] 캐시 사용: {cpath}')
    if HS is None:
        print(f'[분류 전처리] 학습 원본 {len(paths)}개 → 시간별 통계 (분류 담당 hourly_raw)', flush=True)
        HS = hourly_stats(paths, label=True)
        if USE_CACHE:
            os.makedirs(cdir, exist_ok=True)
            HS.to_parquet(cpath, index=False)
            json.dump({'fingerprint': fp, 'files': [os.path.basename(p) for p in paths]},
                      open(mpath, 'w', encoding='utf-8'), ensure_ascii=False)
    HS = HS[HS['reg_type'].isin(C.CLASSES)]
    _MEM['train_hs'] = HS
    return HS


# ---------------------------------------------------------------------
# 특징·확률
# ---------------------------------------------------------------------
def _xy(HS):
    classifier = cls_package()[0]
    X = classifier.vessel_features(HS, keys=('vessel_id', 'month'))       # 분류 담당 특징 함수
    w = HS.groupby(['vessel_id', 'month']).size().reindex(X.index)        # 선박×월 활동 시간 수
    y = HS.groupby(['vessel_id', 'month'])['reg_type'].first().reindex(X.index) if 'reg_type' in HS else None
    return X, y, w


def _probs(clf, X):
    """classifier.predict → prob_<업종> 열을 업종 4개 순서로"""
    out = clf.predict(X)
    P_ = pd.DataFrame({c: out[f'prob_{c}'] if f'prob_{c}' in out else 0.0 for c in C.CLASSES}, index=X.index)
    return P_


def _to_vessel(P_, w):
    """선박 × 월 확률 → 선박 확률 (활동 시간 수 가중 평균)"""
    Pv = P_.mul(w.to_numpy(), axis=0).groupby(level='vessel_id').sum()
    Pv = Pv.div(Pv.sum(axis=1), axis=0)
    Pv.index = Pv.index.astype(str)
    Pv.index.name = C.COL_VESSEL
    return Pv


def train_vessel_oof(n_splits=N_SPLITS):
    """
    학습 선박별 업종 확률 — 선박 단위 K-fold 교차검증(OOF).
    같은 선박의 모든 달은 같은 fold. 그 선박을 학습에 쓰지 않은 모델의 확률이라 불일치 지표가 공정하다.
    반환: index = rfid_id(str), columns = C.CLASSES
    """
    if 'oof' in _MEM:
        return _MEM['oof']
    classifier = cls_package()[0]
    HS = train_hourly_stats()
    X, y, w = _xy(HS)
    groups = pd.Series(X.index.get_level_values('vessel_id'), index=X.index)
    clf = classifier.VesselTypeClassifier(random_state=SEED)
    t0 = time.time()
    if hasattr(clf, 'oof_vessel') or 'weight' in clf.validate.__code__.co_varnames:
        # 분류 담당 validate() 가 선박 단위 K-fold OOF 를 계산해 clf.oof_vessel 로 남긴다 (classifier_report.json 과 같은 값)
        clf.validate(X, y, groups, None, n_splits=n_splits, weight=w)
        Pv = clf.oof_vessel.reindex(columns=C.CLASSES, fill_value=0.0)
        Pv.index = Pv.index.astype(str)
        Pv.index.name = C.COL_VESSEL
        print(f'  [분류 OOF] classifier.validate() 사용 ({time.time() - t0:.0f}초)', flush=True)
    else:
        # 예전 classifier.py (validate 가 단일 분할): 같은 모델 클래스로 K-fold 를 여기서 돌린다
        from sklearn.model_selection import StratifiedGroupKFold
        oof = pd.DataFrame(np.nan, index=X.index, columns=C.CLASSES)
        for k, (tr, va) in enumerate(StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED).split(X, y, groups)):
            m = classifier.VesselTypeClassifier(random_state=SEED).fit(X.iloc[tr], y.iloc[tr])
            oof.iloc[va] = _probs(m, X.iloc[va]).to_numpy()
            print(f'  [분류 OOF] fold {k + 1}/{n_splits} ({time.time() - t0:.0f}초)', flush=True)
        Pv = _to_vessel(oof, w)
    _MEM['oof'] = Pv
    return Pv


def fit_full():
    """학습 데이터 전체로 학습한 분류 모델 (테스트 선박 예측용)"""
    if 'clf' in _MEM:
        return _MEM['clf']
    classifier = cls_package()[0]
    X, y, _ = _xy(train_hourly_stats())
    _MEM['clf'] = classifier.VesselTypeClassifier(random_state=SEED).fit(X, y)
    return _MEM['clf']


def predict_vessels(raw_paths):
    """
    원본 CSV(업종 열 없어도 됨) → 선박별 업종 확률 + 예측 업종.
    반환: index = rfid_id(str), columns = C.CLASSES + ['pred_type', 'pred_prob']
    """
    clf = fit_full()
    HS = hourly_stats(raw_paths, label=False)
    X, _, w = _xy(HS)
    if hasattr(clf, 'predict_vessels'):                       # 분류 담당 함수: 선박 × 월 → 선박
        Pv = clf.predict_vessels(X, w)
        Pv = Pv[[f'prob_{c}' for c in clf.classes_]].rename(columns=lambda c: c.replace('prob_', ''))
        Pv = Pv.reindex(columns=C.CLASSES, fill_value=0.0)
        Pv.index = Pv.index.astype(str)
        Pv.index.name = C.COL_VESSEL
    else:
        Pv = _to_vessel(_probs(clf, X), w)
    Pv['pred_type'] = Pv[C.CLASSES].idxmax(axis=1)
    Pv['pred_prob'] = Pv[C.CLASSES].max(axis=1)
    return Pv


# ---------------------------------------------------------------------
# 단독 실행: 연결 점검 + 선박 단위 교차검증 성능
# ---------------------------------------------------------------------
if __name__ == '__main__':
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    P.show()
    Pv = train_vessel_oof()
    yv = train_hourly_stats().groupby('vessel_id')['reg_type'].agg(lambda s: s.mode().iloc[0])
    yv.index = yv.index.astype(str)
    y = yv.reindex(Pv.index)
    p = Pv.idxmax(axis=1)
    print(f'\n선박 {len(Pv):,}척 | 선박 단위 교차검증 정확도(= micro-F1) {accuracy_score(y, p):.3f} '
          f'| macro-F1 {f1_score(y, p, average="macro"):.3f}')
    print(pd.DataFrame(confusion_matrix(y, p, labels=C.CLASSES), index=[f'실제_{c}' for c in C.CLASSES],
                       columns=[f'예측_{c}' for c in C.CLASSES]).to_string())
