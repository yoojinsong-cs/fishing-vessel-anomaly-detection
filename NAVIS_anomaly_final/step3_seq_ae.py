# -*- coding: utf-8 -*-
"""
3단계. 시퀀스 오토인코더 → 두 번째 이상점수(재구성 오차)

입력 시퀀스 (행 = 선박×1시간, 점 순서 그대로)
  스텝마다 5채널: [log(1+점간 거리), sin(회전각), cos(회전각), 정지(거리 0) 여부, 조각 경계 여부]
  - 절대 좌표는 넣지 않는다 (위치를 외우지 않도록).
  - 점이 SEQ_LEN+1개보다 많으면 균등 간격으로 줄이고, 적으면 0으로 채우고 마스크로 가린다.
  - 보고 빈도가 상태에 따라 달라지는 것 자체가 '정상 패턴'이므로 시간 재샘플링은 하지 않는다.
모델: 1D 합성곱 오토인코더 (64스텝 → 16스텝×16채널로 압축 → 복원)
점수: 실제 스텝만 대상으로 한 평균 제곱 오차 (ae_raw, 클수록 이상)

※ navis_features.py(특징 추출 코드)가 이 파일과 같은 폴더에 있어야 한다.

출력 (OUT_DIR)
  scores_ae.parquet   행별 ae_raw
  model_ae.pt         모델 가중치 + 정규화 값 (6단계에서 재사용)
"""
import os
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from pyproj import Transformer

import anomaly_config as C
import anomaly_common as U

try:
    import navis_features as NF
except ImportError as e:
    raise ImportError('navis_features.py 를 이 스크립트와 같은 폴더에 두세요 (특징 추출 코드).') from e

N_CH = 5


# ---------------------------------------------------------------------
# 시퀀스 만들기
# ---------------------------------------------------------------------
def seq_from_parts(parts, T=C.SEQ_LEN):
    """미터 좌표 조각 리스트 → (T, 5) 배열, 실제 스텝 수. 점이 부족하면 None."""
    pieces = []
    for p in parts:
        pieces += NF.clean_part(p, NF.CFG['JUMP_M'], NF.CFG['MAX_STEP_M'])[0]
    pieces = [q for q in pieces if len(q) > 0]
    if not pieces:
        return None, 0
    pts = np.vstack(pieces)
    piece_id = np.concatenate([np.full(len(q), k) for k, q in enumerate(pieces)])
    if len(pts) < C.SEQ_MIN_POINTS:
        return None, 0
    if len(pts) > T + 1:
        idx = np.round(np.linspace(0, len(pts) - 1, T + 1)).astype(int)
        pts, piece_id = pts[idx], piece_id[idx]
    v = np.diff(pts, axis=0)
    d = np.hypot(v[:, 0], v[:, 1])
    h = np.arctan2(v[:, 1], v[:, 0])
    turn = np.r_[0.0, (np.diff(h) + np.pi) % (2 * np.pi) - np.pi]
    zero = d <= NF.CFG['ZERO_EPS_M']
    turn[zero | np.r_[False, zero[:-1]]] = 0.0          # 정지 중의 방향은 의미 없음
    boundary = (piece_id[1:] != piece_id[:-1]).astype('float32')
    n = len(d)
    x = np.zeros((T, N_CH), dtype='float32')
    x[:n, 0] = np.log1p(d)
    x[:n, 1] = np.sin(turn)
    x[:n, 2] = np.cos(turn)
    x[:n, 3] = zero
    x[:n, 4] = boundary
    return x, n


def build_batch(parts_batch):
    X = np.zeros((len(parts_batch), C.SEQ_LEN, N_CH), dtype='float32')
    L = np.zeros(len(parts_batch), dtype='int32')
    for i, p in enumerate(parts_batch):
        x, n = seq_from_parts(p)
        if x is not None:
            X[i], L[i] = x, n
    return X, L


def build(parts, par, B=2000):
    batches = [parts[i:i + B] for i in range(0, len(parts), B)]
    res = par(delayed(build_batch)(b) for b in batches)
    if not res:
        return np.zeros((0, C.SEQ_LEN, N_CH), 'float32'), np.zeros(0, 'int32')
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


def iter_raw(transformer, swap, frac=None, rng=None):
    """원본 CSV를 청크로 읽어 (키, 미터 좌표 조각 리스트) 반환. row_in_file 은 특징 파일과 같은 방식으로 셈."""
    for path in U.raw_files():
        enc = NF.detect_encoding(path)
        row0 = 0
        for ch in pd.read_csv(path, chunksize=C.CSV_CHUNK_ROWS, encoding=enc, dtype=str, low_memory=False):
            rows = np.arange(row0, row0 + len(ch))
            row0 += len(ch)
            if frac is not None:
                pick = rng.random(len(ch)) < frac
                ch, rows = ch[pick], rows[pick]
            ok = ch[C.COL_LABEL].isin(C.CLASSES).values
            ch, rows = ch[ok], rows[ok]
            parts = NF.records_to_parts(NF.parse_geoms(ch[C.COL_GEOM]), transformer, swap)
            yield os.path.basename(path), rows, parts


# ---------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------
class ConvAE(nn.Module):
    def __init__(self, c=N_CH):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(c, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 16, 5, stride=2, padding=2), nn.ReLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(32, c, 5, padding=2))

    def forward(self, x):            # x: (B, C, T)
        return self.dec(self.enc(x))


def to_tensor(X, L, norm):
    X = X.copy()
    mask = (np.arange(C.SEQ_LEN)[None, :] < L[:, None]).astype('float32')
    X[:, :, 0] = np.where(mask > 0, (X[:, :, 0] - norm['mu']) / norm['sd'], 0)
    return torch.from_numpy(X.transpose(0, 2, 1).copy()), torch.from_numpy(mask)


CH_GROUPS = {'ae_err_dist': [0], 'ae_err_turn': [1, 2], 'ae_err_stop': [3]}   # v2.1: 채널별 오차 (사유 표시용)


def row_errors(model, X, L, norm, batch=4096, channels=False):
    """행별 재구성 오차 (실제 스텝 평균). 점이 부족한 행은 NaN. channels=True면 채널별 오차도 반환."""
    err = np.full(len(X), np.nan, dtype='float32')
    ch = {k: np.full(len(X), np.nan, dtype='float32') for k in CH_GROUPS}
    ok = np.flatnonzero(L > 0)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(ok), batch):
            ii = ok[s:s + batch]
            xb, mb = to_tensor(X[ii], L[ii], norm)
            sq = (model(xb) - xb) ** 2                     # (B, C, T)
            den = mb.sum(1)
            err[ii] = ((sq.mean(1) * mb).sum(1) / den).numpy()
            for k, cs in CH_GROUPS.items():
                ch[k][ii] = ((sq[:, cs, :].mean(1) * mb).sum(1) / den).numpy()
    return (err, ch) if channels else err


def load_model():
    ck = torch.load(U.out('model_ae.pt'), map_location='cpu', weights_only=False)
    m = ConvAE(); m.load_state_dict(ck['state']); m.eval()
    return m, ck['norm']


def feature_transformer():
    with open(C.FEAT_THRESH, encoding='utf-8') as fp:
        thr = json.load(fp)
    tr = Transformer.from_crs(NF.CFG['SRC_CRS'], NF.CFG['DST_CRS'], always_xy=True)
    return tr, bool(thr.get('swap', False))


# ---------------------------------------------------------------------
def main(score_only=None):
    """score_only=True 이면 저장된 model_ae.pt로 채점만 한다 (v2.1: 채널별 오차를 얻기 위해 재학습 없이 실행)"""
    import sys
    if score_only is None:
        score_only = '--score-only' in sys.argv
    t0 = time.time()
    torch.set_num_threads(max(1, C.N_JOBS))
    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    transformer, swap = feature_transformer()

    with Parallel(n_jobs=C.N_JOBS, backend='loky') as par:
        if score_only:
            model, norm = load_model()
            print('[1~2] 저장된 오토인코더를 불러옵니다 (재학습 없음)')
        else:
            # ----- 학습 표본 -----
            print('[1] 학습용 시퀀스 표본 만들기')
            Xs, Ls = [], []
            for name, rows, parts in iter_raw(transformer, swap, frac=C.AE_TRAIN_FRAC, rng=rng):
                X, L = build(parts, par)
                Xs.append(X[L > 0]); Ls.append(L[L > 0])
                if sum(len(x) for x in Xs) >= C.AE_TRAIN_MAX:
                    break
            X = np.concatenate(Xs)[:C.AE_TRAIN_MAX]; L = np.concatenate(Ls)[:C.AE_TRAIN_MAX]
            valid = np.arange(C.SEQ_LEN)[None, :] < L[:, None]
            norm = dict(mu=float(X[:, :, 0][valid].mean()), sd=float(X[:, :, 0][valid].std() + 1e-6))
            print(f'  표본 {len(X):,}개, log 거리 평균 {norm["mu"]:.2f} 표준편차 {norm["sd"]:.2f}')

            # ----- 학습 -----
            print('[2] 오토인코더 학습')
            model = ConvAE()
            opt = torch.optim.Adam(model.parameters(), lr=C.AE_LR)
            xt, mt = to_tensor(X, L, norm)
            n_val = max(1, min(len(X) // 10, 20_000))   # 10%를 검증용으로 (최대 2만 개)
            perm = torch.randperm(len(X))
            val, trn = perm[:n_val], perm[n_val:]

            def masked_loss(xb, mb):
                se = ((model(xb) - xb) ** 2).mean(1)
                return (se * mb).sum() / mb.sum()

            for ep in range(C.AE_EPOCHS):
                model.train()
                order = trn[torch.randperm(len(trn))]
                tot = 0.0
                for s in range(0, len(order), C.AE_BATCH):
                    b = order[s:s + C.AE_BATCH]
                    loss = masked_loss(xt[b], mt[b])
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item() * len(b)
                model.eval()
                with torch.no_grad():
                    vl = masked_loss(xt[val], mt[val]).item()
                print(f'  epoch {ep + 1}/{C.AE_EPOCHS}  학습 손실 {tot / len(trn):.4f}  검증 손실 {vl:.4f}', flush=True)
            torch.save(dict(state=model.state_dict(), norm=norm, seq_len=C.SEQ_LEN), U.out('model_ae.pt'))
            del X, L, xt, mt

        # ----- 전체 점수 -----
        print('[3] 전체 행 재구성 오차 계산')
        outs = []
        for name, rows, parts in iter_raw(transformer, swap):
            X, L = build(parts, par)
            err, ch = row_errors(model, X, L, norm, channels=True)
            outs.append(pd.DataFrame({'src_file': name, 'row_in_file': rows.astype('int64'),
                                      'ae_raw': err, **ch, 'seq_steps': L}))
            print(f'  {name}: 누적 {sum(len(o) for o in outs):,}행 ({time.time() - t0:.0f}초)', flush=True)
    res = pd.concat(outs, ignore_index=True)
    res.to_parquet(U.out('scores_ae.parquet'), index=False)
    print(f'  점수 없는 행(점 {C.SEQ_MIN_POINTS}개 미만): {res.ae_raw.isna().mean():.1%}')
    print(f'완료 ({time.time() - t0:.0f}초)')


if __name__ == '__main__':
    main()
