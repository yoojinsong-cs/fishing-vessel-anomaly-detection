# -*- coding: utf-8 -*-
"""
6단계 (v2.4). 여러 시간에 걸친 인위적 이상으로 검증 → 임계값

v2.4 변경
  - 인위적 이상은 그 선박 모항에서 SYNTH_SEA_KM 밖(바다 위)에만 심는다. v2.3에서는 28%가 항구 안이었다.
  - 사례마다 모항 거리 기준별 '바다 위 장시간 정지 길이'(stop_km*)도 저장한다 (7단계 정지 규칙 선택용).
  - 점수 사건 조합(sh_*)은 두 규칙을 끈 순수 점수 기준이다.

v2.2 변경
  - 사건 판정이 7단계와 같은 새 규칙(빈틈 허용, 평균 점수 기준, 바다 위 신호 단절 규칙 경보)을 쓴다.

v2.3 변경
  - 급정지를 3시간(sudden_stop)과 6시간(sudden_stop_6h)으로 나눠 검증한다.
    3시간 정지는 정상 조업에서도 흔하고, 기관손상 선박은 구조까지 더 오래 멈춰 있다 (4차 보고서 3.3).
  - 사례마다 '점수 사건 조합별 탐지 여부'(sh_*)와 '모항 거리 기준별 끊긴 시간'(loss_km*)을 저장한다.
    7단계가 이것과 실제 데이터의 하루 건수로 임계값·단절 규칙 기준을 자동으로 고른다.

v1과 달라진 점
  - 인위적 이상을 '한 시간'이 아니라 실제 선박의 항해 한가운데에 '여러 시간'으로 심는다.
    그 선박의 앞뒤 실제 레코드와 함께 맥락 특징·선박별 기준·상태 희소성을 다시 계산하므로,
    개선 2~5가 실제로 효과가 있는지 잴 수 있다.
  - 행 단위 탐지율과 함께 '이상 사건'(7단계와 같은 규칙) 단위 탐지율을 보고한다.

인위적 이상 (SYNTH_PATH 가 없을 때 자체 생성)
  drift        표류: 항해 중 한 시점부터 SYNTH_HOURS 시간 동안 0.3~1.0 m/s로 정지 없이 떠밀림
  signal_loss  신호 단절: 움직이던 시간 앞부분 20~40%만 남고, 이후 12~48시간 레코드 없음
  sudden_stop  급정지: 움직이다가 시간 중간에 멈춘 뒤(1시간) 바다 위 한 지점에 3시간 정지
  sudden_stop_6h  같은 급정지 뒤 6시간 정지 (기관손상 후 구조를 기다리는 상황)

출력 (OUT_DIR)
  validation_cases.csv       사례별 결과 (최고 점수, 행·사건 탐지 여부, 원래 구간의 사건 여부, 조합별 탐지 여부)
  validation_summary.csv     유형별 탐지율(행 주의·경보, 사건, 경보 등급 사건), AUROC, 원래 구간 사건 비율
  anomaly_thresholds.json    임계값과 설정
  validation_*.png           그래프
"""
import os
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import anomaly_config as C
import anomaly_common as U
import step3_seq_ae as AE

NF = AE.NF


# ---------------------------------------------------------------------
# 인위적 궤적 (미터 좌표)
# ---------------------------------------------------------------------
def clean_points(parts):
    pcs = []
    for p in parts:
        pcs += NF.clean_part(p, NF.CFG['JUMP_M'], NF.CFG['MAX_STEP_M'])[0]
    return np.vstack(pcs) if pcs else np.zeros((0, 2))


def drift_hours(start, n_list, rng):
    """여러 시간 이어지는 표류 궤적: 시간마다 조각 하나"""
    speed = rng.uniform(0.3, 1.0)
    h0 = rng.uniform(0, 2 * np.pi)
    turn = np.deg2rad(rng.uniform(-20, 20))
    total = 3600 * len(n_list)
    out, pos = [], np.asarray(start, dtype=float)
    for j, n in enumerate(n_list):
        ts = np.sort(rng.uniform(j * 3600, (j + 1) * 3600, n))
        prev_t = j * 3600
        pts = []
        for tt in ts:
            h = h0 + turn * tt / total
            step = speed * (tt - prev_t)
            pos = pos + step * np.array([np.cos(h), np.sin(h)])
            prev_t = tt
            pts.append(pos.copy())
        out.append([np.array(pts) + rng.normal(0, 3, (n, 2))])
    return out


def stop_jitter(center, n, rng):
    return [np.asarray(center) + rng.normal(0, 3, (n, 2))]


def read_rows(keys, transformer, swap):
    want = keys.groupby('src_file')['row_in_file'].apply(set).to_dict()
    found = {}
    for path in U.raw_files():
        name = os.path.basename(path)
        if name not in want:
            continue
        enc = NF.detect_encoding(path)
        row0 = 0
        for ch in pd.read_csv(path, chunksize=C.CSV_CHUNK_ROWS, encoding=enc, dtype=str, low_memory=False):
            rows = np.arange(row0, row0 + len(ch)); row0 += len(ch)
            m = np.isin(rows, list(want[name]))
            if m.any():
                parts = NF.records_to_parts(NF.parse_geoms(ch.loc[m, C.COL_GEOM]), transformer, swap)
                for r, p in zip(rows[m], parts):
                    found[(name, int(r))] = p
    return found


# ---------------------------------------------------------------------
# 한 선박 구간 채점 (실제 파이프라인과 같은 순서)
# ---------------------------------------------------------------------
class Scorer:
    def __init__(self, ref):
        self.refc = U.make_ref(ref)
        self.base = pd.read_parquet(U.out('normal_baseline.parquet'))
        bundle = joblib.load(U.out('model_if.joblib'))
        self.models, self.state_model = bundle['models'], bundle['state_model']
        self.ports = bundle.get('ports')                                  # [v2.1 D]
        aeb_path = U.out('model_ae_baseline.joblib')
        self.aeb = joblib.load(aeb_path) if os.path.exists(aeb_path) else None   # [v2.1 C]
        self.ae = AE.load_model() if os.path.exists(U.out('model_ae.pt')) else (None, None)
        with open(C.FEAT_THRESH, encoding='utf-8') as fp:
            self.fthr = json.load(fp)
        self.names = NF.feature_names(NF.CFG)
        self.ae_ref = ref.set_index(U.KEYS)['ae_raw'] if 'ae_raw' in ref else None
        self.mm = ref.groupby('vessel')['mismatch'].first()

    def features(self, parts):
        return NF.features_batch([parts], self.fthr, NF.CFG, self.names)[0]

    def score(self, fv, syn_parts):
        """fv: 한 선박 구간의 특징 행 (syn 컬럼 = 인위적 행 번호, -1은 실제). syn_parts: {번호: 조각}"""
        fv = fv.copy()
        fv['state'] = np.where(fv['L_total'].notna(), 'active', 'stationary')
        fv = U.add_time_context(fv)
        fv, _ = U.add_port_context(fv, self.ports if self.ports is not None else np.zeros((0, 2)))
        fv['grp'] = U.group_key(fv)
        fv['state_rarity'] = U.state_rarity(fv, self.state_model)
        fv['if_raw'] = np.nan
        fv['reason1'] = ''
        for state, m in self.models.items():
            ii = np.flatnonzero(fv['state'].values == state)
            if len(ii) == 0:
                continue
            sub = fv.iloc[ii]
            Z, _, _, feats = U.zscores(sub, self.base, state)
            if m['vessel_baseline'] is not None:
                Z = U.apply_vessel_baseline(sub['vessel'].values, Z, m['vessel_baseline'], feats)
            fv.loc[fv.index[ii], 'if_raw'] = -m['model'].score_samples(Z)
            fv.loc[fv.index[ii], 'reason1'] = U.top_reasons(Z, feats, k=1)['reason1']
        # AE: 인위적 행은 새로 계산, 실제 행은 3단계 점수 재사용
        if self.ae_ref is not None:
            fv['ae_raw'] = self.ae_ref.reindex(pd.MultiIndex.from_frame(fv[U.KEYS])).values.astype('float64')
            syn = np.flatnonzero(fv['syn'].values >= 0)
            if len(syn) and self.ae[0] is not None:
                X, L = AE.build_batch([syn_parts[k] for k in fv['syn'].values[syn]])
                fv.loc[fv.index[syn], 'ae_raw'] = AE.row_errors(self.ae[0], X, L, self.ae[1])
            if self.aeb is not None:
                fv['ae_adj'] = U.apply_ae_baseline(fv, self.aeb)
        for k, v in U.combine(fv, self.refc).items():
            fv[k] = v
        fv['anomaly_score'] = U.global_percentile(fv['combined'].values, self.refc['combined'])
        mm = self.mm.get(fv['vessel'].iat[0], 0.0)
        fv['mismatch'] = mm
        fv['final_score'] = fv['anomaly_score'] + C.W_MISMATCH * mm * (1 - fv['anomaly_score'])
        return fv


def build_cases(ref, rng):
    """항해 한가운데의 활동 시간을 기준점으로 고른다"""
    r = ref[['vessel', 'trip', 't', 'state', 'L_total'] + U.KEYS +
            (['dist_port_km'] if 'dist_port_km' in ref.columns else [])].copy()
    r['pos'] = r.groupby('trip').cumcount()
    r['tlen'] = r.groupby('trip')['trip'].transform('size')
    hmax = max(C.SYNTH_HOURS_BY_TYPE.values())
    cand = r[(r['state'] == 'active') & r['L_total'].gt(500) & (r['pos'] >= 1)
             & (r['pos'] + hmax <= r['tlen'])]
    if 'dist_port_km' in cand.columns:                     # [v2.4-2] 바다 위에만 심는다
        before = len(cand)
        # 그 시간의 끝 지점과 시작 지점(= 직전 시간의 끝 지점)이 모두 모항에서 멀어야 한다.
        # 신호 단절은 그 시간 앞부분만 남기므로 끊기는 곳이 시작 지점 근처가 된다.
        prev_d = r.groupby('trip')['dist_port_km'].shift(1).reindex(cand.index)
        cand = cand[(cand['dist_port_km'] > C.SYNTH_SEA_KM) & (prev_d > C.SYNTH_SEA_KM)]
        print(f'  바다 위(모항 {C.SYNTH_SEA_KM:g}km 밖) 후보 {len(cand):,} / {before:,}')
    cand = cand.sample(frac=1, random_state=C.SEED).drop_duplicates('trip')
    kinds = list(C.SYNTH_HOURS_BY_TYPE)                     # [v2.3-5] drift, signal_loss, sudden_stop(3h), sudden_stop_6h
    n = min(len(cand) // len(kinds), C.SYNTH_N_PER_TYPE)
    cand = cand.iloc[:n * len(kinds)].copy()
    cand['anomaly_type'] = np.repeat(kinds, n)
    return cand


def main():
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)
    ref = pd.read_parquet(U.out('scores_final.parquet')).sort_values(['vessel', 't']).reset_index(drop=True)
    t1, t2 = float(ref['final_score'].quantile(C.CAUTION_Q)), float(ref['final_score'].quantile(C.ALARM_Q))
    tm = float(ref['final_score'].quantile(C.EVENT_MEAN_Q))                 # [v2.2-4] 묶음 평균 기준
    habit = U.vessel_loss_habit(ref)                                        # [v2.2-2] 상습 단절 선박
    shabit = U.vessel_stop_habit(ref)                                       # [v2.4-3] 상습 정지 선박
    tune_q = {q: float(ref['final_score'].quantile(q)) for q in C.TUNE_ALARM_Q}
    ref_ev, ref_ev_id = U.detect_events(ref, t1, t2, t_mean=tm, habit=habit, stop_habit=shabit)
    ref['event_id'] = ref_ev_id
    sc = Scorer(ref)
    transformer, swap = AE.feature_transformer()
    if C.SYNTH_PATH:
        print('  [안내] SYNTH_PATH 사용 시에는 파일의 행들만으로 맥락을 계산합니다.')

    print('[1] 인위적 이상 사례 고르기')
    cases = build_cases(ref, rng)
    print(f'  {cases.anomaly_type.value_counts().to_dict()}')
    vessels = set(cases['vessel'])
    print('[2] 해당 선박들의 특징과 원본 궤적 읽기')
    feat = U.load_features(vessels=vessels)
    geo = read_rows(cases[U.KEYS], transformer, swap)
    raw_cols = U.raw_feature_names()
    by_vessel = {v: g.sort_values('t') for v, g in feat.groupby('vessel')}

    print('[3] 사례별로 심고 다시 채점', flush=True)
    results = []
    for ci, c in enumerate(cases.itertuples(index=False)):
        p = geo.get((c.src_file, int(c.row_in_file)))
        if p is None:
            continue
        pts = clean_points(p)
        if len(pts) < 10:
            continue
        gv = by_vessel.get(c.vessel)
        if gv is None:
            continue
        fv = gv[(gv['t'] >= c.t - 48) & (gv['t'] <= c.t + 96)].copy()
        fv['syn'] = -1
        k = int(np.flatnonzero(fv['t'].values == c.t)[0]) if (fv['t'].values == c.t).any() else None
        if k is None:
            continue
        idx = fv.index
        syn_parts = {}
        H = C.SYNTH_HOURS_BY_TYPE.get(c.anomaly_type, C.SYNTH_HOURS)
        if c.anomaly_type == 'drift':
            hrs = [i for i in range(k, min(k + H, len(fv))) if fv['t'].iat[i] == c.t + (i - k)]
            n_list = [int(max(10, min(90, fv['n_points_raw'].iat[i] if np.isfinite(fv['n_points_raw'].iat[i]) else 20))) for i in hrs]
            for j, (i, prt) in enumerate(zip(hrs, drift_hours(pts[0], n_list, rng))):
                syn_parts[j] = prt
        elif c.anomaly_type.startswith('sudden_stop'):
            hrs = [i for i in range(k, min(k + H, len(fv))) if fv['t'].iat[i] == c.t + (i - k)]
            h = max(2, len(pts) // 2)
            first = [np.vstack([pts[:h], pts[h - 1] + rng.normal(0, 3, (len(pts) - h, 2))])]
            syn_parts[0] = first
            for j in range(1, len(hrs)):
                syn_parts[j] = stop_jitter(pts[h - 1], 8, rng)
        else:  # signal_loss
            hrs = [k]
            syn_parts[0] = [pts[:max(2, int(len(pts) * rng.uniform(0.2, 0.4)))]]
            cut = syn_parts[0][0][-1]                       # [v2.4-2] 끊기는 지점 자체가 바다 위인 사례만
            if sc.ports is not None and len(sc.ports):
                dcut = U.port_distance_km(np.array([c.vessel]), np.array([cut[0]]), np.array([cut[1]]), sc.ports)[0]
                if np.isfinite(dcut) and dcut <= C.SYNTH_SEA_KM:
                    continue
            lost = rng.uniform(12, 48)
            fv = fv[~((fv['t'] > c.t) & (fv['t'] <= c.t + lost))]
            # 잘라낸 뒤에도 기준점 위치는 그대로
        for j, i in enumerate(hrs):
            row = idx[i]
            allp = np.vstack([q for q in syn_parts[j] if len(q)])            # [v2.1 D] 인위적 궤적의 시작·끝 좌표
            fv.loc[row, ['xs', 'ys', 'xe', 'ye']] = np.float32([allp[0, 0], allp[0, 1], allp[-1, 0], allp[-1, 1]])
            f = sc.features(syn_parts[j])
            for col in raw_cols:
                fv.loc[row, col] = np.float32(f.get(col, np.nan))
            fv.loc[row, 'syn'] = j
        fv = sc.score(fv, syn_parts)
        syn_rows = fv['syn'].values >= 0
        ev, ev_id = U.detect_events(fv, t1, t2, t_mean=tm, habit=habit, stop_habit=shabit)
        hit_ids = np.unique(ev_id[syn_rows][ev_id[syn_rows] >= 0])
        hit_alarm = bool(len(hit_ids)) and bool((ev.set_index('event_id').loc[hit_ids, 'level'] == '경보').any())
        hit_kind = ev.set_index('event_id').loc[hit_ids, 'kind'].iat[0] if len(hit_ids) else ''
        tune_hit = {}
        for q, tq in tune_q.items():          # [v2.3-6] 점수 사건 조합별 탐지 여부 (단절 규칙은 따로)
            for ph in C.TUNE_PERSIST_H:
                for bh in C.TUNE_BRIDGE_H:
                    _, id2 = U.detect_events(fv, t1, tq, persist_h=ph, bridge_h=bh, t_mean=tm,
                                             loss_rule=False, stop_rule=False)
                    tune_hit[f'sh_{q}_{ph}_{bh}'] = bool((id2[syn_rows] >= 0).any())
        for km in C.LOSS_DIAG_KM:             # [v2.3-2] 모항 거리 기준별 '바다 위에서 끊긴 시간' → 7단계에서 시간 기준과 비교
            tune_hit[f'loss_km{km}'] = float(U.loss_hours(fv, km)[syn_rows].max())
        for km in C.STOP_DIAG_KM:             # [v2.4-3] 모항 거리 기준별 '바다 위 장시간 정지 길이'
            tune_hit[f'stop_km{km}'] = float(U.stop_runs(fv, km)[0][syn_rows].max())
        orig = ref[(ref['vessel'] == c.vessel) & ref['t'].isin(fv.loc[syn_rows, 't'])]
        results.append(dict(anomaly_type=c.anomaly_type, vessel=c.vessel, code=fv['code'].iat[0], t=c.t,
                            hours=int(syn_rows.sum()),
                            max_score=float(fv.loc[syn_rows, 'final_score'].max()),
                            max_u_if=float(fv.loc[syn_rows, 'u_if'].max()) if 'u_if' in fv else np.nan,
                            max_u_ae=float(fv.loc[syn_rows, 'u_ae'].max()) if 'u_ae' in fv else np.nan,
                            max_u_state=float(fv.loc[syn_rows, 'u_state'].max()) if 'u_state' in fv else np.nan,
                            reason=fv.loc[syn_rows, 'reason1'].replace('', np.nan).dropna().iloc[0] if fv.loc[syn_rows, 'reason1'].replace('', np.nan).notna().any() else '',
                            row_caution=bool((fv.loc[syn_rows, 'final_score'] >= t1).any()),
                            row_alarm=bool((fv.loc[syn_rows, 'final_score'] >= t2).any()),
                            event=bool((ev_id[syn_rows] >= 0).any()), event_alarm=hit_alarm, event_kind=hit_kind,
                            dist_port_km=float(fv.loc[syn_rows, 'dist_port_km'].iloc[-1]) if 'dist_port_km' in fv else np.nan,
                            loss_h=float(U.loss_hours(fv)[syn_rows].max()),
                            base_max_score=float(orig['final_score'].max()) if len(orig) else np.nan,
                            base_event=bool((orig['event_id'] >= 0).any()) if len(orig) else False, **tune_hit))
        if (ci + 1) % 100 == 0:
            print(f'  {ci + 1}/{len(cases)} ({time.time() - t0:.0f}초)', flush=True)
    R = pd.DataFrame(results)
    R.to_csv(U.out('validation_cases.csv'), index=False, encoding='utf-8-sig')

    print('[4] 유형별 결과')
    normal = ref['final_score'].sample(n=min(len(ref), 50_000), random_state=C.SEED).values
    summ = []
    for kind, g in R.groupby('anomaly_type'):
        y = np.r_[np.zeros(len(normal)), np.ones(len(g))]
        summ.append(dict(anomaly_type=kind, n=len(g),
                         row_caution=g['row_caution'].mean(), row_alarm=g['row_alarm'].mean(),
                         event_recall=g['event'].mean(), alarm_event_recall=g['event_alarm'].mean(),
                         base_event_rate=g['base_event'].mean(),
                         auroc=roc_auc_score(y, np.r_[normal, g['max_score'].values]),
                         median_max_score=g['max_score'].median(), median_base_score=g['base_max_score'].median(),
                         median_u_if=g['max_u_if'].median(), median_u_ae=g['max_u_ae'].median(),
                         median_u_state=g['max_u_state'].median(),
                         top_reason=g['reason'].replace('', np.nan).mode().iat[0] if g['reason'].replace('', np.nan).notna().any() else ''))
    summ = pd.DataFrame(summ)
    vhours = len(ref)
    bg = dict(events=int(len(ref_ev)), vessels_with_event=int(ref_ev['vessel'].nunique()) if len(ref_ev) else 0,
              vessel_share=float(ref_ev['vessel'].nunique() / ref['vessel'].nunique()) if len(ref_ev) else 0.0,
              events_per_1000h=float(len(ref_ev) / vhours * 1000))
    print(f'  임계값: 주의 ≥ {t1:.4f}, 경보 ≥ {t2:.4f}')
    print(f'  실제 데이터 이상 사건: {bg["events"]:,}건, 선박 {bg["vessels_with_event"]:,}척({bg["vessel_share"]:.1%}), '
          f'관측 1,000시간당 {bg["events_per_1000h"]:.2f}건')
    print(summ.round(3).to_string(index=False))
    summ.to_csv(U.out('validation_summary.csv'), index=False, encoding='utf-8-sig')
    U.save_json(dict(caution=t1, alarm=t2, event_mean=tm, caution_q=C.CAUTION_Q, alarm_q=C.ALARM_Q,
                     event_mean_q=C.EVENT_MEAN_Q, persist_h=C.PERSIST_H, bridge_h=C.BRIDGE_H,
                     loss_rule=C.LOSS_RULE, loss_h=C.LOSS_H, loss_km=C.LOSS_KM, habit_loss_rate=C.HABIT_LOSS_RATE, background=bg, combine=C.COMBINE, w_state=C.W_STATE, state_tail=C.STATE_TAIL,
                     vessel_baseline=C.USE_VESSEL_BASELINE, w_mismatch=C.W_MISMATCH, mismatch_tau=C.MISMATCH_TAU,
                     synthetic=summ.to_dict('records')), U.out('anomaly_thresholds.json'))

    plt = U.setup_korean_font()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(summ))
    for j, (col, lab) in enumerate([('row_caution', '행: 주의 이상'), ('row_alarm', '행: 경보'),
                                    ('event_recall', '이상 사건'), ('base_event_rate', '원래 구간(조작 전) 사건')]):
        ax.bar(x + (j - 1.5) * 0.2, summ[col], 0.2, label=lab, color=None if j < 3 else '0.75')
    ax.set_xticks(x, summ['anomaly_type']); ax.set_ylim(0, 1); ax.set_ylabel('비율'); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(U.out('validation_recall.png'), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(ref['final_score'], bins=100, color='0.75', density=True, label='실제 레코드')
    for kind, g in R.groupby('anomaly_type'):
        ax.hist(g['max_score'], bins=30, histtype='step', lw=1.6, density=True, label=f'인위적 이상: {kind} (구간 최고점)')
    ax.axvline(t1, color='orange', ls='--', label='주의'); ax.axvline(t2, color='red', ls='--', label='경보')
    ax.set_xlabel('최종 이상점수'); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(U.out('validation_score_hist.png'), dpi=150); plt.close(fig)
    print(f'완료 ({time.time() - t0:.0f}초)')


if __name__ == '__main__':
    main()
