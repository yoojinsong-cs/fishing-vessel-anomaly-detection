# -*- coding: utf-8 -*-
"""
7단계 (v2.5). 임계값 자동 선택 → 이상 사건 → 최종 결과물

목표 (3차 보고서 5장): 사고로 이어질 수 있는 움직임을 놓치지 않되, 담당자가 매일 확인할 수 있는 양
                       (하루 '경보' ALARM_BUDGET 건 이하)으로, 왜 경보인지 사유를 붙여 낸다.

사건 두 종류 (anomaly_common.detect_events)
  (1) 지속 이상 (점수): 같은 항해에서 주의 이상 시간이 PERSIST_H 이상이고,
      경보 수준 시간이 있으면 '경보', 묶음 평균만 높으면 '주의'
      [v2.4] 바다 위(모항 SCORE_SEA_KM 밖) 시간만 쓴다
  (2) 바다 위 신호 단절 (규칙): 모항에서 LOSS_KM 넘게 떨어진 바다에서 움직이던 중 LOSS_H 시간 이상 끊김.
      상습 단절 선박은 '주의'
  (3) [v2.4] 바다 위 장시간 정지 (규칙): 움직이다가 모항 STOP_KM 밖에서 STOP_H 시간 이상 거의 안 움직임.
      상습 정지 선박은 '주의'

[v2.3-6] 임계값 자동 선택 (AUTO_SELECT)
  점수 조합(경보 분위수 × 최소 주의 시간 × 빈틈) × 단절 조합(끊긴 시간 × 거리) × 정지 조합(정지 시간 × 거리)마다
    - 실제 데이터의 하루 '경보' 건수 (점수 경보 + 상습 아닌 선박의 단절)
    - 인위적 이상의 유형별 탐지율 (6단계 validation_cases.csv)
  를 계산해, 하루 경보 ≤ ALARM_BUDGET 인 조합 중 가중 탐지율(RECALL_WEIGHTS)이 가장 높은 것을 고른다.
  같으면 경보가 적은 쪽. 예산 안에 드는 조합이 없으면 경보가 가장 적은 조합을 고르고 그렇게 표시한다.

출력 (OUT_DIR)
  loss_frequency.csv       바다 위 신호 단절의 하루 건수 (끊긴 시간 × 모항 거리)
  stop_frequency.csv       [v2.4] 바다 위 장시간 정지의 하루 건수 (정지 시간 × 모항 거리)
  selection_grid.csv       모든 조합의 하루 경보·주의 건수와 탐지율 (선택 근거)
  final_thresholds.json    선택된 임계값과 규칙 (기계용)
  임계값_기준.md           주의·경보 기준 설명서 (사람용, 최종 결과물 ③)
  alarm_events.csv         이상 사건 목록
  alarm_vessels.csv        경보 선박 목록 (최종 결과물 ②) — 관측 MIN_OBS_H 시간 이상 선박 먼저, 경보 사건 수 순
  alarm_rows.csv           사건에 속한 시간별 레코드 (시각화 담당용, xe·ye = EPSG:5179 미터)
  scores_deliverable.parquet  선박·시간별 이상점수와 등급 (최종 결과물 ①)
  completion_check.csv     완료 기준 대비 최종 값
"""
import os
import json
import time
import numpy as np
import pandas as pd

import anomaly_config as C
import anomaly_common as U


def top_vessel_share(ev, n_v):
    """사건이 많은 상위 1% 선박이 전체 사건에서 차지하는 비율"""
    if len(ev) == 0:
        return 0.0
    k = max(1, int(round(n_v * 0.01)))
    return float(ev['vessel'].value_counts().head(k).sum() / len(ev))


def score_grid(df, t1, tm, days):
    """점수 사건 조합별 하루 건수 (단절 규칙 제외)"""
    q = df['final_score']
    out = []
    for qq in C.TUNE_ALARM_Q:
        tq = float(q.quantile(qq))
        for ph in C.TUNE_PERSIST_H:
            for bh in C.TUNE_BRIDGE_H:
                e, _ = U.detect_events(df, t1, tq, persist_h=ph, bridge_h=bh, t_mean=tm,
                                       loss_rule=False, stop_rule=False)
                al = int((e['level'] == '경보').sum()) if len(e) else 0
                out.append(dict(alarm_q=qq, persist_h=ph, bridge_h=bh,
                                score_alarm_per_day=al / days, score_total_per_day=len(e) / days))
    return pd.DataFrame(out)


def selection_grid(sg, lf, sf, cases):
    """점수 조합 × 단절 조합 × 정지 조합 → 하루 경보·전체 건수, 유형별 탐지율, 가중 탐지율"""
    lf = lf.rename(columns={'min_gap_h': 'loss_h', 'min_dist_km': 'loss_km', 'per_day': 'loss_per_day',
                            'per_day_alarm': 'loss_alarm_per_day'})[['loss_h', 'loss_km', 'loss_per_day', 'loss_alarm_per_day']]
    sf = sf.rename(columns={'min_stop_h': 'stop_h', 'min_dist_km': 'stop_km', 'per_day': 'stop_per_day',
                            'per_day_alarm': 'stop_alarm_per_day'})[['stop_h', 'stop_km', 'stop_per_day', 'stop_alarm_per_day']]
    g = sg.merge(lf, how='cross').merge(sf, how='cross')
    g['alarm_per_day'] = g['score_alarm_per_day'] + g['loss_alarm_per_day'] + g['stop_alarm_per_day']
    g['total_per_day'] = g['score_total_per_day'] + g['loss_per_day'] + g['stop_per_day']
    types = list(C.RECALL_WEIGHTS)
    if cases is not None and len(cases):
        for t in types:
            sub = cases[cases['anomaly_type'] == t]
            if len(sub) == 0:
                g[f'recall_{t}'] = np.nan; continue
            n = len(sub)
            # 조합마다 사례별 탐지 여부 = 점수 사건 | 단절 규칙 | 정지 규칙
            SH = {c: sub[c].astype(bool).to_numpy() for c in sub.columns if c.startswith('sh_')}
            LK = {km: sub[f'loss_km{km}'].to_numpy() for km in C.LOSS_DIAG_KM if f'loss_km{km}' in sub}
            SK = {km: sub[f'stop_km{km}'].to_numpy() for km in C.STOP_DIAG_KM if f'stop_km{km}' in sub}
            rec = np.empty(len(g))
            for i, r in enumerate(g.itertuples(index=False)):
                hit = SH.get(f'sh_{r.alarm_q}_{r.persist_h}_{r.bridge_h}', np.zeros(n, bool)).copy()
                if r.loss_km in LK:
                    hit |= LK[r.loss_km] >= r.loss_h
                if r.stop_km in SK:
                    hit |= SK[r.stop_km] >= r.stop_h
                rec[i] = hit.mean()
            g[f'recall_{t}'] = rec
        w = pd.Series(C.RECALL_WEIGHTS)
        R = g[[f'recall_{t}' for t in types]].to_numpy()
        ok = np.isfinite(R)
        g['weighted_recall'] = (np.where(ok, R, 0) * w.values).sum(1) / np.maximum((ok * w.values).sum(1), 1e-9)
    else:
        g['weighted_recall'] = np.nan
    return g.round(4)


def choose(g):
    """하루 경보 ≤ ALARM_BUDGET 중 가중 탐지율 최대 (같으면 경보가 적은 쪽)"""
    if g['weighted_recall'].isna().all():
        m = (g['alarm_q'] == C.ALARM_Q) & (g['persist_h'] == C.PERSIST_H) & (g['bridge_h'] == C.BRIDGE_H) & \
            (g['loss_h'] == C.LOSS_H) & (g['loss_km'] == C.LOSS_KM) & (g['stop_h'] == C.STOP_H) & (g['stop_km'] == C.STOP_KM)
        return (g[m].iloc[0] if m.any() else g.sort_values('alarm_per_day').iloc[0]), '설정값 사용 (6단계 결과 없음)'
    feas = g[g['alarm_per_day'] <= C.ALARM_BUDGET]
    # [v2.5-4] 신호 단절과 6시간 정지를 둘 다 지키는 조합을 먼저 (한쪽을 버리는 조합 방지)
    pr = [c for c in ['recall_signal_loss', 'recall_sudden_stop_6h'] if c in g.columns]
    if pr and len(feas):
        both = feas[(feas[pr] >= C.MIN_PRIORITY_RECALL).all(axis=1)]
        if len(both):
            best = both.sort_values(['weighted_recall', 'alarm_per_day'], ascending=[False, True]).iloc[0]
            return best, (f'하루 경보 {C.ALARM_BUDGET}건 이하이면서 신호 단절·6시간 정지 탐지율이 모두 '
                          f'{C.MIN_PRIORITY_RECALL:.0%} 이상인 {len(both)}개 조합 중 가중 탐지율 최대')
    if len(feas) == 0:
        return g.sort_values(['alarm_per_day', 'weighted_recall'], ascending=[True, False]).iloc[0], \
               f'하루 경보 {C.ALARM_BUDGET}건 이하 조합이 없어 경보가 가장 적은 조합을 선택'
    best = feas.sort_values(['weighted_recall', 'alarm_per_day'], ascending=[False, True]).iloc[0]
    return best, f'하루 경보 {C.ALARM_BUDGET}건 이하 {len(feas)}개 조합 중 가중 탐지율 최대'


def write_standard(path, sel, why, t1, t2, tm, chk, counts):
    """임계값_기준.md — 사람이 읽는 주의·경보 기준 (최종 결과물 ③)"""
    rec = lambda t: f"{sel.get(f'recall_{t}', np.nan):.1%}" if pd.notna(sel.get(f'recall_{t}', np.nan)) else '-'
    lines = [
        '# NAVIS 이상탐지 주의·경보 기준 (v2.4)', '',
        '## 1. 이상점수', '',
        '- 선박·시간(1시간)마다 0~1 이상점수를 매긴다. 0.99 = 전체 레코드 중 가장 이상한 1%.',
        '- 점수 = max(Isolation Forest, 시퀀스 오토인코더)를 업종 × 시간대 × 상태(활동/정박) 그룹 안에서 백분위로 바꾼 값',
        '  + 이 시간대에 이 상태가 드문 정도(바다 위에서만) + 분류 모델의 업종 불일치.', '',
        '## 2. 시간 단위 기준', '',
        '| 수준 | 기준 | 뜻 |', '| --- | --- | --- |',
        f'| 주의 | 최종 점수 ≥ {t1:.4f} (상위 {1 - C.CAUTION_Q:.0%}) | 평소와 다른 시간 |',
        f'| 경보 | 최종 점수 ≥ {t2:.4f} (상위 {1 - sel.alarm_q:.1%}) | 매우 드문 시간 |', '',
        '## 3. 사건 기준 (담당자에게 알리는 단위)', '',
        '| 종류 | 조건 | 등급 |', '| --- | --- | --- |',
        f'| 지속 이상 | 모항 {C.SCORE_SEA_KM:g}km 밖 바다 위에서, 같은 항해에서 주의 시간 {int(sel.persist_h)}시간 이상'
        f'{f", 사이 빈틈 {int(sel.bridge_h)}시간 허용" if sel.bridge_h else ""} | 경보 수준 시간이 있으면 경보, '
        f'묶음 평균 ≥ {tm:.4f}만 충족하면 주의 |',
        f'| 바다 위 신호 단절 | 모항에서 {sel.loss_km:g}km 넘게 떨어진 곳에서 움직이던 중 {int(sel.loss_h)}시간 이상 기록 없음 '
        f'(점수와 무관) | 경보. 단 항해의 {C.HABIT_LOSS_RATE:.0%} 이상이 이렇게 끝나는 상습 단절 선박은 주의 |',
        f'| 바다 위 장시간 정지 | 움직이다가 모항에서 {sel.stop_km:g}km 넘게 떨어진 곳에서 {int(sel.stop_h)}시간 이상 '
        f'거의 움직이지 않음 (시간당 {C.IDLE_L_M}m 미만, 점수와 무관) | 경보. 단 항해의 {C.HABIT_STOP_RATE:.0%} 이상에서 '
        '이런 정지를 하는 상습 정지 선박은 주의 |', '',
        f'같은 선박의 경보는 한 주에 {C.CHRONIC_MAX_PER_WEEK}번까지 내고, 그 뒤는 \'주의(반복)\'로 낮춘다 '
        '(한 선박의 반복 경보가 다른 선박의 경보를 가리지 않게).', '',
        '모항 = 그 선박이 출항·귀항한 곳 또는 8시간 이상 정박을 반복한 곳 (선박마다 따로 추정).',
        f'항구 안(모항 {C.SCORE_SEA_KM:g}km 이내)의 점수 이상은 사건으로 만들지 않는다 (항구 안 회전·GPS 흔들림, 사고 예방 대상 아님).', '',
        '## 4. 선택 근거', '',
        f'- {why}.',
        f'- 하루 경보 {sel.alarm_per_day:.1f}건 (점수 {sel.score_alarm_per_day:.1f} + 단절 {sel.loss_alarm_per_day:.1f} '
        f'+ 정지 {sel.stop_alarm_per_day:.1f}, 겹침 포함), '
        f'주의 포함 전체 {sel.total_per_day:.1f}건.',
        f'- 인위적 이상 탐지율: 신호 단절 {rec("signal_loss")}, 6시간 정지 {rec("sudden_stop_6h")}, '
        f'3시간 정지 {rec("sudden_stop")}, 표류 {rec("drift")}.',
        '- 가중치(사고 예방 목적상 신호 단절·장시간 정지 우선): ' +
        ', '.join(f'{k} {v}' for k, v in C.RECALL_WEIGHTS.items()) + '.', '',
        '## 5. 완료 기준 점검', '',
        '| 기준 | 목표 | 값 | 판정 |', '| --- | --- | --- | --- |']
    for r in chk.itertuples(index=False):
        if pd.isna(r.value):
            v = ''
        elif '율' in r.criterion and 'AUROC' not in r.criterion:
            v = f'{r.value:.1%}'
        else:
            v = f'{r.value:g}'
        lines.append(f'| {r.criterion} | {r.target} | {v} | {r.ok} |')
    lines += ['', '## 6. 규모', '',
              f'- 기간 {counts["days"]:.0f}일, 선박 {counts["vessels"]:,}척, 레코드 {counts["rows"]:,}행.',
              f'- 사건 {counts["events"]:,}건 (경보 {counts["alarm_events"]:,}건), 경보 선박 {counts["alarm_vessels"]:,}척.', '']
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines))


def main():
    t0 = time.time()
    df = pd.read_parquet(U.out('scores_final.parquet')).sort_values(['vessel', 't']).reset_index(drop=True)
    q = df['final_score']
    t1, tm = float(q.quantile(C.CAUTION_Q)), float(q.quantile(C.EVENT_MEAN_Q))
    n_v = df['vessel'].nunique()
    days = max(1.0, float(df['t'].max() - df['t'].min() + 1) / 24)
    if 'dist_port_km' not in df.columns or df['dist_port_km'].isna().all():
        print('  [주의] 모항 거리가 없습니다 (step0_endpoints.py → step1·2 실행 필요). 신호 단절 규칙이 작동하지 않습니다.')
    else:
        print(f'  모항 거리를 아는 행 {df["dist_port_km"].notna().mean():.1%}')

    print('[1] 바다 위 신호 단절 빈도')
    lf = U.loss_frequency(df, days)
    lf.to_csv(U.out('loss_frequency.csv'), index=False, encoding='utf-8-sig')
    print(lf.to_string(index=False))
    print('[1-2] 바다 위 장시간 정지 빈도')
    sf = U.stop_frequency(df, days)
    sf.to_csv(U.out('stop_frequency.csv'), index=False, encoding='utf-8-sig')
    print(sf.to_string(index=False))

    print('[2] 점수 조합 × 단절 규칙 조합 → 선택')
    sg = score_grid(df, t1, tm, days)
    cpath = U.out('validation_cases.csv')
    cases = pd.read_csv(cpath) if os.path.exists(cpath) else None
    if cases is not None and not any(c.startswith('sh_') for c in cases.columns):
        print('  [주의] validation_cases.csv 가 v2.3 형식이 아닙니다. 6단계를 다시 실행하세요.'); cases = None
    g = selection_grid(sg, lf, sf, cases)
    g.sort_values(['weighted_recall', 'alarm_per_day'], ascending=[False, True]) \
     .to_csv(U.out('selection_grid.csv'), index=False, encoding='utf-8-sig')
    if C.AUTO_SELECT:
        sel, why = choose(g)
    else:
        m = (g['alarm_q'] == C.ALARM_Q) & (g['persist_h'] == C.PERSIST_H) & (g['bridge_h'] == C.BRIDGE_H) & \
            (g['loss_h'] == C.LOSS_H) & (g['loss_km'] == C.LOSS_KM) & (g['stop_h'] == C.STOP_H) & (g['stop_km'] == C.STOP_KM)
        sel, why = g[m].iloc[0], '설정값 사용 (AUTO_SELECT=False)'
    show = ['alarm_q', 'persist_h', 'bridge_h', 'loss_h', 'loss_km', 'stop_h', 'stop_km', 'alarm_per_day', 'total_per_day', 'weighted_recall'] + \
           [c for c in g.columns if c.startswith('recall_')]
    print('  예산 안 상위 조합:')
    print(g[g['alarm_per_day'] <= C.ALARM_BUDGET].sort_values(['weighted_recall', 'alarm_per_day'],
          ascending=[False, True])[show].head(8).to_string(index=False))
    print(f'  선택: {why}'); print(sel[show].to_string())

    # 선택된 기준으로 최종 사건
    t2 = float(q.quantile(sel.alarm_q))
    lh, lkm = int(sel.loss_h), float(sel.loss_km)
    sh_, skm = int(sel.stop_h), float(sel.stop_km)
    habit = U.vessel_loss_habit(df, lh, lkm)
    shabit = U.vessel_stop_habit(df, sh_, skm)
    ev, ev_id = U.detect_events(df, t1, t2, persist_h=int(sel.persist_h), bridge_h=int(sel.bridge_h), t_mean=tm,
                                habit=habit, loss_h=lh, loss_km=lkm, stop_habit=shabit, stop_h=sh_, stop_km=skm)
    df['event_id'] = ev_id
    # [v2.3-7] 반복 경보 억제: 같은 선박의 '경보'는 한 주에 CHRONIC_MAX_PER_WEEK 번까지
    ev = ev.sort_values(['vessel', 't_start']).reset_index(drop=True)
    ev['week'] = ev['start'].dt.to_period('W').astype(str)
    is_al = ev['level'] == '경보'
    nth = ev[is_al].groupby(['vessel', 'week']).cumcount() + 1
    sup = pd.Series(False, index=ev.index); sup.loc[nth.index] = nth > C.CHRONIC_MAX_PER_WEEK
    ev.loc[sup, 'level'] = '주의(반복)'
    ev = ev.drop(columns='week')
    n_sup = int(sup.sum())
    al = ev[ev['level'] == '경보']
    print(f'  반복 경보 억제: {n_sup:,}건을 주의(반복)으로 (같은 선박 주 {C.CHRONIC_MAX_PER_WEEK}회 초과)')
    print(f'[3] 최종 사건 {len(ev):,}건 (경보 {len(al):,}건 = 하루 {len(al) / days:.1f}건), '
          f'경보 선박 {al.vessel.nunique():,}척 ({al.vessel.nunique() / n_v:.1%})')
    print(pd.crosstab(ev['kind'], ev['level']).to_string())
    er = df[df['event_id'].isin(al['event_id'])]
    if len(er):
        top_r = er['reason1'].replace('', '(복합)').value_counts().head(8).index
        print('  경보 사건 시간의 사유 × 상태:')
        print(pd.crosstab(er['reason1'].replace('', '(복합)'), er['state']).reindex(top_r).to_string())

    ev.drop(columns=['t_start', 't_end']).sort_values(['level', 'max_score'], ascending=[True, False]) \
      .to_csv(U.out('alarm_events.csv'), index=False, encoding='utf-8-sig')
    hours = df.groupby('vessel').size()
    vs = ev.groupby('vessel').agg(code=('code', 'first'), events=('event_id', 'size'),
                                  alarm_events=('level', lambda s: int((s == '경보').sum())),
                                  repeat_suppressed=('level', lambda s: int((s == '주의(반복)').sum())),
                                  loss_events=('kind', lambda s: int(s.str.contains('단절').sum())),
                                  stop_events=('kind', lambda s: int(s.str.contains('정지').sum())),
                                  event_hours=('hours', 'sum'), max_score=('max_score', 'max'),
                                  last_event=('end', 'max'), main_kind=('kind', lambda s: s.mode().iat[0]),
                                  top_reason=('top_reason', lambda s: s.replace('', np.nan).mode().iat[0]
                                              if s.replace('', np.nan).notna().any() else ''))
    vs['observed_hours'] = hours.reindex(vs.index)
    vs['events_per_1000h'] = (vs['events'] / vs['observed_hours'] * 1000).round(2)
    vs['loss_rate'] = habit.reindex(vs.index).round(3)
    vs['habitual_loss'] = vs['loss_rate'] >= C.HABIT_LOSS_RATE
    vs['stop_rate'] = shabit.reindex(vs.index).round(3)
    vs['habitual_stop'] = vs['stop_rate'] >= C.HABIT_STOP_RATE
    vs['rankable'] = vs['observed_hours'] >= C.MIN_OBS_H
    vs = vs.sort_values(['rankable', 'alarm_events', 'events_per_1000h', 'max_score'], ascending=[False] * 4)
    vs.to_csv(U.out('alarm_vessels.csv'), encoding='utf-8-sig')

    df['row_level'] = np.where(q >= t2, '경보', np.where(q >= t1, '주의', '정상'))
    df['loss_h'] = U.loss_hours(df, lkm)
    df['stop_h'] = U.stop_runs(df, skm)[0]
    df = df.merge(ev[['event_id', 'kind', 'level']].rename(columns={'kind': 'event_kind', 'level': 'event_level'}),
                  on='event_id', how='left')
    rows = df[df['event_id'] >= 0]
    cols = ['event_id', 'event_kind', 'event_level', 'vessel', 'code', 'date', 'hour', 'state', 'row_level',
            'final_score', 'u_if', 'u_ae', 'u_state', 'mismatch', 'L_total', 'run_active_h', 'dist_port_km',
            'idle_at_sea_km', 'loss_h', 'stop_h', 'reason1', 'reason2', 'reason3', 'reason_ae', 'xe', 'ye', 'src_file', 'row_in_file']
    rows[[c for c in cols if c in rows.columns]].to_csv(U.out('alarm_rows.csv'), index=False, encoding='utf-8-sig')
    dcols = ['vessel', 'code', 'date', 'hour', 'state', 'final_score', 'anomaly_score', 'u_if', 'u_ae', 'u_state',
             'mismatch', 'row_level', 'event_id', 'event_kind', 'event_level', 'reason1', 'reason_ae',
             'dist_port_km', 'idle_at_sea_km', 'loss_h', 'stop_h', 'xe', 'ye', 'src_file', 'row_in_file']
    df[[c for c in dcols if c in df.columns]].to_parquet(U.out('scores_deliverable.parquet'), index=False)

    # 완료 기준 ('경보' 등급 기준)
    T = C.TARGETS
    ok = lambda b: '충족' if bool(b) else '미달'
    rec = lambda t: float(sel.get(f'recall_{t}', np.nan))
    auroc = np.nan
    vpath = U.out('validation_summary.csv')
    if os.path.exists(vpath):
        vsum = pd.read_csv(vpath).set_index('anomaly_type')
        auroc = float(vsum.loc['drift', 'auroc']) if 'drift' in vsum.index else np.nan
    lo, hi = T['events_per_day']
    apd = len(al) / days
    share = top_vessel_share(al, n_v)
    chk = pd.DataFrame([
        dict(criterion='바다 위 신호 단절 사건 탐지율', target=f'≥ {T["signal_loss_recall"]:.0%}', value=rec('signal_loss'),
             ok=ok(rec('signal_loss') >= T['signal_loss_recall'])),
        dict(criterion='6시간 정지 사건 탐지율', target=f'≥ {T["sudden_stop_recall"]:.0%}', value=rec('sudden_stop_6h'),
             ok=ok(rec('sudden_stop_6h') >= T['sudden_stop_recall'])),
        dict(criterion='3시간 정지 사건 탐지율 (참고)', target='-', value=rec('sudden_stop'), ok='참고'),
        dict(criterion='표류 AUROC', target=f'≥ {T["drift_auroc"]}', value=auroc, ok=ok(auroc >= T['drift_auroc'])),
        dict(criterion='표류 사건 탐지율 (참고)', target='-', value=rec('drift'), ok='참고'),
        dict(criterion='하루 경보 사건 수', target=f'≤ {C.ALARM_BUDGET}건 (운영 가정 {lo}~{hi}건)', value=round(apd, 1),
             ok=ok(apd <= C.ALARM_BUDGET)),
        dict(criterion='상위 1% 선박의 경보 사건 비율', target=f'≤ {T["top1pct_vessel_share"]:.0%}', value=round(share, 3),
             ok=ok(share <= T['top1pct_vessel_share'])),
        dict(criterion='상위 경보 사건 20건 중 설명 가능한 사건', target='절반 이상', value=np.nan,
             ok='8단계 검토표(review_sheet.csv)에서 판정')])
    chk.to_csv(U.out('completion_check.csv'), index=False, encoding='utf-8-sig')
    print('[4] 완료 기준 점검'); print(chk.to_string(index=False))

    final = dict(caution=t1, alarm=t2, event_mean=tm, caution_q=C.CAUTION_Q, alarm_q=float(sel.alarm_q),
                 chronic_max_per_week=C.CHRONIC_MAX_PER_WEEK, repeat_suppressed=n_sup,
                 persist_h=int(sel.persist_h), bridge_h=int(sel.bridge_h), loss_h=lh, loss_km=lkm,
                 stop_h=sh_, stop_km=skm, habit_stop_rate=C.HABIT_STOP_RATE, score_sea_km=C.SCORE_SEA_KM,
                 habit_loss_rate=C.HABIT_LOSS_RATE, alarm_budget=C.ALARM_BUDGET, recall_weights=C.RECALL_WEIGHTS,
                 selection=why, alarm_per_day=round(apd, 2), total_per_day=round(len(ev) / days, 2),
                 recalls={t: rec(t) for t in C.RECALL_WEIGHTS}, drift_auroc=auroc, top1pct_vessel_share=share)
    U.save_json(final, U.out('final_thresholds.json'))
    write_standard(U.out('임계값_기준.md'), sel, why, t1, t2, tm, chk,
                   dict(days=days, vessels=n_v, rows=len(df), events=len(ev), alarm_events=len(al),
                        alarm_vessels=al.vessel.nunique()))
    print('  저장: loss_frequency, selection_grid, final_thresholds.json, 임계값_기준.md, alarm_events, '
          f'alarm_vessels, alarm_rows, scores_deliverable.parquet, completion_check ({time.time() - t0:.0f}초)')


if __name__ == '__main__':
    main()
