# -*- coding: utf-8 -*-
"""
8단계 (v2.5). 상위 경보 사건 사례 검토 — 완료 기준 6 '상위 경보 사건 20건 중 절반 이상 설명 가능'

정답 라벨이 없으므로 마지막 확인은 사람이 직접 항적을 보고 한다.
'경보' 사건을 REVIEW_TOP_N 건 고른다 (같은 선박은 한 건만 → 쏠림이 검토를 가리지 않게).
[v2.4] 사건 종류(점수 이상 / 바다 위 정지 / 바다 위 단절)별로 실제 경보 비율만큼 나눠 뽑되 종류마다 최소 3건,
       종류 안에서는 우선순위(점수: 평균 점수 × 지속 시간, 정지·단절: 시간 × log(1+모항 거리)) 순.
       v2.3에서는 상위 20건의 최고 점수가 모두 1.0이라 순서를 가를 수 없었다.
사건 앞뒤 6시간의 원본 항적을 미터 좌표로 그린다. 검토표에 O/X 와 메모를 적어 판정한다.

그림 읽는 법
  회색 선 = 사건 앞뒤의 항적, 색 점 = 사건 시간 (진할수록 점수가 높음), ★ = 그 선박의 모항,
  × = 사건의 마지막 기록 지점 (신호 단절이면 여기서 끊김)

출력 (OUT_DIR/review)
  event_01.png ...   사건별 지도
  review_sheet.csv   검토표 (설명 가능 O/X, 메모 칸을 채운다)
"""
import os
import time

import joblib
import numpy as np
import pandas as pd

import anomaly_config as C
import anomaly_common as U
import step6_validate as V

PAD_H = 6


def main():
    t0 = time.time()
    rdir = U.out('review')
    os.makedirs(rdir, exist_ok=True)
    ev = pd.read_csv(U.out('alarm_events.csv'), dtype={'vessel': str})
    ev = ev[ev['level'] == '경보'].copy()
    ev['family'] = np.where(ev['kind'].str.contains('정지'), '정지',
                            np.where(ev['kind'] == '바다 위 신호 단절', '단절', '점수'))
    dist = np.log1p(ev['dist_port_km'].fillna(0).clip(lower=0))
    ev['priority'] = np.select([ev['family'] == '점수', ev['family'] == '정지'],
                               [ev['mean_score'] * ev['hours'].clip(upper=24), ev['stop_h'].fillna(0).clip(upper=24) * dist],
                               ev['loss_h'].fillna(0).clip(upper=72) * dist)   # [v2.5-3] 극단적 길이가 위를 독차지하지 않게
    ev = ev.sort_values('priority', ascending=False).drop_duplicates('vessel')
    share = ev['family'].value_counts(normalize=True)
    quota = {f: max(3, int(round(C.REVIEW_TOP_N * share.get(f, 0)))) for f in share.index}
    while sum(quota.values()) > C.REVIEW_TOP_N:              # 합이 넘으면 가장 많은 종류에서 뺀다
        quota[max(quota, key=quota.get)] -= 1
    ev = pd.concat([ev[ev['family'] == f].head(q) for f, q in quota.items()]).reset_index(drop=True)
    print(f'  종류별 검토 수: {ev["family"].value_counts().to_dict()}')
    if len(ev) == 0:
        print('  경보 사건이 없습니다.'); return
    ev['start'] = pd.to_datetime(ev['start']); ev['end'] = pd.to_datetime(ev['end'])
    sc = pd.read_parquet(U.out('scores_deliverable.parquet'),
                         columns=['vessel', 'date', 'hour', 'final_score', 'event_id', 'reason1', 'src_file', 'row_in_file'])
    sc['vessel'] = sc['vessel'].astype(str)
    sc = sc[sc['vessel'].isin(ev['vessel'])].copy()
    sc['ts'] = pd.to_datetime(sc['date'].astype(str)) + pd.to_timedelta(sc['hour'].astype(int), unit='h')
    win = []
    for r in ev.itertuples(index=False):
        m = (sc['vessel'] == r.vessel) & (sc['ts'] >= r.start - pd.Timedelta(hours=PAD_H)) & \
            (sc['ts'] <= r.end + pd.Timedelta(hours=PAD_H))
        win.append(sc[m].assign(case=r.event_id))
    win = pd.concat(win, ignore_index=True)
    print(f'[1] 경보 사건 {len(ev)}건 (선박당 1건), 그릴 레코드 {len(win):,}행 — 원본 CSV 읽는 중')
    transformer, swap = V.AE.feature_transformer()
    geo = V.read_rows(win[U.KEYS].drop_duplicates(), transformer, swap)
    bundle = joblib.load(U.out('model_if.joblib'))
    ports = bundle.get('ports')

    plt = U.setup_korean_font()
    sheet = []
    for i, r in enumerate(ev.itertuples(index=False), 1):
        w = win[win['case'] == r.event_id].sort_values('ts')
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        allxy = []
        for row in w.itertuples(index=False):
            parts = geo.get((row.src_file, int(row.row_in_file)))
            if not parts:
                continue
            pts = np.vstack([p for p in parts if len(p)])
            allxy.append(pts)
            in_ev = row.event_id == r.event_id
            ax.plot(pts[:, 0], pts[:, 1], '-', color='0.75', lw=0.8, zorder=1)
            if in_ev:
                ax.scatter(pts[:, 0], pts[:, 1], s=14, c=[row.final_score] * len(pts), cmap='Reds',
                           vmin=0.9, vmax=1.0, edgecolors='none', zorder=6)
        if allxy:
            xy = np.vstack(allxy)
            last_ev = w[w['event_id'] == r.event_id].tail(1)
            if len(last_ev):
                lp = geo.get((last_ev['src_file'].iat[0], int(last_ev['row_in_file'].iat[0])))
                if lp:
                    e = np.vstack([p for p in lp if len(p)])[-1]
                    ax.scatter([e[0]], [e[1]], marker='x', s=60, c='k', linewidths=1, zorder=4)
            # 화면은 항적에 맞추고, 20km 안의 모항만 화면에 넣는다 (먼 모항은 거리만 표시)
            pad = max(1000.0, 0.1 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
            lim = [xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad]
            if isinstance(ports, pd.DataFrame) and len(ports):
                pv = ports[ports['vessel'].astype(str) == str(r.vessel)]
                if len(pv):
                    cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
                    d = np.hypot(pv['x'].values - cx, pv['y'].values - cy) / 1000
                    near = pv[d <= 20]
                    ax.scatter(near['x'], near['y'], marker='*', s=160, c='gold', edgecolors='k', zorder=5)
                    for px, py in near[['x', 'y']].to_numpy():
                        lim = [min(lim[0], px - pad), max(lim[1], px + pad), min(lim[2], py - pad), max(lim[3], py + pad)]
                    ax.text(0.01, 0.99, f'가장 가까운 모항까지 {d.min():.1f} km', transform=ax.transAxes,
                            fontsize=8, va='top', color='0.25')
            ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
        ax.set_aspect('equal'); ax.ticklabel_format(useOffset=False, style='plain')
        ax.text(0.01, 0.01, '회색 = 앞뒤 6시간 항적, 빨간 점 = 사건 시간, × = 사건 마지막 기록, ★ = 모항',
                transform=ax.transAxes, fontsize=7, color='0.35', va='bottom')
        ax.tick_params(labelsize=7); ax.set_xlabel('x (m, EPSG:5179)', fontsize=8); ax.set_ylabel('y (m)', fontsize=8)
        loss = f', 끊김 {r.loss_h:.0f}시간' if pd.notna(r.loss_h) and r.loss_h > 0 else ''
        stp = f', 정지 {r.stop_h:.0f}시간' if pd.notna(r.stop_h) and r.stop_h > 0 else ''
        ax.set_title(f'#{i} 선박 {r.vessel} ({r.code}) · {r.kind}\n'
                     f'{r.start:%m-%d %H시}~{r.end:%m-%d %H시}, {r.hours:.0f}시간, 평균 {r.mean_score:.3f}{loss}{stp}\n'
                     f'주된 사유: {r.top_reason or "(복합)"}', fontsize=9)
        png = f'event_{i:02d}.png'
        fig.tight_layout(); fig.savefig(os.path.join(rdir, png), dpi=130); plt.close(fig)
        sheet.append(dict(rank=i, family=r.family, priority=round(r.priority, 3), event_id=r.event_id,
                          vessel=r.vessel, code=r.code, kind=r.kind,
                          start=r.start, end=r.end, hours=r.hours, max_score=round(r.max_score, 4),
                          top_reason=r.top_reason, loss_h=r.loss_h, stop_h=r.stop_h, dist_port_km=r.dist_port_km, png=png,
                          **{'설명가능(O/X)': '', '메모': ''}))
    pd.DataFrame(sheet).to_csv(os.path.join(rdir, 'review_sheet.csv'), index=False, encoding='utf-8-sig')
    print(f'  저장: {rdir} (그림 {len(sheet)}장 + review_sheet.csv) ({time.time() - t0:.0f}초)')
    print('  review_sheet.csv 의 설명가능(O/X) 칸을 채우면 완료 기준 6을 판정할 수 있습니다 (O가 절반 이상이면 충족).')


if __name__ == '__main__':
    main()
