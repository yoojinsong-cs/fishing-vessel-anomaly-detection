# -*- coding: utf-8 -*-
"""
1단계 (v2.3). 시간대별 EDA → 업종 × 시간대 '정상' 기준 초안

출력 (OUT_DIR)
  eda_hourly.csv          업종×시간대별 레코드 수, 정박 비율, 이동 거리·점 수 분포
  eda_monthly.csv         업종×월별 같은 지표 (계절 변화 확인용)
  eda_*.png               시간대별 정박 비율 / 시간당 이동 거리 / 점 수 / 레코드 공백 그래프
  normal_baseline.parquet 업종×시간대×상태 그룹별, 특징별 정상 분포(1·5·25·50·75·95·99 백분위)
  normal_baseline_summary.csv  주요 특징의 정상 범위를 원래 단위로 정리한 표 (보고서용)
"""
import time
import numpy as np
import pandas as pd

import anomaly_config as C
import anomaly_common as U


def main():
    t0 = time.time()
    print('[1] 특징 테이블 읽기')
    df = U.load_features()
    df = U.add_time_context(df)          # [개선 1·5] 공백(첫·마지막은 중립), 항해 맥락 특징
    df, _ = U.add_port_context(df)       # [v2.1 D] 모항까지 거리, 바다 위 끊김
    df['grp'] = U.group_key(df)
    print(f'  {len(df):,}행, 선박 {df.vessel.nunique():,}척')

    # ----- 시간대별 EDA -----
    print('[2] 업종 × 시간대 EDA')
    act = df['state'] == 'active'
    g = df.groupby(['code', 'hour'])
    hourly = pd.DataFrame({
        'rows': g.size(),
        'stationary_share': g['state'].apply(lambda s: (s == 'stationary').mean()),
        'npts_median': g['n_points_raw'].median(),
        'gap_before_median_h': g['gap_before_h'].median(),
    })
    ga = df[act].groupby(['code', 'hour'])['L_total']
    hourly['dist_median_m'] = ga.median()
    hourly['dist_p90_m'] = ga.quantile(0.9)
    # 각 업종의 활동 행이 시간대에 분포한 비율 (업종별 합 100%)
    share = df[act].groupby(['code', 'hour']).size()
    hourly['active_share_of_day_pct'] = share / share.groupby(level=0).transform('sum') * 100
    hourly = hourly.round(4)
    hourly.to_csv(U.out('eda_hourly.csv'), encoding='utf-8-sig')

    gm = df.groupby(['code', 'month'])
    monthly = pd.DataFrame({'rows': gm.size(),
                            'stationary_share': gm['state'].apply(lambda s: (s == 'stationary').mean()),
                            'npts_median': gm['n_points_raw'].median(),
                            'dist_median_m': df[act].groupby(['code', 'month'])['L_total'].median()}).round(4)
    monthly.to_csv(U.out('eda_monthly.csv'), encoding='utf-8-sig')

    # 콘솔 요약: 업종별 정박이 가장 많은/적은 시간대
    for code, h in hourly.groupby(level=0):
        s = h['stationary_share'].droplevel(0)
        d = h['dist_median_m'].droplevel(0)
        print(f'  {code:8s} 정박 비율 최고 {s.idxmax():2d}시({s.max():.0%}) 최저 {s.idxmin():2d}시({s.min():.0%}) | '
              f'이동 거리 최대 {d.idxmax():2d}시({d.max():,.0f} m)')

    # ----- 그래프 -----
    plt = U.setup_korean_font()
    plots = [('stationary_share', '정박 행 비율', 'eda_stationary_share.png', 100, '%'),
             ('dist_median_m', '시간당 이동 거리 중앙값 (활동 행)', 'eda_distance.png', 1, 'm'),
             ('npts_median', '시간당 원래 점 수 중앙값', 'eda_npoints.png', 1, '개'),
             ('active_share_of_day_pct', '하루 활동 중 각 시간대 비율', 'eda_active_share.png', 1, '%')]
    for col, title, fname, mul, unit in plots:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for code in C.CLASSES:
            if code in hourly.index.get_level_values(0):
                y = hourly.loc[code, col] * mul
                ax.plot(y.index, y.values, marker='o', ms=3, lw=2.2 if code == '채낚기연승낚시' else 1.4, label=code)
        ax.set_title(title); ax.set_xlabel('시간대(시)'); ax.set_ylabel(unit)
        ax.set_xticks(range(0, 24, 3)); ax.grid(alpha=.3); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(U.out(fname), dpi=150); plt.close(fig)

    # ----- 정상 기준 (그룹별 분위수) -----
    print('[3] 업종 × 시간대 × 상태별 정상 기준 계산')
    base = U.compute_baseline(df)
    base.to_parquet(U.out('normal_baseline.parquet'), index=False)
    small = base.groupby('grp')['n'].max()
    if (small < 200).any():
        print(f'  [주의] 표본이 200개 미만인 그룹 {int((small < 200).sum())}개: {list(small[small < 200].index)[:6]} ...')

    # 보고서용 요약: 주요 특징의 정상 범위(5~95 백분위)를 원래 단위로
    key = ['L_total', 'straightness', 'th_mean_100', 'stop_n', 'gap_after_h']
    s = base[base['feature'].isin(key) & base['grp'].str.endswith('|active')].copy()
    back = lambda v, f: np.expm1(v) if f in C.LOG_FEATURES else v
    for c in ['p05', 'med', 'p95']:
        s[c] = [back(v, f) for v, f in zip(s[c], s['feature'])]
    s[['code', 'hour', 'state']] = s['grp'].str.split('|', expand=True).iloc[:, :3]
    summ = s.pivot_table(index=['code', 'hour'], columns='feature', values=['p05', 'med', 'p95'])
    summ.columns = [f'{f}_{q}' for q, f in summ.columns]
    summ = summ[[f'{f}_{q}' for f in key for q in ['p05', 'med', 'p95'] if f'{f}_{q}' in summ.columns]].round(3)
    summ.to_csv(U.out('normal_baseline_summary.csv'), encoding='utf-8-sig')
    print(f'  그룹 {base.grp.nunique()}개 × 특징 → {U.out("normal_baseline.parquet")}')
    print(f'완료 ({time.time() - t0:.0f}초)')


if __name__ == '__main__':
    main()
