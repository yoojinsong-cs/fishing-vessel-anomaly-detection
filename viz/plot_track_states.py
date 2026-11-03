# -*- coding: utf-8 -*-
"""
자망 선박의 운항 상태(이동 / 저속 이동(조업 추정) / 정박)별 항적 예시 그림.

  상단: 세 상태의 궤적을 점과 선으로 (파랑=이동, 주황=저속 이동, 회색=정박)
  하단: 선박 ID, 업종, 위치점, 궤적 조각, 이동거리(m), 정지비율 표

navis_features.py 의 함수(parse_geoms, records_to_parts, clean_part, record_features)와
NAVIS_features/thresholds.json 의 임계값을 그대로 사용한다. 원본 CSV 가 없으면 --demo 로 샘플 재현.

실행:  python viz/plot_track_states.py [--demo] [--month 1] [--vessel ID ID ID]
출력:  out/track_states.png
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'NAVIS_anomaly_code_v2.5'))

COLORS = {'move': '#1f77b4', 'slow': '#ff7f0e', 'dock': '#8c8c8c'}
LABELS = {'move': '이동', 'slow': '저속 이동 (조업 추정)', 'dock': '정박'}
CODE = '자망'
DATA_DIR = os.path.join(ROOT, 'data', 'vpass_1월_6월 데이터')
THR_PATH = os.path.join(ROOT, 'NAVIS_features', 'thresholds.json')
OUT = os.path.join(ROOT, 'out', 'track_states.png')


def setup_font():
    for name in ('Malgun Gothic', 'NanumGothic', 'AppleGothic'):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------
# 실제 데이터: 자망 레코드에서 상태별 대표 1건씩 고르기
# ---------------------------------------------------------------------
def load_real(month, vessels=None, max_chunks=6, per_chunk=400):
    import navis_features as NF
    from pyproj import Transformer
    thr = json.load(open(THR_PATH, encoding='utf-8'))
    path = os.path.join(DATA_DIR, f'vpass_{month}월 데이터.csv')
    enc = NF.detect_encoding(path)
    tf = Transformer.from_crs(NF.CFG['SRC_CRS'], NF.CFG['DST_CRS'], always_xy=True)
    tf_back = Transformer.from_crs(NF.CFG['DST_CRS'], NF.CFG['SRC_CRS'], always_xy=True)
    names = NF.feature_names(NF.CFG)

    cands = []
    for ci, ch in enumerate(NF.read_chunks(path, enc)):
        if ci >= max_chunks:
            break
        ch = ch[ch[NF.LABEL_COL].astype(str).str.strip() == CODE]
        if vessels:
            ch = ch[ch[NF.ID_COL].isin(vessels)]
        if ch.empty:
            continue
        ch = ch.sample(n=min(len(ch), per_chunk), random_state=ci).reset_index(drop=True)   # 속도: 청크당 표본만 계산
        parts = NF.records_to_parts(NF.parse_geoms(ch[NF.GEOM_COL]), tf, thr['swap'])
        for i, p in enumerate(parts):
            n = int(sum(len(q) for q in p))
            if n < 4:
                continue
            f = NF.record_features(p, thr, NF.CFG, names)
            if not f.get('spd_valid'):                  # 점 수가 최빈값 근처인 행만 상태 비율이 계산됨
                continue
            pcs = []
            for q in p:
                pcs += NF.clean_part(q, NF.CFG['JUMP_M'], NF.CFG['MAX_STEP_M'])[0]
            pcs = [q for q in pcs if len(q)]
            if not pcs:
                continue
            row = ch.iloc[i]
            dist = float(sum(NF.pathlen(q) for q in pcs))          # 조각별 경로 길이의 합 (m)
            pts = np.vstack(pcs)
            cands.append(dict(vessel=row[NF.ID_COL], code=row[NF.LABEL_COL].strip(), pieces=pcs,
                              n_pts=n, n_parts=len(p), dist=dist,
                              span=float(np.ptp(pts, axis=0).max()),
                              stop=f['spd_bin_stop'],                # 정지비율 = 점간 거리가 정지 임계 미만인 스텝 비율
                              s_stop=f['spd_st_stop'], slow=f['spd_st_fish'], move=f['spd_st_move']))
        if len(set(classify(cands))) == 3 and len(cands) > 300:
            break
    return cands, tf_back


def classify(cands):
    """레코드의 지배적 상태 (평활한 상태 비율 중 가장 큰 것)로 분류. 키 집합 반환용"""
    out = []
    for c in cands:
        s = {'dock': c['s_stop'], 'slow': c['slow'], 'move': c['move']}
        c['state'] = max(s, key=s.get)
        out.append(c['state'])
    return out


def pick(cands, vessels=None):
    classify(cands)
    rng = np.random.default_rng(0)
    rules = {   # 상태별 대표 레코드 조건: (필터, 정렬 키)
        'move': (lambda c: c['n_parts'] <= 2 and c['dist'] > 5000,        lambda c: -c['move']),
        'slow': (lambda c: c['n_parts'] == 1 and 500 < c['dist'] < 5000 and c['span'] < 0.5 * c['dist'],
                 lambda c: -c['slow']),                                    # 좁은 구역을 왕복하는 궤적
        'dock': (lambda c: c['n_parts'] == 1 and c['span'] < 100,         lambda c: -c['s_stop']),
    }
    chosen = {}
    for st, (ok, key) in rules.items():
        pool = [c for c in cands if c['state'] == st and ok(c)]
        if not pool:
            raise SystemExit(f'{st} 상태 레코드를 찾지 못했습니다 (--month 를 바꾸거나 max_chunks 를 늘리세요)')
        pool.sort(key=key)
        top = pool[:max(5, len(pool) // 10)]
        chosen[st] = top[int(rng.integers(len(top)))]
    return chosen


# ---------------------------------------------------------------------
# 샘플 데이터 (원본이 없을 때)
# ---------------------------------------------------------------------
def demo():
    rng = np.random.default_rng(1)
    def walk(n, step, turn, start):
        ang = rng.uniform(0, 2 * np.pi)
        xy = [np.array(start, float)]
        for _ in range(n - 1):
            ang += rng.normal(0, turn)
            xy.append(xy[-1] + step * np.array([np.cos(ang), np.sin(ang)]) * rng.uniform(0.6, 1.4))
        return np.array(xy)
    spec = {'move': (14, 700, 0.12, 99000001, 3, 9800, 0.07),
            'slow': (14, 60, 0.9, 99000002, 2, 840, 0.36),
            'dock': (14, 1.5, 1.5, 99000003, 1, 22, 0.93)}
    out = {}
    for st, (n, step, turn, vid, parts, dist, stop) in spec.items():
        xy = walk(n, step, turn, (0, 0))
        out[st] = dict(vessel=vid, code=CODE, pieces=[xy], n_pts=n, n_parts=parts, dist=dist, stop=stop, state=st)
    return out, None


# ---------------------------------------------------------------------
# 그리기
# ---------------------------------------------------------------------
def draw(chosen, back, out, demo_mode):
    setup_font()
    fig = plt.figure(figsize=(13, 6.9), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.1], hspace=0.28, wspace=0.22,
                          left=0.05, right=0.98, top=0.93, bottom=0.04)
    for k, st in enumerate(('move', 'slow', 'dock')):
        ax = fig.add_subplot(gs[0, k])
        c = chosen[st]
        col = COLORS[st]
        o = c['pieces'][0][0]                                   # 첫 점을 원점(0,0)으로, 단위 m
        allp = np.vstack(c['pieces']) - o
        for piece in c['pieces']:
            x, y = (piece - o).T
            ax.plot(x, y, '-', color=col, lw=1.4, alpha=0.9, zorder=2)
            ax.scatter(x, y, s=22, color=col, edgecolor='white', linewidth=0.5, zorder=3)
        ax.scatter([0], [0], s=80, facecolor='none', edgecolor='black', linewidth=1.2, zorder=4)
        cx, cy = (allp.max(0) + allp.min(0)) / 2
        half = max(np.ptp(allp, axis=0).max() / 2 * 1.15, 60)    # 최소 표시 범위 120 m (정박의 GPS 흔들림이 과장되지 않게)
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect('equal')
        ax.set_title(LABELS[st], color=col, fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('동서 (m, ○ = 첫 점)', fontsize=8); ax.set_ylabel('남북 (m)', fontsize=8)
        for s_ in ax.spines.values():
            s_.set_edgecolor(col); s_.set_linewidth(1.5)
    ax = fig.add_subplot(gs[1, :]); ax.axis('off')
    cols = ['상태', '선박 ID', '업종', '위치점', '궤적 조각', '이동거리(m)', '정지비율']
    rows = []
    for st in ('move', 'slow', 'dock'):
        c = chosen[st]
        rows.append([LABELS[st], str(c['vessel']), c['code'], f"{c['n_pts']:,}", f"{c['n_parts']}",
                     f"{c['dist']:,.0f}", f"{c['stop']:.2f}"])
    tb = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 1.9)
    for (r, cidx), cell in tb.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_facecolor('#f0f0f0'); cell.set_text_props(fontweight='bold')
        elif cidx == 0:
            cell.set_text_props(color=COLORS[('move', 'slow', 'dock')[r - 1]], fontweight='bold')
    fig.suptitle('자망 선박의 운항 상태별 항적 특징 예시' + (' (샘플 데이터)' if demo_mode else ' (V-Pass 실제 레코드, 1시간 단위)'),
                 fontsize=14, fontweight='bold')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print('저장:', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', action='store_true', help='원본 없이 샘플 데이터로 재현')
    ap.add_argument('--month', type=int, default=1)
    ap.add_argument('--vessel', nargs='*', help='특정 선박 ID로 한정')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    if a.demo:
        chosen, back = demo()
    else:
        cands, back = load_real(a.month, set(a.vessel) if a.vessel else None)
        print(f'후보 레코드 {len(cands):,}건')
        chosen = pick(cands)
    for st, c in chosen.items():
        print(st, c['vessel'], f"점 {c['n_pts']}, 조각 {c['n_parts']}, 거리 {c['dist']:.0f} m, 정지 {c['stop']:.2f}")
    draw(chosen, back, a.out, a.demo)


if __name__ == '__main__':
    main()
