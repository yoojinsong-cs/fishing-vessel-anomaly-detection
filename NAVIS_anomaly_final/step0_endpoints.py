# -*- coding: utf-8 -*-
"""
0단계 (v2.1 신규). 행별 시작·끝 좌표 추출  [다음 단계 4: 모항 위치 맥락]

특징 테이블에는 좌표가 없어서 '항구에서 끊겼는지, 바다에서 끊겼는지'를 구분할 수 없었다.
원본 CSV를 한 번 읽어 각 행(선박×1시간)의 첫 점과 마지막 점을 미터 좌표(EPSG:5179)로 저장한다.
이후 1·2단계에서 선박별 모항(항해가 자주 시작·끝나는 지점)을 찾고, 모항까지의 거리를 특징으로 쓴다.

출력: OUT_DIR/endpoints.parquet  (src_file, row_in_file, xs, ys, xe, ye)
실행 시간: 원본 CSV를 한 번 읽음 (수 분~10분)
"""
import os
import json
import time

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

import anomaly_config as C
import anomaly_common as U

try:
    import navis_features as NF
except ImportError as e:
    raise ImportError('navis_features.py 를 이 스크립트와 같은 폴더에 두세요.') from e


def endpoints(geoms, transformer, swap):
    """레코드별 (첫 점, 마지막 점) 미터 좌표. 좌표가 없으면 NaN."""
    n = len(geoms)
    out = np.full((n, 4), np.nan)
    ok = np.flatnonzero(~shapely.is_missing(geoms))
    if len(ok) == 0:
        return out
    xy, idx = shapely.get_coordinates(geoms[ok], return_index=True)
    if len(xy) == 0:
        return out
    if swap:
        xy = xy[:, ::-1]
    x, y = transformer.transform(xy[:, 0], xy[:, 1])
    first = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
    last = np.r_[first[1:] - 1, len(idx) - 1]
    rec = ok[idx[first]]
    out[rec, 0], out[rec, 1] = x[first], y[first]
    out[rec, 2], out[rec, 3] = x[last], y[last]
    return out


def main():
    t0 = time.time()
    with open(C.FEAT_THRESH, encoding='utf-8') as fp:
        swap = bool(json.load(fp).get('swap', False))
    tr = Transformer.from_crs(NF.CFG['SRC_CRS'], NF.CFG['DST_CRS'], always_xy=True)
    outs = []
    for path in U.raw_files():
        enc = NF.detect_encoding(path)
        row0 = 0
        for ch in pd.read_csv(path, chunksize=C.CSV_CHUNK_ROWS, encoding=enc, dtype=str,
                              usecols=[C.COL_GEOM], low_memory=False):
            e = endpoints(NF.parse_geoms(ch[C.COL_GEOM]), tr, swap)
            outs.append(pd.DataFrame({'src_file': os.path.basename(path),
                                      'row_in_file': np.arange(row0, row0 + len(ch), dtype='int64'),
                                      'xs': e[:, 0].astype('float32'), 'ys': e[:, 1].astype('float32'),
                                      'xe': e[:, 2].astype('float32'), 'ye': e[:, 3].astype('float32')}))
            row0 += len(ch)
        print(f'  {os.path.basename(path)}: {row0:,}행 ({time.time() - t0:.0f}초)', flush=True)
    res = pd.concat(outs, ignore_index=True)
    res.to_parquet(C.ENDPOINTS, index=False)
    print(f'  좌표 없는 행 {res["xs"].isna().mean():.2%} | 저장: {C.ENDPOINTS}')
    print(f'완료 ({time.time() - t0:.0f}초)')


if __name__ == '__main__':
    main()
