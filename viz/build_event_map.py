"""NAVIS 사건 지도 (인터랙티브 HTML) 생성.
입력: 이상탐지 결과 폴더(기본 out/NAVIS_anomaly/)의 alarm_events.csv, alarm_rows.csv  (좌표: EPSG:5179 미터)
출력: 같은 폴더의 event_map.html (단일 파일, 인터넷 연결 시 Leaflet/지도 타일 로드)
실행: python viz/build_event_map.py [--anon] [--out 폴더]
  --anon : 선박 번호를 앞 3자리만 남기고 가림
"""
import json, sys
from pathlib import Path
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "NAVIS_anomaly_final"))
import navis_paths as P  # noqa: E402  이상탐지 결과 폴더 (기본 out/NAVIS_anomaly)
D = Path(P.OUT_DIR)
ANON = "--anon" in sys.argv
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else D   # 저장 폴더 (공개용 등)

ev = pd.read_csv(D / "alarm_events.csv", encoding="utf-8-sig")
rw = pd.read_csv(D / "alarm_rows.csv", encoding="utf-8-sig",
                 usecols=["event_id", "date", "hour", "xe", "ye", "final_score"])

lon, lat = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True).transform(
    rw.xe.values, rw.ye.values)
rw["lat"], rw["lon"] = lat, lon
rw = rw.sort_values(["event_id", "date", "hour"])

def group(k):
    return "복합" if "+" in k else ("신호 단절" if "단절" in k else ("장시간 정지" if "정지" in k else "지속 이상"))
KINDS = ["신호 단절", "장시간 정지", "지속 이상", "복합"]
LEVELS = ["경보", "주의", "주의(반복)"]
CODES = sorted(ev.code.unique())

paths = {i: g[["lat", "lon"]].round(4).values.tolist() for i, g in rw.groupby("event_id")}
data = []
for r in ev.itertuples():
    p = paths.get(r.event_id)
    if not p:
        continue
    v = str(r.vessel)
    if ANON:
        v = v[:3] + "*" * (len(v) - 3)
    data.append([
        int(r.event_id), p[-1][0], p[-1][1], KINDS.index(group(r.kind)), r.kind,
        LEVELS.index(r.level), CODES.index(r.code), v, float(r.hours),
        round(float(r.max_score), 3), round(float(r.dist_port_km), 1),
        str(r.top_reason), r.start[:16], r.end[:16], p[:60],
    ])

html = (Path(__file__).parent / "event_map_template.html").read_text(encoding="utf-8")
html = (html.replace("__LAND__", (Path(__file__).parent / "korea_land.json").read_text(encoding="utf-8"))
            .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
            .replace("__KINDS__", json.dumps(KINDS, ensure_ascii=False))
            .replace("__LEVELS__", json.dumps(LEVELS, ensure_ascii=False))
            .replace("__CODES__", json.dumps(CODES, ensure_ascii=False)))
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "event_map.html"
out.write_text(html, encoding="utf-8")
print(f"{len(data)} events -> {out} ({out.stat().st_size/1e6:.1f} MB)")
