"""Natural Earth(world-atlas 10m)에서 한반도 주변 육지를 잘라 korea_land.json(GeoJSON)으로 저장.
사용: python viz/make_land.py <countries-10m.json 경로>
출처: Natural Earth (퍼블릭 도메인), https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-10m.json
"""
import json, sys
from pathlib import Path
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

topo = json.load(open(sys.argv[1], encoding="utf-8"))
sc, tr = topo["transform"]["scale"], topo["transform"]["translate"]
arcs = []
for a in topo["arcs"]:
    x = y = 0; pts = []
    for dx, dy in a:
        x += dx; y += dy
        pts.append((x * sc[0] + tr[0], y * sc[1] + tr[1]))
    arcs.append(pts)

def ring(idx):
    out = []
    for i in idx:
        p = arcs[i] if i >= 0 else arcs[~i][::-1]
        out += p if not out else p[1:]
    return out

BBOX = box(122.0, 31.0, 133.5, 40.5)
polys = []
def add(geom):
    rings = [r for r in (ring(r) for r in geom) if len(r) >= 4]
    if not rings:
        return
    p = Polygon(rings[0], rings[1:]).buffer(0)
    if not p.is_empty:
        polys.append(p)
for g in topo["objects"]["countries"]["geometries"]:
    if g["type"] == "Polygon":
        add(g["arcs"])
    elif g["type"] == "MultiPolygon":
        for part in g["arcs"]:
            add(part)
land = unary_union(polys).intersection(BBOX).simplify(0.005)
# 소수점 3자리, [lat, lon] 순서의 링 목록으로 압축
def rings_of(poly):
    return [[[round(y, 3), round(x, 3)] for x, y in r.coords] for r in [poly.exterior, *poly.interiors]]
geoms = list(land.geoms) if land.geom_type == "MultiPolygon" else [land]
out = [rings_of(p) for p in geoms if p.area > 0.002]
Path(__file__).with_name("korea_land.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(len(out), "polygons,", Path(__file__).with_name("korea_land.json").stat().st_size // 1024, "KB")
