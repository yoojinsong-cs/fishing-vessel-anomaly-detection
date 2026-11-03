"""NAVIS 이상탐지 실제 결과 대시보드 (실시간 갱신형).

입력(이상탐지 결과 폴더, 기본 out/NAVIS_anomaly/): alarm_events.csv, alarm_rows.csv, alarm_vessels.csv,
                      vessel_mismatch.csv, final_thresholds.json, completion_check.csv
출력(같은 폴더): dashboard.html, dashboard_data.json
경로는 NAVIS_anomaly_final/navis_paths.py 에서 가져온다 (run_all.py 결과를 그대로 읽음)

사용법
  python viz/build_dashboard.py            # 한 번 생성 (dashboard.html은 파일로 바로 열어도 됨)
  python viz/build_dashboard.py --serve    # 로컬 서버 + 결과 파일이 바뀌면 자동 재생성,
                                           # 열려 있는 화면은 15초마다 새 데이터를 가져옴
  옵션: --port 8000   --anon (선박 번호 가리기)

파이프라인(step4~step8)을 다시 실행해 CSV가 갱신되면 --serve 중인 대시보드에 자동 반영된다.
"""
import argparse, json, os, sys, threading, time, functools, http.server, socketserver
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "NAVIS_anomaly_final"))
import navis_paths as P  # noqa: E402  이상탐지 결과 폴더 (기본 out/NAVIS_anomaly)
import test_config as T  # noqa: E402  테스트B 결과 폴더 (기본 out/NAVIS_anomaly/test_scoring/out)
D = Path(P.OUT_DIR)
HERE = Path(__file__).resolve().parent
INPUTS = ["alarm_events.csv", "alarm_rows.csv", "alarm_vessels.csv", "vessel_mismatch.csv",
          "final_thresholds.json", "completion_check.csv"]
CODES = ["자망", "통발", "연안복합", "채낚기연승낚시"]
CELL = 0.25          # 위험 해역 집계 격자(도)
TOP_VESSELS = 30     # 시계열을 담을 상위 선박 수


def rd(name):
    return pd.read_csv(D / name, encoding="utf-8-sig")


def ms(s):
    return (pd.to_datetime(s) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")


def grid_cells(al):
    """경보 사건(lat, lon, vessel, kind 열 필요)을 격자별로 집계 -> [lat0, lon0, 건수, 선박수, 단절, 정지, 등급]"""
    al = al.copy()
    al["gi"], al["gj"] = np.floor(al.lat / CELL).astype(int), np.floor(al.lon / CELL).astype(int)
    g = al.groupby(["gi", "gj"]).agg(n=("vessel", "size"), vessels=("vessel", "nunique"),
                                     loss=("kind", lambda s: int(s.str.contains("단절").sum())),
                                     stop=("kind", lambda s: int(s.str.contains("정지").sum()))).reset_index()
    q_hi, q_mid = g.n.quantile(0.90), g.n.quantile(0.70)
    g["tier"] = np.where(g.n >= q_hi, 2, np.where(g.n >= q_mid, 1, 0))
    return [[round(r.gi * CELL, 3), round(r.gj * CELL, 3), int(r.n), int(r.vessels), r.loss, r.stop, int(r.tier)]
            for r in g.itertuples()]


def build(anon=False):
    ev, rw, vs = rd("alarm_events.csv"), rd("alarm_rows.csv"), rd("alarm_vessels.csv")
    mm, cc = rd("vessel_mismatch.csv"), rd("completion_check.csv")
    th = json.loads((D / "final_thresholds.json").read_text(encoding="utf-8"))
    mask = (lambda v: str(v)[:3] + "*" * (len(str(v)) - 3)) if anon else str
    ev["vessel"], rw["vessel"], vs["vessel"] = [x["vessel"].astype(str) for x in (ev, rw, vs)]

    # --- 결과 1: 혼동행렬 (모든 선박, 교차검증 예측) ---
    ct = pd.crosstab(mm.code, mm.pred_code).reindex(index=CODES, columns=CODES, fill_value=0)
    n_row = ct.sum(axis=1)
    pct = (ct.div(n_row.replace(0, np.nan), axis=0) * 100).round(0).fillna(0).astype(int)
    diag = np.diag(ct.values)
    confusion = dict(labels=CODES, counts=ct.values.tolist(), pct=pct.values.tolist(),
                     n=n_row.tolist(), acc=round(float(diag.sum() / ct.values.sum()) * 100, 1),
                     bal_acc=round(float(np.mean(diag / n_row.replace(0, np.nan).values)) * 100, 1))

    # --- 사건 위치 (마지막 관측 지점) ---
    rw["ts"] = ms(pd.to_datetime(rw.date) + pd.to_timedelta(rw.hour, unit="h"))
    rw = rw.sort_values(["event_id", "ts"])
    lon, lat = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True).transform(rw.xe.values, rw.ye.values)
    rw["lat"], rw["lon"] = lat, lon
    last = rw.groupby("event_id").tail(1).set_index("event_id")
    ev = ev.join(last[["lat", "lon"]], on="event_id").dropna(subset=["lat"])
    ev["s_ms"], ev["e_ms"] = ms(ev.start), ms(ev.end) + 3600_000

    # --- 결과 3: 위험 해역 (경보 사건을 격자별로 집계) ---
    cells = grid_cells(ev[ev.level == "경보"])
    data_end = ev.e_ms.max()
    cur = ev[(ev.level == "경보") & (ev.e_ms >= data_end - 24 * 3600_000)]
    current = [[round(r.lat, 4), round(r.lon, 4), mask(r.vessel), r.code, r.kind, r.start[:16], r.end[:16],
                round(float(r.dist_port_km), 1)] for r in cur.itertuples()]

    # --- 결과 4 + 결과 2: 상위 선박과 시계열 ---
    ok = vs[(vs.alarm_events > 0) & vs.rankable & ~vs.habitual_loss & ~vs.habitual_stop]
    top = ok.sort_values(["alarm_events", "events_per_1000h"], ascending=False).head(TOP_VESSELS)
    vessels = []
    for r in top.itertuples():
        rows = rw[rw.vessel == r.vessel]
        es = ev[ev.vessel == r.vessel]
        vessels.append(dict(
            vessel=mask(r.vessel), code=r.code, alarm_events=int(r.alarm_events), events=int(r.events),
            max_score=round(float(r.max_score), 4), kind=r.main_kind, reason=r.top_reason,
            last=str(r.last_event)[:16], per1000=round(float(r.events_per_1000h), 1),
            rows=[[int(t), round(float(s), 4), lv] for t, s, lv in zip(rows.ts, rows.final_score, rows.row_level)],
            spans=[[int(a), int(b), lvl] for a, b, lvl in zip(es.s_ms, es.e_ms, es.level)]))

    kpi = dict(events=len(ev), alarm_events=int((ev.level == "경보").sum()),
               vessels=int(vs.shape[0]), alarm_vessels=int((vs.alarm_events > 0).sum()),
               per_day=th.get("alarm_per_day"), data_start=ev.start.min()[:10], data_end=ev.end.max()[:10])
    comp = cc.fillna("").astype(str).values.tolist()
    return dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), thresholds=dict(
        caution=th["caution"], alarm=th["alarm"], budget=th["alarm_budget"], per_day=th["alarm_per_day"],
        loss_h=th["loss_h"], loss_km=th["loss_km"], stop_h=th["stop_h"], stop_km=th["stop_km"],
        persist_h=th["persist_h"], recalls=th["recalls"]),
        confusion=confusion, cells=cells, cell=CELL, current=current, vessels=vessels,
        kpi=kpi, completion=comp, data_end_ms=int(data_end))


# 테스트B 화면 입력: 테스트셋 전체 결과(step10_full.py + make_submission.py)가 있으면 그것, 없으면 9단계(제출샘플 선박) 결과
FULLD = Path(T.TEST_DIR) / "full"
FULL_FILES = {"events": FULLD / "full_testB_events.csv", "scores": FULLD / "full_testB_scores.parquet",
              "xlsx": ROOT / "out" / "submission.xlsx", "why": FULLD / "full_testB_근거.csv"}
OUTD = Path(T.TEST_OUT)
STEP9_FILES = {"events": OUTD / "testB_events.csv", "scores": OUTD / "testB_scores.parquet",
               "xlsx": OUTD / "testB_submission.xlsx", "why": OUTD / "testB_submission_근거.csv"}
TOP_B = 300          # 테스트B 화면 목록·시계열에 담을 선박 수 (사고시각 점수 높은 순)


def testB_files():
    for fs in (FULL_FILES, STEP9_FILES):
        if all(f.exists() for f in fs.values()):
            return fs
    return None


def build_testB(anon=False):
    """테스트B 제출 결과 -> 대시보드 데이터. 구조는 build()와 같고 mode='testB'."""
    fs = testB_files()
    ev = pd.read_csv(fs["events"], encoding="utf-8-sig")
    sc = pd.read_parquet(fs["scores"], columns=["vessel", "date", "hour", "xe", "ye"])
    gr = pd.read_csv(fs["why"], encoding="utf-8-sig")
    xl = pd.read_excel(fs["xlsx"], sheet_name=None)
    A, B = xl["테스트A"], xl["테스트B"]
    mask = (lambda v: str(v)[:3] + "*" * (len(str(v)) - 3)) if anon else str
    tr = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
    ev["vessel"], gr["rfid_id"], B["rfid_id"] = ev.vessel.astype(str), gr.rfid_id.astype(str), B.rfid_id.astype(str)
    N = len(gr)                                   # 테스트B 선박 수 (제출샘플 100척 또는 테스트셋 전체)
    gr["final_score"] = gr["final_score"].fillna(gr["이상항적점수"])     # 유효한 항적이 없던 선박
    gr["업종"], gr["업종근거"] = gr["업종"].fillna("-"), gr["업종근거"].fillna("-")
    sc["vessel"] = sc.vessel.astype(str)
    sc["key"] = sc.date.dt.strftime("%Y%m%d") + " " + sc.hour.astype(str).str.zfill(2)
    lon, lat = tr.transform(sc.xe.values.astype(float), sc.ye.values.astype(float))
    sc["lat"], sc["lon"] = lat, lon
    pos = sc.drop_duplicates(["vessel", "key"]).set_index(["vessel", "key"])[["lat", "lon"]]

    ev["kend"] = pd.to_datetime(ev["end"]).dt.strftime("%Y%m%d %H")
    ev["kstart"] = pd.to_datetime(ev["start"]).dt.strftime("%Y%m%d %H")

    def locate(r):  # 사건 끝 시각 -> 시작 시각 순으로 위치를 찾는다
        for k in (r.kend, r.kstart):
            if (r.vessel, k) in pos.index:
                return pos.loc[(r.vessel, k)].tolist()
        return [np.nan, np.nan]
    ev[["lat", "lon"]] = pd.DataFrame([locate(r) for r in ev.itertuples()], index=ev.index)
    located = ev.dropna(subset=["lat"])
    cells = grid_cells(located[located.level == "경보"])

    # 사고 지점
    got = pos.reindex(list(zip(gr.rfid_id, gr.사고시각)))
    gr["lat"], gr["lon"] = got["lat"].values, got["lon"].values
    current = [[round(r.lat, 4), round(r.lon, 4), mask(r.rfid_id), r.업종, (r.사건종류 if isinstance(r.사건종류, str) else "사건 없음"),
                r.사고시각, f"점수 {r.이상항적점수:.2f}", round(float(r.모항거리_km), 1)]
               for r in gr.dropna(subset=["lat"]).itertuples()]

    # 선박별 24시간 점수열 (xlsx 테스트B 시트)
    def pts(row):
        out = []
        pairs = [("사고시각", "이상항적 점수", "사고시각")] + [(f"시각{i}", f"시각{i} 이상항적 점수", "이전") for i in range(1, 25)]
        for tc, sc_, lv in pairs:
            if tc in row.index and isinstance(row[tc], str) and pd.notna(row[sc_]):
                t = (pd.to_datetime(row[tc], format="%Y%m%d %H") - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
                out.append([int(t), round(float(row[sc_]), 4), lv])
        return sorted(out)
    series = {str(r["rfid_id"]): pts(r) for _, r in B.iterrows()}
    nev = ev.groupby("vessel").agg(events=("event_id", "size"), alarm=("level", lambda s: int((s == "경보").sum())))
    vessels = []
    for r in gr.sort_values(["이상항적점수", "final_score"], ascending=False).head(TOP_B).itertuples():
        n = nev.reindex([r.rfid_id]).fillna(0).iloc[0]
        why = r.사건종류 if isinstance(r.사건종류, str) else (r.주된사유 if isinstance(r.주된사유, str) else "사건 없음")
        vessels.append(dict(vessel=mask(r.rfid_id), code=r.업종, alarm_events=int(n.alarm), events=int(n.events),
                            max_score=round(float(r.이상항적점수), 4), model_score=round(float(r.final_score), 4),
                            kind=why, reason=why, last=r.사고시각, per1000=0, accident=r.사고시각, code_src=r.업종근거,
                            rows=series.get(r.rfid_id, []), spans=[]))
    sc_sub, sc_mod = gr.이상항적점수.round(4), gr.final_score.round(4)
    diff = int((~np.isclose(sc_sub, sc_mod, atol=1e-3)).sum())
    at_floor = int((sc_sub == 0.9505).sum())
    try:
        th = json.loads((D / "final_thresholds.json").read_text(encoding="utf-8"))
    except Exception:
        th = dict(caution=0.9506, alarm=0.999, alarm_budget=20, alarm_per_day=0, loss_h=6, loss_km=5, stop_h=3, stop_km=5, persist_h=3, recalls={})
    span_days = max((pd.to_datetime(ev.end).max() - pd.to_datetime(ev.start).min()).days, 1)
    a_empty = bool(A.iloc[:, 1:].isna().all().all())
    n_ev_acc = int(gr.사건종류.notna().sum())
    n_hi = int((sc_sub >= th["caution"]).sum())
    comp = [
        ["테스트A 조업형태 예측 작성", f"{len(A):,}척 모두", "비어 있음" if a_empty else "작성됨", "미작성" if a_empty else "충족"],
        ["제출 점수 = 모델 점수(final_score)", "-", f"{N - diff:,}/{N:,} 일치 (사건 규칙을 반영하면 다를 수 있음)", "참고"],
        ["제출 점수가 0.9505로 같은 행", "-", f"{at_floor:,}/{N:,}", "확인 필요" if at_floor > N * 0.1 else "참고"],
        ["사고 시각에 사건이 잡힌 선박", "-", f"{n_ev_acc:,}/{N:,}", "참고"],
        ["사고 시각 점수 ≥ 주의 기준", "-", f"{n_hi:,}/{N:,}", "참고"],
        ["업종을 학습 데이터가 아닌 분류 모델·추정으로 정한 선박", "-", f"{int((gr.업종근거 != '학습 데이터 업종').sum()):,}/{N:,}", "참고"],
    ]
    n_al = int((ev.level == "경보").sum())
    kpi = dict(events=len(ev), alarm_events=n_al, vessels=N, alarm_vessels=int(ev[ev.level == "경보"].vessel.nunique()),
               per_day=round(n_al / span_days, 2), data_start=ev.start.min()[:10], data_end=ev.end.max()[:10])
    kpi_list = [[f"{N:,}척", "테스트 선박"], [f"{len(ev):,}건", f"사건 (경보 {n_al:,})"],
                [f"{n_ev_acc:,}/{N:,}", "사고 시각에 사건이 잡힌 선박"], [f"{n_hi:,}/{N:,}", "사고 시각 점수 ≥ 주의 기준"],
                [f"{kpi['data_start']} ~ {kpi['data_end']}", "테스트 기간"]]
    return dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), mode="testB", thresholds=dict(
        caution=th["caution"], alarm=th["alarm"], budget=th["alarm_budget"], per_day=kpi["per_day"], loss_h=th["loss_h"],
        loss_km=th["loss_km"], stop_h=th["stop_h"], stop_km=th["stop_km"], persist_h=th["persist_h"], recalls=th.get("recalls", {})),
        confusion=None, cm_info=dict(a_empty=a_empty, codes=gr.업종.value_counts().to_dict(), src=gr.업종근거.value_counts().to_dict()),
        cells=cells, cell=CELL, current=current, vessels=vessels, kpi=kpi, kpiList=kpi_list, completion=comp,
        data_end_ms=int(ms(ev.end).max()) + 3600_000, ev_total=len(ev), ev_located=len(located),
        score_diff=diff, at_floor=at_floor)


def write(anon=False, out=None):
    out = Path(out) if out else D          # --out: 공개용 폴더 등 다른 곳에 저장
    out.mkdir(parents=True, exist_ok=True)
    land = (HERE / "korea_land.json").read_text(encoding="utf-8")
    tpl = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
    jobs = [(build, "dashboard")]
    if testB_files():
        jobs.append((build_testB, "dashboard_testB"))
    for fn, name in jobs:
        data = fn(anon)
        data["dataFile"] = f"{name}_data.json"
        data["hasTestB"] = len(jobs) > 1
        js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        (out / f"{name}_data.json").write_text(js, encoding="utf-8")
        (out / f"{name}.html").write_text(tpl.replace("__LAND__", land).replace("__DATA__", js), encoding="utf-8")
        print(f"[{data['generated']}] {name}.html 생성 (선박 {len(data['vessels'])}척, 격자 {len(data['cells'])}곳)", flush=True)


def mtimes():
    fs = [D / f for f in INPUTS] + list(FULL_FILES.values()) + list(STEP9_FILES.values())
    return [f.stat().st_mtime if f.exists() else 0 for f in fs]


def serve(port, anon):
    last = mtimes()

    def watch():
        nonlocal last
        while True:
            time.sleep(5)
            try:
                cur = mtimes()
                if cur != last:
                    time.sleep(3)  # 파일 쓰기가 끝나길 잠깐 기다림
                    write(anon)
                    last = mtimes()
            except Exception as e:  # 쓰는 중에 읽다 실패해도 다음 주기에 재시도
                print("재생성 실패:", e, flush=True)
    threading.Thread(target=watch, daemon=True).start()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as srv:
        rel = (D / "dashboard.html").relative_to(ROOT).as_posix()
        print(f"http://127.0.0.1:{port}/{rel}  (Ctrl+C로 종료)", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--anon", action="store_true")
    ap.add_argument("--out", default=None, help="저장 폴더 (기본: 이상탐지 결과 폴더). 공개용: --anon --out ../navis-dashboard")
    a = ap.parse_args()
    write(a.anon, a.out)
    if a.serve:
        serve(a.port, a.anon)
