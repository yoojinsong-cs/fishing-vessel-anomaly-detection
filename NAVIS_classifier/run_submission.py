"""
NAVIS 제출 파일 만들기 (testA: 조업형태 분류 / testB: 시각별 이상항적 점수)

  python run_submission.py                      ← 데이터를 ./data 폴더에 넣어 두면 인자 없이 실행
  python run_submission.py --data ./data --out results     (--out 생략 시에도 results 에 저장, 기존 파일은 최신 결과로 덮어씀)

  --data        : 원본 zip/csv/폴더/와일드카드 (월별 V-Pass csv + 테스트 xlsx가 들어 있는 대회 zip 그대로 가능)
                  생략하면 환경변수 NAVIS_DATA → ./data → 코드 옆 data 폴더 → 현재 폴더 순으로 찾음
  --test-xlsx   : 테스트 대상 엑셀을 따로 지정할 때 (생략하면 --data 안에서 찾음)
  --sample-vessels : 빠른 테스트용 (월별 학습 선박 수 제한)

결과
  submission.xlsx                     ← 제출 파일 (testA, testB 시트)
  classifier_report.json, classifier_confusion_matrix.csv, classifier_feature_importance.csv
  hourly_validation_report.json       ← 이상탐지 검증 (AUC, 사고시각 적중률, 임계값)
  testB_hourly_scores.csv             ← testB 선박 시간별 점수 + 이상 사유 (긴 형식)
  eda_hourly_stop_move.csv/png
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from navis_anomaly import eda
from navis_anomaly.classifier import VesselTypeClassifier, vessel_features
from navis_anomaly.config import HOURLY_FEATURES, HOURLY_RISK, Config
from navis_anomaly.hourly_raw import build_hourly
from navis_anomaly.io_utils import prepare_features
from navis_anomaly.match_accidents import accident_hours, candidates_in_month, choose_matches
from navis_anomaly.pipeline import NavisAnomalyDetector, plot_validation, split_train_val
from navis_anomaly.raw_sources import default_data_paths, iter_raw_sources, iter_xlsx, read_source
from navis_anomaly.test_targets import load_targets
from navis_anomaly.vpass_raw import RawConfig, hourly_summary


def forward_score(df: pd.DataFrame, col="anomaly_score") -> np.ndarray:
    """
    제출용 시각별 점수 = (해당 시간 점수 + 다음 시간 점수) / 2  (다음 시간 기록이 없으면 해당 시간 점수)
    사고는 보통 그 시간 중간에 일어나서 '다음 시간'에 이상이 가장 뚜렷해지므로,
    한 시간 앞을 같이 봐야 점수 최고 시각이 실제 사고 시각에 맞춰짐 (인위적 사고 검증으로 확인)
    """
    out = np.empty(len(df))
    key = df["case_id"].where(df["case_id"].astype(str) != "", df["vessel_id"].astype(str)) \
        if "case_id" in df.columns else df["vessel_id"].astype(str)
    for _, idx in df.groupby(key).indices.items():
        g = df.iloc[idx]
        s = pd.Series(g[col].values, index=g["H"].values)
        nxt = s.reindex(s.index + 1).values
        out[idx] = np.where(np.isnan(nxt), s.values, (s.values + nxt) / 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=None,
                    help="원본 zip/csv/폴더/와일드카드. 생략하면 환경변수 NAVIS_DATA → ./data → 현재 폴더 순으로 찾음")
    ap.add_argument("--test-xlsx", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"),
                    help="결과 저장 폴더 (기본: 이 파일 옆 results — 실행할 때마다 최신 결과로 덮어씀)")
    ap.add_argument("--train-hour-rate", type=float, default=0.2, help="학습용 정상 창 샘플링 비율")
    ap.add_argument("--synth-rate", type=float, default=0.15, help="선박별 인위적 사고 생성 확률")
    ap.add_argument("--sample-vessels", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--match-km", type=float, default=10.0, help="사고 위치 매칭 최대 반경(km)")
    args = ap.parse_args()
    if not args.data:
        args.data = default_data_paths()
        if not args.data:
            ap.error("원본 데이터를 찾지 못함: --data 로 경로를 주거나 환경변수 NAVIS_DATA 를 설정하거나 ./data 폴더에 넣어 주세요")
        print(f"[데이터] --data 생략 → {args.data[0]} 사용")
    T0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = RawConfig(synth_rate=args.synth_rate)

    # ---------------- 0. 테스트 대상 ----------------
    xl = [(args.test_xlsx, lambda: open(args.test_xlsx, "rb"))] if args.test_xlsx else list(iter_xlsx(args.data))
    targets = load_targets(xl)
    sources = list(iter_raw_sources(args.data))
    print(f"[데이터] 원본 csv {len(sources)}개:", *[lab for lab, _ in sources], sep="\n  ")

    # 사고 목록(rfid 없음)이면: 1차로 원본을 훑어 사고 위치·시각에 있던 선박을 찾음
    matches = None
    if targets["accidents"] is not None:
        acc = accident_hours(targets["accidents"])
        cands = []
        for label, opener in sources:
            t0 = time.time()
            df = read_source(opener)
            c = candidates_in_month(df, acc, max_km=args.match_km)
            del df
            cands.append(c)
            print(f"[매칭] {label}: 후보 {len(c):,}건 | {time.time() - t0:.0f}s")
        matches = choose_matches(acc, pd.concat(cands, ignore_index=True))
        matches.to_csv(out / "accident_vessel_match.csv", index=False, encoding="utf-8-sig")
        ok = matches["rfid_id"].notna()
        print(f"[매칭] 사고 {len(matches)}건 중 {int(ok.sum())}건 선박 매칭 "
              f"(어업면허 일치 {int(matches['license_match'].fillna(False).sum())}건, "
              f"거리 중앙값 {matches['match_dist_km'].median():.2f}km, 1km 안 후보 2척 이상 {int((matches['n_within_1km'] >= 2).sum())}건)")
        mm = matches[ok]
        targets["A"] = mm[["rfid_id"]].drop_duplicates().reset_index(drop=True)
        targets["B"] = pd.DataFrame({"rfid_id": mm["rfid_id"].values,
                                     "target_date": pd.to_datetime(mm["acc_time"]).dt.normalize().values})
        targets["day_window"] = 0          # 사고 당일 시각만
    idsA = set(targets["A"]["rfid_id"]) if targets["A"] is not None else set()
    idsB = set(targets["B"]["rfid_id"]) if targets["B"] is not None else set()
    test_ids = idsA | idsB
    print(f"[테스트] testA 선박 {len(idsA):,}척 | testB 선박 {len(idsB):,}척 (학습에서 제외)")

    # ---------------- 1. 월별 변환 ----------------
    Ws, HSs, test_raw = [], [], []
    for mi, (label, opener) in enumerate(sources):
        t0 = time.time()
        df = read_source(opener)
        df["rfid_id"] = df["rfid_id"].astype(str)
        is_test = df["rfid_id"].isin(test_ids)
        test_raw.append(df[is_test])
        tr = df[~is_test]
        del df
        if "fishing_code" not in tr.columns or tr["fishing_code"].isna().all():
            print(f"[건너뜀] 업종 정보 없음: {label}")
            continue
        tr = tr[tr["fishing_code"].notna()]
        if args.sample_vessels:
            v = tr["rfid_id"].unique()
            keep = np.random.default_rng(mi).choice(v, min(args.sample_vessels, len(v)), replace=False)
            tr = tr[tr["rfid_id"].isin(keep)]
        W, _, HS = build_hourly(tr, rc, "train", args.train_hour_rate, args.n_jobs, seed=mi)
        del tr
        W["month"] = mi
        HS["month"] = mi
        Ws.append(W)
        HSs.append(HS)
        print(f"[변환] {label}: 시간 {len(HS):,} → 학습 창 {int((W.is_synthetic == 0).sum()):,} "
              f"+ 인위적 사고 창 {int(W.is_synthetic.sum()):,} ({W.loc[W.is_synthetic == 1, 'case_id'].nunique()}건) "
              f"| {time.time() - t0:.0f}s")
    W = pd.concat(Ws, ignore_index=True)
    HS = pd.concat(HSs, ignore_index=True)
    del Ws, HSs

    # 시간대별 EDA
    hourly = hourly_summary(HS[["reg_type", "hour", "h_speed"]].rename(columns={"h_speed": "speed_kn"}), rc.stop_kn)
    hourly.to_csv(out / "eda_hourly_stop_move.csv", index=False, encoding="utf-8-sig")
    eda.plot_hourly(hourly, Config(), out)

    # ---------------- 2. 분류모델 (testA) ----------------
    print("\n[분류] 선박×월 단위 특징 생성")
    X = vessel_features(HS, ["vessel_id", "month"])
    y = HS.groupby(["vessel_id", "month"])["reg_type"].first().reindex(X.index)
    groups = pd.Series(X.index.get_level_values("vessel_id"), index=X.index)
    clf = VesselTypeClassifier()
    rep = clf.validate(X.reset_index(drop=True), y.reset_index(drop=True), groups.reset_index(drop=True), out)
    a = rep["a_unseen_vessels_oof"]
    print(f"[분류] {rep['model']} | 검증 선박 {a['n_vessels']:,}척 | 정확도 {a['accuracy(=micro_F1)']:.3f} | macro F1 {a['macro_F1']:.3f}")
    clf.fit(X.reset_index(drop=True), y.reset_index(drop=True))
    clf.save_report(out)
    del HS

    # ---------------- 3. 이상탐지 (시간 단위) ----------------
    cfg = Config(features=list(HOURLY_FEATURES), risk_features=dict(HOURLY_RISK), w_ae=0.0, combine="mean")
    cfg.passthrough = cfg.passthrough + ["case_id", "case_k", "H", "hour"]
    feat = prepare_features(W, cfg)
    split = split_train_val(feat, cfg)
    det = NavisAnomalyDetector(cfg).fit(feat[split == "train"])
    sc = det.score(feat)
    sc["split"] = split.values
    val = sc[sc["split"] == "val"]
    det.set_thresholds(val.loc[val["is_synthetic"] == 0, "anomaly_score"].values)
    sc["level"] = det.apply_level(sc["anomaly_score"])
    hrep = evaluate_hourly(sc, det)
    with open(out / "hourly_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(hrep, f, ensure_ascii=False, indent=2, default=str)
    det.baseline.table().to_csv(out / "normal_baseline_hourly.csv", index=False, encoding="utf-8-sig")
    det.save(out / "hourly_detector.joblib")
    del W, feat

    # ---------------- 4. 테스트 데이터 점수 ----------------
    raw = pd.concat(test_raw, ignore_index=True)
    print(f"\n[테스트] 원본 {len(raw):,}행 (선박 {raw['rfid_id'].nunique():,}척)")
    Wt, _, HSt = build_hourly(raw, rc, "test", n_jobs=args.n_jobs)
    del raw

    # testA: 분류
    found = set(HSt["vessel_id"].unique())
    sheetA = None
    if idsA:
        XA = vessel_features(HSt[HSt["vessel_id"].isin(idsA)], ["vessel_id"])
        PA = clf.predict(XA)
        sheetA = targets["A"][["rfid_id"]].drop_duplicates().copy()
        sheetA = sheetA.merge(PA[["pred_type", "pred_prob"]], left_on="rfid_id", right_index=True, how="left")
        miss = sheetA["pred_type"].isna().sum()
        if miss:
            print(f"[알림] testA 중 항적이 없는 선박 {miss}척 → 학습 데이터 최다 업종으로 채움")
            sheetA["pred_type"] = sheetA["pred_type"].fillna(y.value_counts().idxmax())
            sheetA["pred_prob"] = sheetA["pred_prob"].fillna(0.0)
        sheetA = sheetA.rename(columns={"pred_type": "예측 조업형태", "pred_prob": "예측조업형태 확률"})
        sheetA["예측조업형태 확률"] = sheetA["예측조업형태 확률"].round(4)
        sheetA = _id_numeric(sheetA)

    # testB: 시각별 이상점수
    sheetB, longB = None, None
    if idsB:
        wb = Wt[Wt["vessel_id"].isin(idsB)].copy()
        # 업종: 원본 등록 업종이 있으면 사용, 없으면 분류모델 예측 업종
        pb = clf.predict(vessel_features(HSt[HSt["vessel_id"].isin(idsB)], ["vessel_id"]))
        wb = wb.merge(pb[["pred_type", "pred_prob"]], left_on="vessel_id", right_index=True, how="left")
        unknown = wb["reg_type"].isin(["미상", "nan", ""]) | wb["reg_type"].isna()
        wb.loc[unknown, "reg_type"] = wb.loc[unknown, "pred_type"]
        preds = pb[["pred_type", "pred_prob"]].reset_index().rename(columns={"index": "vessel_id"})
        preds["vessel_id"] = preds["vessel_id"].astype(str)
        featB = prepare_features(wb.drop(columns=["pred_type", "pred_prob"]), cfg)
        sB = det.score(featB, preds=preds, pred_key="vessel_id")
        sB["level"] = det.apply_level(sB["anomaly_score"])
        sB = sB.sort_values(["vessel_id", "H"]).reset_index(drop=True)
        sB["case_id"] = ""
        sB["score_submit"] = forward_score(sB)
        sB["시각"] = sB["date"].dt.strftime("%Y%m%d") + " " + sB["hour"].astype(int).map("{:02d}".format)
        sB = restrict_to_target_dates(sB, targets["B"], targets.get("day_window", 1))
        longB = sB
        sheetB = make_sheetB(sB, targets["B"])
        sB.to_csv(out / "testB_hourly_scores.csv", index=False, encoding="utf-8-sig")

    # ---------------- 4-1. 실제 사고 기준 검증 ----------------
    real_eval = None
    if matches is not None and longB is not None:
        real_eval = evaluate_real_accidents(matches, longB, sheetA)
        pd.DataFrame(real_eval["_detail"]).to_csv(out / "real_accident_detail.csv", index=False, encoding="utf-8-sig")
        with open(out / "real_accident_eval.json", "w", encoding="utf-8") as f:
            json.dump(real_eval, f, ensure_ascii=False, indent=2, default=str)

    # ---------------- 5. 제출 파일 ----------------
    with pd.ExcelWriter(out / "submission.xlsx", engine="openpyxl") as xw:
        if sheetA is not None:
            sheetA.to_excel(xw, sheet_name="testA", index=False)
        if sheetB is not None:
            sheetB.to_excel(xw, sheet_name="testB", index=False)

    print("\n========== 요약 ==========")
    print(f"[분류] {rep['model']} 정확도 {a['accuracy(=micro_F1)']:.3f} | macro F1 {a['macro_F1']:.3f}")
    print(f"[이상탐지] 임계값 주의 ≥ {det.thresholds['caution']:.3f} | 경보 ≥ {det.thresholds['alert']:.3f}")
    print(f"[이상탐지] 사고 이후 시간 vs 정상 ROC-AUC {hrep['window_auc']}")
    for k, v in hrep["accident_time"].items():
        print(f"  [{k}] {v['n']}건 | 사고시각 정확히 맞힘 {v['exact']:.1%} | ±1시간 {v['within_1h']:.1%} | 주의 이상 감지 {v['flagged']:.1%}")
    if sheetA is not None:
        print(f"[testA] {len(sheetA):,}척 예측 분포: {sheetA['예측 조업형태'].value_counts().to_dict()}")
    if sheetB is not None:
        print(f"[testB] {len(sheetB):,}척, 선박당 시각 수 중앙값 {longB.groupby('vessel_id').size().median():.0f}")
    if real_eval:
        print(f"[실제 사고 검증] 매칭 {real_eval['n_matched']}건 | 조업형태 정답률(어업면허 기준) {real_eval['testA_accuracy']}"
              f" | 사고시각 정확히 {real_eval['hour_exact']} · ±1시간 {real_eval['hour_within_1h']} · ±2시간 {real_eval['hour_within_2h']}")
    print(f"제출 파일 → {out / 'submission.xlsx'} | 총 {time.time() - T0:.0f}s")


def evaluate_hourly(sc: pd.DataFrame, det) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    t = det.thresholds
    val = sc[sc["split"] == "val"]
    neg = val[val["is_synthetic"] == 0]
    syn = sc[sc["is_synthetic"] == 1].copy()
    pos = syn[((syn["anomaly_type"] == "drift") & (syn["H"] >= syn["case_k"])) |
              ((syn["anomaly_type"] == "signal_loss") & (syn["H"] == syn["case_k"]))]
    rep = {"thresholds": t, "n_val_normal_windows": int(len(neg)), "n_cases": int(syn["case_id"].nunique()),
           "false_alarm_rate": {"주의이상": float((neg["anomaly_score"] >= t["caution"]).mean()),
                                "경보": float((neg["anomaly_score"] >= t["alert"]).mean())}}
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    rep["window_auc"] = {c: round(float(roc_auc_score(y, np.r_[neg[c], pos[c]])), 4)
                         for c in ["s_if", "s_range", "anomaly_score"] if len(pos)}
    rep["window_pr_auc"] = round(float(average_precision_score(y, np.r_[neg["anomaly_score"], pos["anomaly_score"]])), 4) if len(pos) else None
    syn = syn.sort_values(["case_id", "H"]).reset_index(drop=True)
    syn["score_submit"] = forward_score(syn)
    rows = []
    for cid, g in syn.groupby("case_id"):
        k = g["case_k"].iloc[0]
        h = g.loc[g["score_submit"].idxmax(), "H"]
        rows.append((g["anomaly_type"].iloc[0], h - k, g["score_submit"].max() >= t["caution"]))
    r = pd.DataFrame(rows, columns=["type", "off", "flag"])
    rep["accident_time"] = {
        k: {"n": int(len(g)), "exact": float((g["off"] == 0).mean()), "within_1h": float((g["off"].abs() <= 1).mean()),
            "flagged": float(g["flag"].mean())} for k, g in r.groupby("type")}
    rep["accident_time"]["전체"] = {"n": int(len(r)), "exact": float((r["off"] == 0).mean()),
                                  "within_1h": float((r["off"].abs() <= 1).mean()), "flagged": float(r["flag"].mean())}
    return rep


def evaluate_real_accidents(matches: pd.DataFrame, longB: pd.DataFrame, sheetA) -> dict:
    """테스트 엑셀의 실제 발생일시·어업면허와 비교"""
    m = matches[matches["rfid_id"].notna()].copy()
    rows = []
    for r in m.itertuples():
        day = pd.Timestamp(r.acc_time).normalize()
        g = longB[(longB["vessel_id"] == r.rfid_id) & (longB["date"] == day)]
        if g.empty:
            rows.append((r.acc_id, np.nan, np.nan))
            continue
        top = g.loc[g["score_submit"].idxmax()]
        rows.append((r.acc_id, int(top["hour"]), float(top["score_submit"])))
    e = pd.DataFrame(rows, columns=["acc_id", "pred_hour", "pred_score"])
    m = m.merge(e, on="acc_id", how="left")
    m["true_hour"] = pd.to_datetime(m["acc_time"]).dt.hour
    m["hour_diff"] = (m["pred_hour"] - m["true_hour"]).abs()
    ev = m.dropna(subset=["pred_hour"])
    res = {"n_accidents": int(len(matches)), "n_matched": int(len(m)), "n_scored": int(len(ev)),
           "hour_exact": round(float((ev["hour_diff"] == 0).mean()), 3) if len(ev) else None,
           "hour_within_1h": round(float((ev["hour_diff"] <= 1).mean()), 3) if len(ev) else None,
           "hour_within_2h": round(float((ev["hour_diff"] <= 2).mean()), 3) if len(ev) else None}
    if sheetA is not None and "license" in m.columns:
        a = sheetA.assign(rfid_id=sheetA["rfid_id"].astype(str)).set_index("rfid_id")["예측 조업형태"]
        m["pred_type"] = m["rfid_id"].astype(str).map(a)
        ok = m["license"].isin(a.unique()) | m["license"].isin(["연안복합", "자망", "통발", "채낚기연승낚시"])
        res["testA_accuracy"] = round(float((m.loc[ok, "pred_type"] == m.loc[ok, "license"]).mean()), 3) if ok.any() else None
        res["license_dist"] = m["license"].value_counts().to_dict()
    if "원본_발생유형" in m.columns:
        res["hour_within_1h_by_type"] = ev.assign(t=ev["원본_발생유형"]).groupby("t")["hour_diff"] \
            .apply(lambda v: f"{(v <= 1).mean():.0%} (n={len(v)})").to_dict()
    res["_detail"] = m[["acc_id", "acc_time", "license", "rfid_id", "match_dist_km", "license_match",
                        "pred_hour", "true_hour", "hour_diff", "pred_score"]].to_dict("records")
    return res


def restrict_to_target_dates(sB: pd.DataFrame, tB: pd.DataFrame, day_window: int = 1) -> pd.DataFrame:
    """테스트 엑셀에 사고 날짜가 있으면 그 날짜 전후 day_window일 안의 시각만 사용"""
    if tB is None or "target_date" not in tB.columns or tB["target_date"].isna().all():
        return sB
    d = tB.dropna(subset=["target_date"]).groupby("rfid_id")["target_date"].apply(list).to_dict()
    keep = np.ones(len(sB), bool)
    for i, (v, dt) in enumerate(zip(sB["vessel_id"].values, sB["date"].values)):
        if v in d:
            keep[i] = any(abs((pd.Timestamp(dt).normalize() - pd.Timestamp(x)).days) <= day_window for x in d[v])
    return sB[keep].reset_index(drop=True)


def make_sheetB(sB: pd.DataFrame, tB: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = tB["rfid_id"].drop_duplicates().tolist()
    groups = dict(tuple(sB.groupby("vessel_id")))
    width = 0
    for v in order:
        g = groups.get(v)
        r = {"rfid_id": v}
        if g is None or g.empty:
            r.update({"사고시각": "", "이상항적 점수": np.nan})
            rows.append((r, []))
            continue
        g = g.sort_values("H")
        top = g.loc[g["score_submit"].idxmax()]
        r.update({"사고시각": top["시각"], "이상항적 점수": round(float(top["score_submit"]), 4)})
        pairs = list(zip(g["시각"], g["score_submit"].round(4)))
        width = max(width, len(pairs))
        rows.append((r, pairs))
    recs = []
    for r, pairs in rows:
        for i, (tm, s) in enumerate(pairs, 1):
            r[f"시각{i}"] = tm
            r[f"시각{i} 이상항적 점수"] = s
        recs.append(r)
    cols = ["rfid_id", "사고시각", "이상항적 점수"] + [c for i in range(1, width + 1) for c in (f"시각{i}", f"시각{i} 이상항적 점수")]
    out = pd.DataFrame(recs).reindex(columns=cols)
    return _id_numeric(out)


def _id_numeric(d: pd.DataFrame) -> pd.DataFrame:
    """rfid_id가 전부 숫자면 숫자로 저장 (예시 형식과 동일)"""
    s = d["rfid_id"].astype(str)
    if s.str.fullmatch(r"\d+").all():
        d["rfid_id"] = s.astype("int64")
    return d


if __name__ == "__main__":
    main()
