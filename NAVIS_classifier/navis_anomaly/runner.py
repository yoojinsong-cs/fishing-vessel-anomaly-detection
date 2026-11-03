"""학습 → 점수 → 임계값 → 검증 → 저장 (run_anomaly.py, run_vpass.py 공통)"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from . import eda
from .config import Config
from .io_utils import prepare_features
from .pipeline import NavisAnomalyDetector, evaluate, plot_validation, split_train_val, vessel_alerts


def run(feat_raw: pd.DataFrame, cfg: Config, out, X_seq=None, seq_ids=None,
        preds=None, pred_key=None, hourly_eda: pd.DataFrame = None):
    t0 = time.time()
    c = cfg.cols
    out = Path(out)
    (out / "models").mkdir(parents=True, exist_ok=True)

    feat = prepare_features(feat_raw, cfg)
    print(f"[데이터] 특징 테이블 {len(feat):,}행 | 선박 {feat[c.vessel_id].nunique():,}척 | "
          f"인위적 이상 {int(feat[c.is_synthetic].sum()):,}개 | 특징 {len(cfg.features)}개")

    # ---- EDA ----
    if hourly_eda is not None and not hourly_eda.empty:
        hourly_eda.to_csv(out / "eda_hourly_stop_move.csv", index=False, encoding="utf-8-sig")
        eda.plot_hourly(hourly_eda, cfg, out)
    eda.time_band_summary(feat, cfg).to_csv(out / "eda_type_timeband_summary.csv", index=False, encoding="utf-8-sig")

    # ---- 학습 ----
    split = split_train_val(feat, cfg)
    det = NavisAnomalyDetector(cfg).fit(feat[split == "train"], X_seq, seq_ids)
    det.baseline.table().to_csv(out / "normal_baseline.csv", index=False, encoding="utf-8-sig")

    # ---- 점수 + 임계값 ----
    scores = det.score(feat, X_seq, seq_ids, preds, pred_key)
    scores.insert(0, "split", split.values)
    val = scores[scores["split"] == "val"]
    det.set_thresholds(val.loc[val[c.is_synthetic] == 0, "anomaly_score"].values)
    scores["level"] = det.apply_level(scores["anomaly_score"])
    val = scores[scores["split"] == "val"]

    # ---- 검증 ----
    report = evaluate(val, det)
    report["thresholds"] = det.thresholds
    report["features_used"] = cfg.features
    report["ae_backend"] = det.ae.backend if det.ae else None
    report["level_counts_real_units"] = scores.loc[scores[c.is_synthetic] == 0, "level"].value_counts().to_dict()
    report["level_counts_val_real_by_type"] = (
        val[val[c.is_synthetic] == 0].groupby(c.reg_type)["level"].value_counts().unstack(fill_value=0).to_dict("index"))
    plot_validation(val, det, out)

    # ---- 저장 ----
    real = scores[scores[c.is_synthetic] == 0].drop(columns=[c.is_synthetic, c.anomaly_type])
    real.sort_values("anomaly_score", ascending=False).to_csv(out / "anomaly_scores.csv", index=False, encoding="utf-8-sig")
    val.to_csv(out / "validation_scores.csv", index=False, encoding="utf-8-sig")
    alerts = vessel_alerts(scores, det)
    alerts.to_csv(out / "alert_vessels.csv", index=False, encoding="utf-8-sig")
    with open(out / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(det.thresholds, f, ensure_ascii=False, indent=2)
    with open(out / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    Z = det._last_Z.copy()
    Z.insert(0, c.unit_id, feat[c.unit_id].values)
    Z.insert(0, c.vessel_id, feat[c.vessel_id].values)
    Z["anomaly_score"] = scores["anomaly_score"].values
    Z[(feat[c.is_synthetic] == 0).values].to_csv(out / "shap_input_iforest.csv", index=False, encoding="utf-8-sig")
    joblib.dump(det.iforest.model, out / "models" / "iforest.joblib")
    det.save(out / "models" / "navis_anomaly_detector.joblib")

    # ---- 요약 ----
    print("\n========== 결과 요약 ==========")
    print(f"임계값  주의 ≥ {det.thresholds['caution']:.3f} | 경보 ≥ {det.thresholds['alert']:.3f}")
    print(f"오탐률(검증 정상)  주의이상 {report['false_alarm_rate']['주의이상']:.1%} | 경보 {report['false_alarm_rate']['경보']:.1%}")
    if "auc" in report:
        for k, v in report["auc"].items():
            print(f"  {k:<18} ROC-AUC {v['ROC_AUC']:.3f}  PR-AUC {v['PR_AUC']:.3f}")
        for k, v in report["detection_rate_by_type"].items():
            print(f"  [{k}] n={v['n']}  주의이상 탐지 {v['주의이상_탐지율']:.1%} | 경보 탐지 {v['경보_탐지율']:.1%}")
    else:
        print(report.get("note", ""))
    print(f"실제 궤적 등급: {report['level_counts_real_units']}")
    print(f"경보/주의 선박: {len(alerts):,}척  →  {out / 'alert_vessels.csv'}")
    print(f"소요 {time.time() - t0:.1f}s")
    return det, scores, report
