"""
이상탐지 전체 흐름
  정상 기준(업종×시간대) → Isolation Forest → 시퀀스 오토인코더 → 업종 불일치
  → 0~1 이상점수 → 주의/경보 2단계 임계값 → 인위적 이상 궤적으로 검증
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .baseline import GroupBaseline, top_reasons
from .config import Config
from .models import GroupErrorNormalizer, IFDetector, ScoreCalibrator, SeqAutoencoder

LEVELS = ["정상", "주의", "경보"]


# ---------------------------------------------------------------------------
# 학습/검증 분리
# ---------------------------------------------------------------------------
def split_train_val(feat: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    반환: 'train' / 'val' 표시
      - 인위적 이상 궤적은 학습에 절대 쓰지 않고 전부 검증에 사용
      - 날짜가 있으면 기간 분리 (앞 80% 기간 학습, 뒤 20% 검증)
      - 날짜가 없으면 선박 단위 분리 (같은 배가 학습·검증에 동시에 들어가지 않게)
    """
    c = cfg.cols
    split = pd.Series("train", index=feat.index)
    normal = feat[c.is_synthetic] == 0
    if c.date and c.date in feat.columns and feat.loc[normal, c.date].notna().sum() > 0:
        dates = np.sort(feat.loc[normal, c.date].dropna().unique())
        cut = dates[int(len(dates) * (1 - cfg.val_ratio))] if len(dates) > 1 else dates[-1]
        split[normal & (feat[c.date] >= cut)] = "val"
        print(f"[분할] 기간 분리: {pd.Timestamp(cut).date()} 이후 = 검증")
    else:
        idx = feat.index[normal]
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_ratio, random_state=cfg.random_state)
        _, va = next(gss.split(idx, groups=feat.loc[idx, c.vessel_id]))
        split[idx[va]] = "val"
        print("[분할] 선박 단위 분리")
    split[~normal] = "val"
    return split


# ---------------------------------------------------------------------------
# 이상탐지기
# ---------------------------------------------------------------------------
class NavisAnomalyDetector:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.baseline = GroupBaseline(self.cfg)
        self.iforest = IFDetector(self.cfg)
        self.range_cal = ScoreCalibrator()
        self.ae: Optional[SeqAutoencoder] = None
        self.ae_norm: Optional[GroupErrorNormalizer] = None
        self.ae_cal: Optional[ScoreCalibrator] = None
        self.thresholds = {}

    # ---- 학습 ----
    def fit(self, feat_train: pd.DataFrame, X_seq=None, seq_ids=None):
        c, cfg = self.cfg.cols, self.cfg
        print(f"[학습] 정상 궤적 {len(feat_train):,}개")

        # 1) 조건부 정상 기준
        self.baseline.fit(feat_train)
        Z, _, _ = self.baseline.transform(feat_train)

        # 2) Isolation Forest
        self.iforest.fit(Z)
        self.range_cal.fit(self._range_raw(feat_train))
        print("[학습] 정상범위 기준 + Isolation Forest 완료")

        # 3) 시퀀스 오토인코더
        if X_seq is not None and len(X_seq):
            pos = pd.Series(np.arange(len(seq_ids)), index=pd.Index(seq_ids).astype(str))
            ids = feat_train[c.unit_id].astype(str)
            has = ids.isin(pos.index)
            if has.sum() >= 50:
                tr = feat_train[has]
                Xtr = X_seq[pos[tr[c.unit_id].astype(str)].values]
                self.ae = SeqAutoencoder(cfg)
                print(f"[학습] 시퀀스 오토인코더 ({self.ae.backend}) — 시퀀스 {len(Xtr):,}개")
                self.ae.fit(Xtr)
                le = np.log(self.ae.recon_error(Xtr) + 1e-9)
                self.ae_norm = GroupErrorNormalizer(cfg).fit(le, tr[c.reg_type].values, tr[c.time_band].values)
                z = self.ae_norm.transform(le, tr[c.reg_type].values, tr[c.time_band].values)
                self.ae_cal = ScoreCalibrator().fit(z)
            else:
                print("[알림] 특징 테이블과 매칭되는 시퀀스가 50개 미만 → 오토인코더 생략")
        return self

    def _range_raw(self, feat, P=None):
        """정상범위 이탈 raw 점수 = -log10(가장 드문 특징의 꼬리확률)"""
        P = self.baseline.tail_probs(feat) if P is None else P
        cols = [f for f in self.cfg.risk_features if f in P.columns] if self.cfg.range_mode == "risk" else list(P.columns)
        if not cols:
            cols = list(P.columns)
        return -np.log10(P[cols].min(axis=1).to_numpy())

    # ---- 점수 ----
    def score(self, feat: pd.DataFrame, X_seq=None, seq_ids=None, preds=None, pred_key=None) -> pd.DataFrame:
        c, cfg = self.cfg.cols, self.cfg
        out = feat[[col for col in [c.vessel_id, c.unit_id, c.date, c.reg_type, c.time_band, c.sea_area,
                                    c.is_synthetic, c.anomaly_type] + cfg.passthrough
                    if col and col in feat.columns]].copy()

        Z, level, V = self.baseline.transform(feat)
        out["baseline_level"] = level
        out["n_out_of_range"] = (V != 0).sum(axis=1)
        out["s_if"] = self.iforest.score(Z)
        P = self.baseline.tail_probs(feat)
        out["s_range"] = self.range_cal.transform(self._range_raw(feat, P))

        out["s_ae"] = np.nan
        if self.ae is not None and X_seq is not None and len(X_seq):
            pos = pd.Series(np.arange(len(seq_ids)), index=pd.Index(seq_ids).astype(str))
            ids = feat[c.unit_id].astype(str)
            has = ids.isin(pos.index).values
            if has.any():
                sub = feat[has]
                le = np.log(self.ae.recon_error(X_seq[pos[ids[has]].values]) + 1e-9)
                z = self.ae_norm.transform(le, sub[c.reg_type].values, sub[c.time_band].values)
                out.loc[has, "s_ae"] = self.ae_cal.transform(z)

        # 업종 불일치 (분류모델 예측 ≠ 등록 업종)
        out[c.pred_type] = np.nan
        out["s_mismatch"] = 0.0
        if preds is not None:
            key = pred_key or c.unit_id
            m = out[[key]].astype(str).merge(preds, on=key, how="left")
            out[c.pred_type] = m[c.pred_type].values
            prob = m[c.pred_prob].fillna(0).astype(float).values
            mism = m[c.pred_type].notna().values & (m[c.pred_type].astype(str).values != out[c.reg_type].astype(str).values)
            out["type_mismatch"] = mism.astype(int)
            out["s_mismatch"] = np.where(mism, prob, 0.0)

        # 점수 결합: (IF, 정상범위 이탈, AE) → max 또는 가중평균
        #          업종 불일치는 '확률적 OR'로 가산 (점수를 올리기만 함)
        w = np.array([cfg.w_if, cfg.w_range, cfg.w_ae], float)
        S = out[["s_if", "s_range", "s_ae"]].to_numpy()
        S = np.where(w[None, :] > 0, S, np.nan)
        avail = ~np.isnan(S)
        if cfg.combine == "max":
            base = np.nanmax(np.where(avail, S, -np.inf), axis=1)
        else:
            base = np.nansum(np.nan_to_num(S) * w, axis=1) / np.clip((avail * w).sum(axis=1), 1e-9, None)
        out["s_base"] = base
        out["anomaly_score"] = 1 - (1 - base) * (1 - cfg.w_mismatch * out["s_mismatch"].to_numpy())
        out["anomaly_score"] = out["anomaly_score"].clip(0, 1)

        out["reasons"] = top_reasons(Z, P, k=cfg.top_k_reasons)
        ae_hi = (out["s_ae"].fillna(0) >= 0.95) & (cfg.w_ae > 0)   # AE를 결합에 쓸 때만 사유로 표시
        out.loc[ae_hi, "reasons"] = "움직임 순서패턴 이상(AE), " + out.loc[ae_hi, "reasons"]
        if "type_mismatch" in out.columns:
            mm = out["type_mismatch"] == 1
            out.loc[mm, "reasons"] = out.loc[mm].apply(
                lambda r: (r["reasons"] + ", " if r["reasons"] else "") + f"업종불일치(예측:{r[c.pred_type]})", axis=1)
        if self.thresholds:
            out["level"] = self.apply_level(out["anomaly_score"])
        self._last_Z = Z
        return out

    # ---- 임계값 ----
    def set_thresholds(self, val_normal_scores: np.ndarray):
        cfg = self.cfg
        cau = float(np.quantile(val_normal_scores, cfg.caution_q))
        ale = float(np.quantile(val_normal_scores, cfg.alert_q))
        if ale <= cau:
            ale = min(1.0, cau + 1e-3)
        self.thresholds = {"caution": cau, "alert": ale,
                           "method": f"검증용 정상 궤적 이상점수의 {cfg.caution_q:.0%}/{cfg.alert_q:.0%} 분위수"}
        return self.thresholds

    def apply_level(self, s: pd.Series) -> pd.Series:
        t = self.thresholds
        return pd.Series(np.select([s >= t["alert"], s >= t["caution"]], ["경보", "주의"], "정상"), index=s.index)

    def save(self, path):
        if self.ae is not None and self.ae.backend == "torch":
            self.ae.model.cpu()
            self.ae.device = "cpu"
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "NavisAnomalyDetector":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------
def evaluate(val: pd.DataFrame, det: NavisAnomalyDetector) -> dict:
    c = det.cfg.cols
    y = val[c.is_synthetic].astype(int).values
    rep = {"n_val_normal": int((y == 0).sum()), "n_val_synthetic": int((y == 1).sum())}
    t = det.thresholds
    normal = val[y == 0]
    rep["false_alarm_rate"] = {
        "주의이상": float((normal["anomaly_score"] >= t["caution"]).mean()),
        "경보": float((normal["anomaly_score"] >= t["alert"]).mean()),
    }
    if y.sum() == 0 or y.sum() == len(y):
        rep["note"] = "인위적 이상 궤적이 없어 AUC·탐지율 계산 생략 (1번 담당에게 받으면 다시 실행)"
        return rep

    rep["auc"] = {}
    for col, name in [("s_if", "IsolationForest"), ("s_range", "정상범위이탈"), ("s_ae", "AutoEncoder"),
                      ("s_base", "결합"), ("anomaly_score", "최종(불일치 포함)")]:
        ok = val[col].notna().values
        if ok.sum() and len(np.unique(y[ok])) == 2:
            rep["auc"][name] = {"ROC_AUC": round(float(roc_auc_score(y[ok], val.loc[ok, col])), 4),
                                "PR_AUC": round(float(average_precision_score(y[ok], val.loc[ok, col])), 4)}

    syn = val[y == 1]
    rep["detection_rate_by_type"] = {}
    for at, g in syn.groupby(c.anomaly_type):
        rep["detection_rate_by_type"][at] = {
            "n": int(len(g)),
            "주의이상_탐지율": round(float((g["anomaly_score"] >= t["caution"]).mean()), 4),
            "경보_탐지율": round(float((g["anomaly_score"] >= t["alert"]).mean()), 4),
        }
    rep["detection_rate_all"] = {
        "주의이상_탐지율": round(float((syn["anomaly_score"] >= t["caution"]).mean()), 4),
        "경보_탐지율": round(float((syn["anomaly_score"] >= t["alert"]).mean()), 4),
    }

    # 참고: F1 최대 임계값 (인위적 이상 비율에 따라 달라지므로 운영 기준으로는 분위수 방식 사용)
    s = val["anomaly_score"].values
    best = (0, 0)
    for th in np.unique(np.round(s, 3)):
        p = s >= th
        tp = (p & (y == 1)).sum()
        if p.sum() == 0:
            continue
        prec, rec = tp / p.sum(), tp / y.sum()
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best[0]:
            best = (f1, th)
    rep["reference_best_f1"] = {"threshold": float(best[1]), "F1": round(float(best[0]), 4)}
    return rep


def vessel_alerts(scores: pd.DataFrame, det: NavisAnomalyDetector) -> pd.DataFrame:
    """선박 단위 경보 목록 (실제 선박만, 인위적 이상 제외)"""
    c = det.cfg.cols
    s = scores[scores[c.is_synthetic] == 0].copy()
    s["lv"] = s["level"].map({k: i for i, k in enumerate(LEVELS)})
    idx = s.groupby(c.vessel_id)["anomaly_score"].idxmax()
    worst = s.loc[idx].set_index(c.vessel_id)
    agg = s.groupby(c.vessel_id).agg(
        reg_type=(c.reg_type, "first"),
        n_units=(c.unit_id, "size"),
        max_score=("anomaly_score", "max"),
        mean_score=("anomaly_score", "mean"),
        n_caution=("lv", lambda v: int((v == 1).sum())),
        n_alert=("lv", lambda v: int((v == 2).sum())),
    )
    agg["level"] = agg["max_score"].pipe(det.apply_level)
    agg["worst_unit"] = worst[c.unit_id]
    if c.date and c.date in worst.columns:
        agg["worst_date"] = worst[c.date]
    agg["worst_time_band"] = worst[c.time_band]
    agg["worst_reasons"] = worst["reasons"]
    agg = agg[agg["level"] != "정상"].sort_values(["max_score", "n_alert"], ascending=False)
    return agg.reset_index()


def plot_validation(val: pd.DataFrame, det: NavisAnomalyDetector, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .eda import _set_korean_font
    _set_korean_font()
    c = det.cfg.cols
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, 41)
    for at, g in val.groupby(c.anomaly_type):
        ax.hist(g["anomaly_score"], bins=bins, alpha=0.55, density=True,
                label=("정상(검증)" if at == "normal" else f"인위적 이상: {at}") + f" n={len(g)}")
    for k, ls, lab, yy in [("caution", "--", "주의", 0.92), ("alert", "-", "경보", 0.80)]:
        ax.axvline(det.thresholds[k], color="k", ls=ls, lw=1)
        ax.text(det.thresholds[k], ax.get_ylim()[1] * yy, f"{lab} {det.thresholds[k]:.3f} ",
                fontsize=9, ha="right", bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.set_xlabel("이상점수 (0~1)")
    ax.set_ylabel("밀도")
    ax.set_title("검증 세트 이상점수 분포와 2단계 임계값")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "validation_score_hist.png", dpi=150)
    plt.close(fig)
