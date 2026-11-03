"""
조업유형(업종) 분류 — 항적만 보고 '이 배가 어떤 업종인지' 예측 (제출 testA)
------------------------------------------------
- 선박별 시간 통계(hourly_raw)를 선박 단위로 묶어 특징을 만든다
    활동 시간대 분포(0~23시 비중), 속력 분포, 직진성·침로변화·정지 비율, 활동 해역(위경도 중앙값·범위) 등
- 학습 표본: 선박 × 월 (같은 배의 다른 달은 검증에 같이 들어가지 않게 '선박 단위'로 분리)
- 모델: LightGBM (없으면 sklearn HistGradientBoosting)
- 평가: 정확도(= 단일 라벨 micro-F1, 대회 지표), macro F1, 혼동행렬

[개선] (원본 대비)
  1) class_weight 기본값 None — 대회 지표가 micro-F1(=정확도)이라 'balanced'는 다수 업종을 덜 맞혀 손해
  2) 위치 특징 추가 — 정박 위치(모항), 활동 해역 범위(p10·p90), 모항에서 떨어진 거리 분포, 하루 출항 수
  3) 검증 두 가지 — (a) 처음 보는 선박: 선박 단위 K-fold OOF
                    (b) 같은 선박·다른 기간: 앞 달로 학습 → 뒤 달 예측 (테스트가 '학습 선박의 다른 해' 이므로)
  4) 예측은 선박 단위로 합침 — 선박 × 월 확률을 활동 시간 수로 가중 평균 → 선박당 1줄
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .vpass_raw import _hav

STOP_KN = 0.3          # 정박으로 보는 시간 속력
MOVE_KN = 1.0


def vessel_features(HS: pd.DataFrame, keys=("vessel_id",)) -> pd.DataFrame:
    """시간별 통계 → 선박(×기간) 단위 특징"""
    keys = list(keys)
    d = HS.reset_index(drop=True)          # 여러 파일을 이어 붙인 HS의 중복 인덱스 방지
    d["moving"] = (d["h_speed"] >= MOVE_KN).astype(float)
    d["fast"] = (d["h_speed"] >= 6.0).astype(float)
    d["slow"] = ((d["h_speed"] >= 0.2) & (d["h_speed"] < 2.0)).astype(float)
    d["night"] = ((d["hour"] >= 18) | (d["hour"] < 6)).astype(float)
    g = d.groupby(keys)
    F = pd.DataFrame({
        "n_hours": g.size(),
        "n_days": g["day"].nunique(),
        "spd_mean": g["h_speed"].mean(), "spd_med": g["h_speed"].median(),
        "spd_p10": g["h_speed"].quantile(0.1), "spd_p90": g["h_speed"].quantile(0.9),
        "spd_std": g["h_speed"].std(),
        "moving_frac": g["moving"].mean(), "fast_frac": g["fast"].mean(), "slow_frac": g["slow"].mean(),
        "night_frac": g["night"].mean(),
        "straight_mean": g["h_straight"].mean(), "straight_med": g["h_straight"].median(),
        "turn_mean": g["h_turn"].mean(), "turn_med": g["h_turn"].median(),
        "stop_mean": g["h_stop"].mean(),
        "npts_med": g["h_npts"].median(), "nseg_mean": g["h_nseg"].mean(),
        "lat_med": g["h_lat"].median(), "lon_med": g["h_lon"].median(),
        "lat_std": g["h_lat"].std(), "lon_std": g["h_lon"].std(),
    })
    F["hours_per_day"] = F["n_hours"] / F["n_days"]
    F["km_per_day"] = g["h_speed"].sum() * 1.852 / F["n_days"]
    # 활동 반경: 중앙 위치에서 p90 거리
    c = d.join(F[["lat_med", "lon_med"]], on=keys)
    c["r"] = _hav(c["h_lat"].values, c["h_lon"].values, c["lat_med"].values, c["lon_med"].values)
    F["radius_p90"] = c.groupby(keys)["r"].quantile(0.9)
    # 이동 중일 때의 직진성/침로변화 (조업 방식 차이가 드러나는 부분)
    mv = d[d["moving"] == 1].groupby(keys)
    F["straight_moving"] = mv["h_straight"].mean()
    F["turn_moving"] = mv["h_turn"].mean()
    # 0~23시 활동 비중 (이동 중인 시간 기준도 추가: 언제 조업하러 나가는지)
    hod = pd.crosstab([d[k] for k in keys], d["hour"], normalize="index")
    hod = hod.reindex(columns=range(24), fill_value=0)
    hod.columns = [f"hod_{h:02d}" for h in range(24)]
    F = F.join(hod)
    dm = d[d["moving"] == 1]
    if len(dm):
        hm = pd.crosstab([dm[k] for k in keys], dm["hour"], normalize="index").reindex(columns=range(24), fill_value=0)
        hm.columns = [f"hod_mv_{h:02d}" for h in range(24)]
        F = F.join(hm)

    # ---- [개선 2] 위치 특징 ----
    # 정박 위치(모항): 거의 안 움직인 시간들의 위치 중앙값 (없으면 전체 중앙값)
    st = d[d["h_speed"] < STOP_KN].groupby(keys)
    F["home_lat"] = st["h_lat"].median().reindex(F.index).fillna(F["lat_med"])
    F["home_lon"] = st["h_lon"].median().reindex(F.index).fillna(F["lon_med"])
    # 이동 중 활동 해역의 범위와 중심
    F["mv_lat_med"], F["mv_lon_med"] = mv["h_lat"].median(), mv["h_lon"].median()
    for q in (0.1, 0.9):
        F[f"mv_lat_p{int(q * 100)}"] = mv["h_lat"].quantile(q)
        F[f"mv_lon_p{int(q * 100)}"] = mv["h_lon"].quantile(q)
    # 모항에서 떨어진 거리 분포 (연안 조업 vs 먼바다 조업)
    c = d.join(F[["home_lat", "home_lon"]], on=keys)
    c["dh"] = _hav(c["h_lat"].values, c["h_lon"].values, c["home_lat"].values, c["home_lon"].values)
    gh = c[c["moving"] == 1].groupby(keys)["dh"]
    F["dhome_p50"], F["dhome_p90"], F["dhome_max"] = gh.median(), gh.quantile(0.9), gh.max()
    # 하루 출항 수: 정박 → 이동으로 바뀐 횟수 / 일수
    c = c.sort_values(keys + ["H"])
    prev = c.groupby(keys)["moving"].shift(1).fillna(0)
    c["depart"] = ((prev == 0) & (c["moving"] == 1)).astype(float)
    F["departs_per_day"] = c.groupby(keys)["depart"].sum() / F["n_days"]
    return F.drop(columns=["n_hours", "n_days"]).fillna(0)


def _make_model(n_classes, random_state=42, class_weight=None):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  class_weight=class_weight, random_state=random_state, verbose=-1), "LightGBM"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, class_weight=class_weight,
                                              random_state=random_state), "HistGradientBoosting(sklearn)"


def to_vessel(P: pd.DataFrame, weight: pd.Series = None, level="vessel_id") -> pd.DataFrame:
    """(선박 × 월) 확률 → 선박 확률. weight(활동 시간 수)로 가중 평균"""
    w = (weight.reindex(P.index).fillna(1.0) if weight is not None else pd.Series(1.0, index=P.index)).to_numpy()
    Pv = P.mul(w, axis=0).groupby(level=level).sum()
    return Pv.div(Pv.sum(axis=1), axis=0)


class VesselTypeClassifier:
    def __init__(self, random_state=42, class_weight=None):
        self.random_state = random_state
        self.class_weight = class_weight        # [개선 1] 기본 None (대회 지표 = 정확도)
        self.model = None
        self.classes_ = None
        self.features = None
        self.report = {}

    def _model(self, n):
        return _make_model(n, self.random_state, self.class_weight)

    def _vessel_scores(self, y_v, Pv):
        p = Pv.idxmax(axis=1)
        y_v = y_v.reindex(Pv.index)
        return {"n_vessels": int(len(Pv)),
                "accuracy(=micro_F1)": round(float(accuracy_score(y_v, p)), 4),
                "macro_F1": round(float(f1_score(y_v, p, average="macro")), 4),
                "per_class": classification_report(y_v, p, output_dict=True, zero_division=0)}, p

    def validate(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, out_dir: Path = None,
                 n_splits=5, weight: pd.Series = None, month_level="month"):
        """
        (a) 처음 보는 선박: 선박 단위 K-fold OOF → 선박별로 합쳐 정확도
        (b) 같은 선박·다른 기간: 앞쪽 달 학습 → 뒤쪽 달 예측 (X 인덱스에 month 가 있을 때)
        반환 report, 그리고 self.oof_vessel (선박별 OOF 확률 = 이상탐지 vessel_oof.csv 용)
        """
        groups = pd.Series(np.asarray(groups), index=X.index)
        y_v = y.groupby(groups.values).agg(lambda s: s.mode().iloc[0])
        labels = sorted(y.unique())
        oof = pd.DataFrame(np.nan, index=X.index, columns=labels)
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in skf.split(X, y, groups):
            m, name = self._model(y.nunique())
            m.fit(X.iloc[tr], y.iloc[tr])
            oof.iloc[va] = pd.DataFrame(m.predict_proba(X.iloc[va]), columns=m.classes_)[labels].to_numpy()
        oof.index = pd.MultiIndex.from_arrays([groups.values, np.arange(len(oof))], names=["vessel_id", "_i"])
        w = None if weight is None else pd.Series(weight.reindex(X.index).to_numpy(), index=oof.index)
        Pv = to_vessel(oof, w)
        self.oof_vessel = Pv
        rep_a, p = self._vessel_scores(y_v, Pv)
        cm = confusion_matrix(y_v.reindex(Pv.index), p, labels=labels)
        self.report = {"model": name, "class_weight": str(self.class_weight),
                       "a_unseen_vessels_oof": rep_a,
                       "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
                       "class_counts": y_v.value_counts().to_dict()}
        # (b) 시간 분리
        if month_level in (X.index.names or []):
            mo = pd.Series(X.index.get_level_values(month_level)).astype(str)
            cut = sorted(mo.unique())[len(mo.unique()) // 2]
            tr, va = np.flatnonzero(mo < cut), np.flatnonzero(mo >= cut)
            if len(tr) and len(va):
                m, _ = self._model(y.nunique())
                m.fit(X.iloc[tr], y.iloc[tr])
                P = pd.DataFrame(m.predict_proba(X.iloc[va]), columns=m.classes_)[labels]
                P.index = pd.MultiIndex.from_arrays([groups.values[va], np.arange(len(va))], names=["vessel_id", "_i"])
                wb = None if weight is None else pd.Series(weight.reindex(X.index).to_numpy()[va], index=P.index)
                self.report["b_same_vessels_later_months"] = self._vessel_scores(y_v, to_vessel(P, wb))[0]
                self.report["b_split"] = f"{cut} 이전 학습 → {cut} 이후 예측"
        if out_dir is not None:
            pd.DataFrame(cm, index=[f"실제_{l}" for l in labels], columns=[f"예측_{l}" for l in labels]) \
                .to_csv(Path(out_dir) / "classifier_confusion_matrix.csv", encoding="utf-8-sig")
        return self.report

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.features = list(X.columns)
        self.model, self.model_name = self._model(y.nunique())
        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        P = self.model.predict_proba(X.reindex(columns=self.features, fill_value=0))
        idx = P.argmax(1)
        out = pd.DataFrame({"pred_type": np.array(self.classes_)[idx], "pred_prob": P.max(1)}, index=X.index)
        for j, c in enumerate(self.classes_):
            out[f"prob_{c}"] = P[:, j]
        return out

    def predict_vessels(self, X: pd.DataFrame, weight: pd.Series = None, level="vessel_id") -> pd.DataFrame:
        """[개선 4] 선박 × 월 → 선박당 1줄 (확률 가중 평균 후 최댓값 업종)"""
        P = self.predict(X)[[f"prob_{c}" for c in self.classes_]]
        Pv = to_vessel(P, weight, level)
        out = pd.DataFrame({"pred_type": Pv.idxmax(axis=1).str.replace("prob_", "", regex=False),
                            "pred_prob": Pv.max(axis=1)}, index=Pv.index)
        return out.join(Pv)

    def importance(self) -> pd.Series:
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            return pd.Series(dtype=float)
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def save_report(self, out_dir: Path):
        with open(Path(out_dir) / "classifier_report.json", "w", encoding="utf-8") as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2, default=str)
        self.importance().to_csv(Path(out_dir) / "classifier_feature_importance.csv", encoding="utf-8-sig")

