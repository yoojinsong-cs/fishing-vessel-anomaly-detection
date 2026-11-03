"""
업종 × 시간대 조건부 정상 기준
------------------------------------------------
- 그룹별 각 특징의 중앙값, IQR, 정상 범위(1%~99% 분위수) 계산
- 표본이 적은 그룹은 (업종) → (전체) 기준으로 자동 대체
- 각 궤적을 '자기 그룹 기준' robust z-score로 바꿔서 Isolation Forest 입력으로 사용
  → 같은 속력이라도 자망·야간에서는 정상, 통발·주간에서는 이상일 수 있음을 반영
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config

FEATURE_KO = {
    "speed_mean": "평균속력", "speed_max": "최대속력", "speed_std": "속력변동",
    "distance_km": "이동거리", "straightness": "직진성",
    "course_change_mean": "침로변화량", "sharp_turn_cnt": "급변침횟수",
    "stop_ratio": "체류비율", "max_stop_min": "최장정지시간",
    "radius_km": "활동반경", "n_points": "위치점수", "n_segments": "궤적조각수",
    "max_gap_min": "최장신호단절", "active_hours": "신호수신시간",
    "drift_hours": "표류의심시간", "drift_hour_ratio": "표류의심비율",
    "c_speed": "해당시간속력", "c_stop": "해당시간정지비율", "d_speed_prev": "직전대비속력변화",
    "gap_after_min": "이후신호공백", "gap_before_min": "이전신호공백", "spd_drop": "속력급감",
    "drift_onset": "표류시작", "home_dist_km": "모항거리", "gap_at_sea": "해상신호단절",
}


class GroupBaseline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stats: Dict[Tuple, pd.DataFrame] = {}
        self.sorted: Dict[Tuple, Dict[str, np.ndarray]] = {}   # 그룹별 정상값 정렬 배열 (분위 계산용)
        self.features: List[str] = []

    @staticmethod
    def _summ(x: pd.DataFrame, lq: float, hq: float) -> pd.DataFrame:
        med = x.median()
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        scale = (q3 - q1) / 1.349
        std = x.std()
        scale = scale.where(scale > 1e-9, std)          # IQR이 0이면 표준편차로
        scale = scale.where(scale > 1e-9, 1.0)          # 그래도 0이면 1
        return pd.DataFrame({"median": med, "scale": scale,
                             "low": x.quantile(lq), "high": x.quantile(hq), "n": len(x)})

    def fit(self, df: pd.DataFrame):
        c, cfg = self.cfg.cols, self.cfg
        self.features = list(cfg.features)
        X = df[self.features]
        lq, hq = cfg.baseline_low_q, cfg.baseline_high_q
        rng = np.random.default_rng(cfg.random_state)

        def _add(key, g):
            self.stats[key] = self._summ(g[self.features], lq, hq)
            if len(g) > 20000:
                g = g.iloc[rng.choice(len(g), 20000, replace=False)]
            self.sorted[key] = {f: np.sort(g[f].dropna().to_numpy(float)) for f in self.features}

        _add(("__all__",), df)
        for t, g in df.groupby(c.reg_type):
            if len(g) >= cfg.min_group_size:
                _add((t,), g)
        for (t, b), g in df.groupby([c.reg_type, c.time_band]):
            if len(g) >= cfg.min_group_size:
                _add((t, b), g)
        return self

    def tail_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        자기 업종×시간대 정상 분포에서 이 값이 얼마나 드문지 (꼬리확률, 작을수록 드묾)
          - 위험지표(config.risk_features)는 위험 방향 한쪽 꼬리만 ('+'=클수록 위험, '-'=작을수록 위험)
          - 나머지 특징은 양쪽 꼬리 중 작은 쪽
        """
        c = self.cfg.cols
        risk = self.cfg.risk_features
        P = pd.DataFrame(1.0, index=df.index, columns=self.features)
        for (t, b), g in df.groupby([c.reg_type, c.time_band]):
            key = (t, b) if (t, b) in self.sorted else ((t,) if (t,) in self.sorted else ("__all__",))
            arrs = self.sorted[key]
            for f in self.features:
                a = arrs[f]
                n = len(a)
                if n == 0:
                    continue
                x = g[f].to_numpy(float)
                ge = n - np.searchsorted(a, x, side="left")      # 정상 중 x 이상인 개수
                le = np.searchsorted(a, x, side="right")         # 정상 중 x 이하인 개수
                up, lo = (ge + 1) / (n + 1), (le + 1) / (n + 1)
                p = up if risk.get(f) == "+" else lo if risk.get(f) == "-" else np.minimum(np.minimum(up, lo) * 2, 1)
                p = np.where(np.isnan(x), 1.0, p)
                P.loc[g.index, f] = p
        return P

    def _lookup(self, t, b) -> Tuple[pd.DataFrame, str]:
        if (t, b) in self.stats:
            return self.stats[(t, b)], "업종×시간대"
        if (t,) in self.stats:
            return self.stats[(t,)], "업종"
        return self.stats[("__all__",)], "전체"

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """반환: robust z (N×F), 사용된 기준 레벨, 정상범위 이탈 여부 (N×F, -1/0/+1)"""
        c = self.cfg.cols
        Z = pd.DataFrame(index=df.index, columns=self.features, dtype=float)
        V = pd.DataFrame(0, index=df.index, columns=self.features, dtype=int)
        level = pd.Series("", index=df.index)
        for (t, b), g in df.groupby([c.reg_type, c.time_band]):
            st, lv = self._lookup(t, b)
            x = g[self.features].astype(float)
            x = x.fillna(st["median"])
            Z.loc[g.index] = ((x - st["median"]) / st["scale"]).values
            V.loc[g.index] = (np.where(x < st["low"], -1, 0) + np.where(x > st["high"], 1, 0))
            level.loc[g.index] = lv
        Z = Z.clip(-50, 50)  # 극단값 하나가 모델을 지배하지 않도록
        return Z, level, V

    def table(self) -> pd.DataFrame:
        """보고서/공유용 정상 기준 테이블"""
        rows = []
        for key, st in self.stats.items():
            t = key[0] if key[0] != "__all__" else "전체"
            b = key[1] if len(key) > 1 else "전체"
            for f, r in st.iterrows():
                rows.append({"업종": t, "시간대": b, "특징": f, "특징명": FEATURE_KO.get(f, f),
                             "n": int(r["n"]), "중앙값": r["median"],
                             "정상범위_하한": r["low"], "정상범위_상한": r["high"]})
        return pd.DataFrame(rows).sort_values(["업종", "시간대", "특징"]).reset_index(drop=True)


def top_reasons(Z: pd.DataFrame, P: pd.DataFrame, k: int = 3, max_p: float = 0.05) -> pd.Series:
    """
    각 궤적에서 자기 업종×시간대 정상 분포 대비 가장 드문 특징 k개를 글로 표시
    예) 최장신호단절↑(상위 0.4%), 평균속력↓(하위 1.2%)
    """
    Pv, zv = P.to_numpy(), Z.to_numpy()
    order = np.argsort(Pv, axis=1)[:, :k]
    cols = Z.columns.to_numpy()
    out = []
    for i in range(len(Z)):
        parts = []
        for rank, j in enumerate(order[i]):
            if Pv[i, j] > max_p and rank > 0:   # 최소 1개는 표시
                continue
            name = FEATURE_KO.get(cols[j], cols[j])
            up = zv[i, j] >= 0
            pct = Pv[i, j] * 100
            pct_s = "<0.1%" if pct < 0.1 else f"{pct:.1f}%"
            parts.append(f"{name}{'↑' if up else '↓'}({'상위' if up else '하위'} {pct_s})")
        out.append(", ".join(parts))
    return pd.Series(out, index=Z.index)
