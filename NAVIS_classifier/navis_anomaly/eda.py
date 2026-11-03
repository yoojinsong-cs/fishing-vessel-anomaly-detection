"""시간대별 EDA: 몇 시에 정지·이동이 많은지 (업종별)"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def _set_korean_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def hourly_stop_move(points: pd.DataFrame, unit_type: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    점 단위 데이터에서 업종 × 시(0~23)별 정지/이동 비율 계산
    points: unit_id, timestamp, speed 필요
    unit_type: unit_id → reg_type 매핑
    """
    c = cfg.cols
    if not (c.seq_time and c.seq_time in points.columns and c.speed in points.columns):
        print("[EDA] timestamp 또는 speed 컬럼이 없어 시간대별 EDA 생략")
        return pd.DataFrame()
    p = points[[c.unit_id, c.seq_time, c.speed]].copy()
    p[c.unit_id] = p[c.unit_id].astype(str)
    if c.reg_type in points.columns:
        p[c.reg_type] = points[c.reg_type].astype(str).values
    else:
        p = p.merge(unit_type, on=c.unit_id, how="left")
    p["hour"] = pd.to_datetime(p[c.seq_time], errors="coerce").dt.hour
    p = p.dropna(subset=["hour"])
    p["is_stop"] = (p[c.speed] < cfg.stop_speed_kn).astype(int)

    g = p.groupby([c.reg_type, "hour"])
    out = g.agg(n_points=("is_stop", "size"), stop_ratio=("is_stop", "mean"),
                speed_mean=(c.speed, "mean")).reset_index()
    out["move_ratio"] = 1 - out["stop_ratio"]
    # 업종 안에서 그 시간대 점이 차지하는 비중 (언제 많이 활동하는지)
    out["activity_share"] = out["n_points"] / out.groupby(c.reg_type)["n_points"].transform("sum")
    out["hour"] = out["hour"].astype(int)
    return out


def plot_hourly(eda: pd.DataFrame, cfg: Config, out_dir: Path):
    if eda.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _set_korean_font()
    c = cfg.cols
    fig, axes = plt.subplots(1, 2, figsize=(15, max(3, 0.5 * eda[c.reg_type].nunique() + 2)))
    for ax, col, title in [(axes[0], "stop_ratio", "업종 × 시간별 정지 비율"),
                           (axes[1], "activity_share", "업종 × 시간별 활동 비중")]:
        piv = eda.pivot(index=c.reg_type, columns="hour", values=col).reindex(columns=range(24))
        im = ax.imshow(piv.values, aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("시 (0~23)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_hourly_stop_move.png", dpi=150)
    plt.close(fig)


def time_band_summary(feat: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """특징 테이블 기준 업종×시간대별 요약 (궤적 수, 주요 특징 중앙값) — 정상 기준 초안"""
    c = cfg.cols
    normal = feat[feat[c.is_synthetic] == 0]
    agg = normal.groupby([c.reg_type, c.time_band])[cfg.features].median().round(3)
    agg.insert(0, "n_units", normal.groupby([c.reg_type, c.time_band]).size())
    return agg.reset_index()
