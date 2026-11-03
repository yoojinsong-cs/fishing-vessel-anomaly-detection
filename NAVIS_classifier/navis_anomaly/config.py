"""
NAVIS 이상탐지(3번 파트) 설정 파일
------------------------------------------------
1번(데이터) 담당이 넘겨준 파일의 컬럼명이 다르면 이 파일만 고치면 된다.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Columns:
    # ---- 식별자 ----
    vessel_id: str = "vessel_id"          # 선박 ID
    unit_id: str = "trip_id"              # 분석 단위 ID (항차/일자 단위 궤적). 없으면 vessel_id+date로 자동 생성
    date: Optional[str] = "date"          # 날짜 (기간 분리 검증에 사용, 없으면 선박 기준 분리)
    start_time: Optional[str] = "start_time"  # 궤적 시작 시각 (time_band 없을 때 주간/야간 계산용)

    # ---- 조건(그룹) 변수 ----
    reg_type: str = "reg_type"            # 등록 업종 (자망, 통발, 연승 ...)
    time_band: str = "time_band"          # 시간대 구분 (주간/야간). 없으면 start_time으로 계산
    sea_area: Optional[str] = "sea_area"  # 해역 구분 (선택)

    # ---- 검증용 인위적 이상 궤적 표시 ----
    is_synthetic: str = "is_synthetic"    # 1 = 1번 담당이 만든 인위적 이상 궤적
    anomaly_type: str = "anomaly_type"    # drift(표류) / signal_loss(신호 단절) 등

    # ---- 시퀀스(점 단위) 데이터 ----
    seq_order: str = "seq_idx"            # 점 순서
    seq_time: Optional[str] = "timestamp" # 점 시각 (있으면 시간 간격 dt를 특징으로 사용)
    lat: str = "lat"
    lon: str = "lon"
    speed: str = "speed"                  # 속력 (knot)
    course: str = "course"                # 침로 (deg, 0~360)

    # ---- 분류모델(2번) 예측 결과 ----
    pred_type: str = "pred_type"          # 예측 업종
    pred_prob: str = "pred_prob"          # 예측 업종 확률


# 특징 테이블에서 이상탐지에 쓸 특징 변수 (1번 담당 특징 목록 기준)
DEFAULT_FEATURES: List[str] = [
    "speed_mean", "speed_max", "speed_std",
    "distance_km", "straightness",
    "course_change_mean", "sharp_turn_cnt",
    "stop_ratio", "max_stop_min",
    "radius_km", "n_points", "n_segments",
    "max_gap_min",           # 최장 신호 단절 시간 (없으면 자동 제외)
    "active_hours",          # 시간대 안에서 신호가 잡힌 시간 수 (없으면 자동 제외)
    "drift_hours",           # 저속·직선 이동(표류 의심)이 연속된 시간 수 (없으면 자동 제외)
    "drift_hour_ratio",      # 표류 의심 시간 비율 (없으면 자동 제외)
]

# 시간 단위(제출 testB용) 이상탐지 특징: 5시간 창 특징 + 중심 시간 특징
HOURLY_FEATURES: List[str] = [f for f in DEFAULT_FEATURES] + [
    "c_speed",          # 중심 시간 속력
    "c_stop",           # 중심 시간 정지 비율
    "d_speed_prev",     # 직전 시간 대비 속력 변화 (급감 = 사고 의심)
    "gap_after_min",    # 중심 시간 이후 신호 공백 (사고 후 신호 끊김)
    "gap_before_min",
    "spd_drop",         # 앞 2시간 대비 이후 속력 감소
    "drift_onset",      # 이후 표류의심 시간 수 - 이전 표류의심 시간 수 (표류 '시작' 시점에서 최대)
    "home_dist_km",     # 모항(정박 위치)에서 거리
    "gap_at_sea",       # 모항에서 3km 이상 떨어진 바다에서의 신호 공백
]
HOURLY_RISK = {
    "gap_at_sea": "+",       # 바다 한가운데서 신호 끊김
    "drift_onset": "+",      # 표류 시작
    "spd_drop": "+",         # 급감속
    "max_stop_min": "+",     # 해상 장시간 정지
}

# 결과 파일에 그대로 넘길 컬럼 (4번 지도용 좌표 등, 있으면)
PASSTHROUGH: List[str] = ["center_lat", "center_lon", "removed_pts"]


# 위험지표: 정상 범위 이탈을 '위험 방향' 한쪽으로만 본다 (+ 클수록 위험 / - 작을수록 위험)
DEFAULT_RISK = {
    "max_gap_min": "+",       # 신호 단절이 길수록
    "drift_hours": "+",       # 저속 직선 표류 의심 시간이 길수록
    "drift_hour_ratio": "+",
    "max_stop_min": "+",      # 해상 장시간 정지
}


@dataclass
class Config:
    cols: Columns = field(default_factory=Columns)
    features: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    passthrough: List[str] = field(default_factory=lambda: list(PASSTHROUGH))
    risk_features: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_RISK))
    range_mode: str = "risk"       # risk: 위험지표만 위험 방향으로 / all: 전체 특징 양방향

    # 주간/야간 구분 (time_band 컬럼이 없을 때만 사용)
    day_start_hour: int = 6
    day_end_hour: int = 18
    day_label: str = "주간"
    night_label: str = "야간"

    # 정지 판정 속력 기준 (knot) - 시간대별 EDA용
    stop_speed_kn: float = 0.5

    # 조건부 정상 기준: 그룹 표본 수가 적으면 상위 그룹으로 대체
    #   (업종×시간대) → (업종) → (전체)
    min_group_size: int = 30
    baseline_low_q: float = 0.01   # 정상 범위 하한 분위수
    baseline_high_q: float = 0.99  # 정상 범위 상한 분위수

    # Isolation Forest
    if_n_estimators: int = 300
    if_max_samples: str = "auto"
    if_contamination: str = "auto"

    # 시퀀스 오토인코더
    seq_len: int = 64              # 모든 궤적을 이 길이로 리샘플링
    ae_hidden: int = 64
    ae_latent: int = 16
    ae_epochs: int = 30
    ae_batch: int = 128
    ae_lr: float = 1e-3
    ae_backend: str = "auto"       # auto | torch | sklearn (torch 없으면 sklearn MLP로 대체)

    # 점수 결합
    #   s_if    : Isolation Forest (여러 특징의 조합이 드문 정도)
    #   s_range : 정상범위 이탈 (자기 업종×시간대 정상 분포에서 가장 드문 특징 하나의 꼬리확률)
    #   s_ae    : 시퀀스 오토인코더 재구성 오차
    combine: str = "mean"          # mean: 가중평균 / max: 하나라도 강하게 반응하면 반영
    w_if: float = 1.0
    w_range: float = 1.0
    w_ae: float = 0.0              # 1월 실데이터 검증에서 AE(sklearn 대체판)는 탐지 기여가 없어(ROC≈0.45) 결합에서 제외.
                                   # 점수(s_ae)는 계속 계산·저장되므로, PyTorch LSTM으로 다시 검증해 좋아지면 가중치를 올릴 것
    w_mismatch: float = 0.3        # 예측 업종 ≠ 등록 업종일 때 가산 비중 (확률 가중)

    # 2단계 임계값: 검증용 정상 데이터 점수 분포의 분위수
    caution_q: float = 0.95        # 주의: 정상 선박 상위 5%
    alert_q: float = 0.99          # 경보: 정상 선박 상위 1%

    # 검증 분할
    val_ratio: float = 0.2
    random_state: int = 42

    # 이상 사유로 표시할 상위 특징 수
    top_k_reasons: int = 3
