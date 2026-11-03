# NAVIS — 3번 파트: 이상탐지 (정상 기준 · 경보 임계값)

업종 × 시간대별 "정상" 기준을 세우고, 평소와 다르게 움직이는 어선을 **0~1 이상점수**로 찾아 **주의/경보 2단계**로 분류한다.

## ★ 데이터 준비 (가장 먼저)

원본 데이터는 용량이 커서 저장소에 들어 있지 않다. **데이터가 들어 있는 폴더를 환경변수 `NAVIS_DATA`로 알려 주면** 프로그램이 알아서 찾아 읽는다.

**필요한 데이터** (한 폴더에 모아 두면 됨, 하위 폴더에 나눠 있어도 됨)
- 월별 V-Pass csv 6개 (`vpass_1월 데이터.csv` ~ `vpass_6월 데이터.csv`) — 또는 이것들이 들어 있는 대회 zip 그대로
- `26년 1~6월 어선 사고현황(테스트 데이터).xlsx`
- 그 폴더 안의 다른 파일(.json, .tmp 등)은 자동으로 무시된다

**① 환경변수 `NAVIS_DATA` 설정 — 데이터 폴더 경로를 적는다**
아래 명령에서 **`<데이터 폴더 경로>` 부분(꺾쇠 `< >` 포함)을 본인 컴퓨터의 실제 데이터 폴더 경로로 바꿔서** 입력한다. (그대로 복사해 붙이면 동작하지 않는다)
- 경로 복사 방법(Windows): 데이터 폴더를 연 상태에서 위쪽 주소창을 클릭 → 경로가 선택되면 `Ctrl+C`
- 경로 복사 방법(Mac): 폴더를 우클릭 → `Option` 키를 누른 채 "경로 이름 복사"

| 환경 | 입력할 명령 |
|---|---|
| Windows PowerShell | `[Environment]::SetEnvironmentVariable("NAVIS_DATA", "<데이터 폴더 경로>", "User")` |
| Windows cmd | `setx NAVIS_DATA "<데이터 폴더 경로>"` |
| Mac / Linux | `echo 'export NAVIS_DATA="<데이터 폴더 경로>"' >> ~/.bashrc` (Mac은 `~/.zshrc`) |

**명령을 입력한 뒤에는 PowerShell/cmd/터미널을 닫았다가 새로 열어야 적용된다.** (한 번 설정하면 컴퓨터를 껐다 켜도 유지된다.)
설정이 됐는지 확인(새로 연 창에서): PowerShell `echo $env:NAVIS_DATA` / cmd `echo %NAVIS_DATA%` / Mac·Linux `echo $NAVIS_DATA` → 경로가 나오면 성공.
(Windows는 시작 메뉴에서 "환경 변수 편집"을 검색해 사용자 변수에 `NAVIS_DATA`를 추가해도 된다.)

**② 실행 — 새로 연 창에서, 이 README가 있는 폴더(`run_submission.py`가 있는 곳)로 이동해서 실행**

```bash
pip install -r requirements.txt
python run_submission.py
```
실행하면 맨 앞에 `[데이터] --data 생략 → (설정한 경로) 사용`이 출력된다. 내 경로가 보이면 제대로 읽은 것이다. 그다음 `원본 csv 6개: vpass_1월 …` 목록이 나오면 데이터 인식까지 끝난 것이다.

**환경변수를 쓰기 싫을 때 (같은 효과)**
- 데이터를 `./data` 폴더(실행하는 위치 바로 아래)에 넣으면 인자 없이 `python run_submission.py`만 쳐도 찾는다.
- 또는 직접 지정: `python run_submission.py --data "<데이터 폴더 경로>"` (파일·폴더·와일드카드, 상대/절대경로 모두 가능. zip 그대로도 가능)

**데이터를 찾는 순서** (`--data`를 안 줬을 때): ① 환경변수 `NAVIS_DATA` → ② `./data` → ③ 코드 옆 `data` 폴더 → ④ 현재 폴더. 이 중 zip/csv가 들어 있는 첫 번째 위치를 쓴다. 모두 없으면 `원본 데이터를 찾지 못함` 안내를 출력하고 멈춘다.

**자주 나는 문제**
| 증상 | 원인 / 해결 |
|---|---|
| `원본 데이터를 찾지 못함` | `NAVIS_DATA`를 설정하지 않았거나, 설정 전에 열어 둔 창에서 실행함, 폴더에 zip/csv가 없음. 확인: `echo $env:NAVIS_DATA` |
| 설정했는데 안 먹힘 | 이미 열려 있던 창에는 적용되지 않음 → 창을 닫고 새로 열기 |
| 경로에 공백·한글이 있음 | 경로 전체를 큰따옴표(`" "`)로 감싸기 |
| 메모리 부족·너무 느림 | 월별 csv가 0.6~1.2GB. RAM 16GB 이상 권장. 먼저 `--sample-vessels 300`으로 동작만 확인 가능 |

## ★ 제출 파일 만들기 (testA·testB)

```bash
python run_submission.py                                   # NAVIS_DATA(또는 ./data)에서 읽음
python run_submission.py --sample-vessels 300 --out sample  # 빠른 동작 확인용 (제출용 아님, results를 덮지 않게 --out 지정)
```
결과는 `--out`(기본 `results`)에 저장되며, **실행할 때마다 기존 파일을 최신 결과로 덮어쓴다.** 대회 zip을 그대로 넣으면 월별 V-Pass csv 6개와 테스트 xlsx를 알아서 찾는다.

| 단계 | 내용 |
|---|---|
| ① 사고 선박 매칭 | 테스트 엑셀에 rfid_id가 없으므로, 사고마다 발생시각 -2h~+1h의 V-Pass 위치 중 사고 위치와 가장 가까운 선박을 찾음 (같은 어업면허 우선, 시간차 1시간당 0.5km 벌점, 최대 10km) → `accident_vessel_match.csv` |
| ② testA 분류 | 선박×월 단위 특징(0~23시 활동 비중, 속력·직진성·침로변화·정지 분포, 활동 해역)으로 LightGBM(없으면 HistGB, class_weight 기본 None) 학습 → 매칭된 선박의 조업형태·확률 예측. 테스트 선박은 학습에서 제외 |
| ③ testB 시각별 점수 | 각 시간 h마다 주변 5시간 [h-2, h+2] 항적 특징 + 해당 시간 특징(속력, 이후 신호공백, 속력급감, 표류시작, 모항거리 등)을 업종×시간대 조건부 이상탐지(Isolation Forest + 위험지표 정상범위 이탈)로 0~1 점수화. 제출 점수 = (해당 시간 + 다음 시간)/2 → 사고 당일 중 최고 점수 시각 = 사고시각 |
| ④ 검증 | 인위적 사고(표류·신호단절, 사고 시각을 아는 궤적)로 사고시각 적중률 + 실제 사고 167건의 발생시각·어업면허와 비교 |

출력: `submission.xlsx`(testA·testB 시트), `real_accident_eval.json`·`real_accident_detail.csv`(실제 사고 대비 적중률), `hourly_validation_report.json`, `classifier_report.json`·`classifier_confusion_matrix.csv`, `testB_hourly_scores.csv`(시간별 점수 + 이상 사유)

1월 데이터로 돌려본 검증(참고): 분류 정확도 0.53 / macro F1 0.47 (HistGB, 1개월) · 인위적 사고 사고시각 ±1시간 적중 83.5% (표류 88%, 신호단절 79%) · 사고 이후 시간 vs 정상 ROC-AUC 0.93

## 빠른 실행 (6시간대 단위 이상탐지 분석용)

```bash
pip install -r requirements.txt

# V-Pass 원본(part_dt, rfid_id, rfid_time, geom, fishing_code)에서 바로 — 1~6월 zip/csv 여러 개 가능
python run_vpass.py --raw "data/vpass_*월*.zip" --out outputs_vpass

# 한 번 변환해 두면 다음부터는 변환 생략 (몇 분 → 1분)
python run_vpass.py --features outputs_vpass/features_vpass.csv --sequences outputs_vpass/sequences_vpass.npz --out outputs_vpass

# 1번 담당 특징 테이블이 나오면 그걸로
python run_anomaly.py --features features.csv --sequences points.csv --preds pred_type.csv --out outputs/
```

- 2번 분류 결과가 나오면 `--preds pred_type.csv`(trip_id 또는 vessel_id, pred_type, pred_prob)만 붙이면 업종 불일치가 자동 반영된다
- 빠른 테스트: `--sample-vessels 500`

## 흐름

```
V-Pass 원본 ─▶ geom 파싱 · 해역 밖/GPS 튐/비정상 고속 좌표 제거 ─▶ 분석단위(선박×날짜×6시간대) 특징 + 시퀀스
                                                                    │
     ┌──────────────────────────────────────────────────────────────┘
     ├─▶ 업종×시간대 정상 기준 ─┬─▶ ① Isolation Forest (조건부 정규화 특징의 조합이 드문 정도)
     │                          └─▶ ② 정상범위 이탈 (위험지표가 자기 그룹에서 얼마나 드문지)
     ├─▶ ③ 시퀀스 오토인코더 (재구성 오차, 현재 결합 제외·참고용)
     └─▶ 2번 예측 업종 ≠ 등록 업종 → 예측확률만큼 가산
                         ▼
        이상점수(0~1) = 평균(①, ②) ⊕ 업종불일치  →  주의/경보 임계값  →  인위적 이상 궤적으로 검증
```

| 단계 | 내용 |
|---|---|
| 분석 단위 | 선박 × 날짜 × 6시간대 (새벽 00-06 / 오전 06-12 / 오후 12-18 / 야간 18-24) |
| 시간대별 EDA | 업종 × 시(0~23)별 정지 비율·활동 비중 → `eda_hourly_stop_move.csv/png` |
| 정상 기준 | 업종×시간대 그룹별 특징 분포(중앙값·1~99% 정상범위). 표본 30개 미만 그룹은 업종 → 전체 기준으로 대체 |
| ① Isolation Forest | 각 궤적을 **자기 그룹 기준 robust z-score**로 바꾼 뒤 학습 → 같은 값이라도 업종·시간대에 따라 정상/이상이 달라짐 |
| ② 정상범위 이탈 | 위험지표(최장 신호단절, 표류의심시간·비율, 최장 정지시간)가 **자기 업종×시간대 정상 분포에서 상위 몇 %인지** → 가장 드문 값의 꼬리확률 |
| ③ 시퀀스 AE | 점 순서 [근사속력, sin/cos 침로, dx·dy, 시간간격]을 64스텝으로 맞춰 LSTM AE(없으면 sklearn MLP) |
| 점수 | 각 점수를 "정상 학습 데이터 대비 백분위"(0~1)로 통일 → ①②평균 → 업종불일치 시 `1-(1-s)(1-0.3·예측확률)` |
| 임계값 | 검증 기간 **정상** 궤적 점수의 95% 분위수 = 주의, 99% = 경보 |
| 검증 | 날짜 기준 기간 분리(마지막 20% 기간 = 검증). 인위적 이상 궤적은 학습에서 빼고 검증에만 사용 |
| 이상 사유 | 각 궤적에서 정상 분포 대비 가장 드문 특징 3개 (예: `최장신호단절↑(상위 0.1%)`) |

## 1월 데이터 결과 (2026-01, 선박 5,014척)

- 분석단위 117,405 → 조업 궤적 74,571 (항내 정박 36,678·점 부족 6,156 제외) + 인위적 이상 1,408 (표류 827, 신호단절 581)
- 학습: 1/1~1/24 정상 궤적 55,135 / 검증: 1/25~1/31 정상 19,436 + 인위적 이상 1,408
- **임계값: 주의 ≥ 0.905, 경보 ≥ 0.975** (검증 정상 궤적 오탐률 5.0% / 1.0%)

| 모델 | ROC-AUC | PR-AUC |
|---|---|---|
| Isolation Forest | 0.898 | 0.323 |
| 정상범위 이탈 | 0.953 | 0.533 |
| 오토인코더 (sklearn 대체판) | 0.420 | 0.065 |
| **최종 (IF + 정상범위)** | **0.955** | **0.569** |

| 인위적 이상 | 주의 이상 탐지율 | 경보 탐지율 |
|---|---|---|
| 표류 (drift) | 73.3% | 28.9% |
| 신호 단절 (signal_loss) | 77.8% | 25.1% |

- 업종별(연안복합·자망·통발 ≈ 5%, 채낚기연승낚시 7.3%)·시간대별(4~6%) 주의 이상 비율이 크게 치우치지 않음 → 조건부 정상 기준이 업종·시간대 차이를 흡수
- 실제 궤적: 정상 70,853 / 주의 3,077 / 경보 641 → 경보 선박 359척, 주의 선박 920척

## 데이터 가정 · 한계 (보고서에 명시할 것)

1. **점별 시각·속력이 원본에 없음** → 한 시간(rfid_time) 안의 점들이 고르게 찍혔다고 보고 근사 시각 부여, 속력은 근사값
2. **시간대 내 총 이동 1km 미만은 항내 정박으로 보고 제외** → 해상에서 계속 정지해 있는 배는 이 기준으로 걸러질 수 있음 (항구 좌표가 있으면 개선 가능)
3. 근사 속력 50kn 초과·GPS 튐 좌표 약 93만 개(전체의 약 5%) 제거
4. 인위적 이상: 표류 = 궤적 중간 이후를 0.3~1.0kn 한 방향 이동으로 교체, 신호단절 = 중간 2~3시간 점 삭제 (1번 담당 버전이 나오면 교체)
5. **오토인코더는 1월 검증에서 기여가 없어(ROC 0.42) 결합에서 제외** (`config.w_ae = 0`). 이상 궤적이 오히려 단순해져 재구성이 쉬워지는 문제. 점수(s_ae)는 계속 저장되므로 PyTorch LSTM(`--ae-backend torch`)으로 6개월 데이터에서 다시 확인 후 가중치 조정
6. 2번 분류 예측이 아직 없어 업종 불일치는 미반영

## 입력 형식 (컬럼명이 다르면 `navis_anomaly/config.py`의 `Columns`만 수정)

**특징 테이블** (궤적 1개 = 1행): `vessel_id`, `trip_id`, `reg_type`, `date`, `start_time` 또는 `time_band`, 특징들, 인위적 이상이면 `is_synthetic=1`, `anomaly_type`
— 없는 특징은 자동 제외. 특징 목록·위험지표 방향은 `config.py`의 `DEFAULT_FEATURES`, `DEFAULT_RISK`

**시퀀스**: 점 단위 csv/parquet (`trip_id, seq_idx, timestamp, lat, lon, speed, course`) 또는 npz (`X`(N,T,F), `ids`)

## 출력 (→ 4번 시각화 담당에게)

| 파일 | 내용 |
|---|---|
| `anomaly_scores.csv` | **선박·시간대별 이상점수**: s_if, s_range, s_ae, anomaly_score, level, reasons, center_lat/lon(지도용) |
| `alert_vessels.csv` | **경보·주의 선박 목록** (최고 점수, 주의/경보 횟수, 가장 이상했던 날짜·시간대·사유) |
| `thresholds.json` | **임계값 기준** |
| `normal_baseline.csv` | 업종×시간대별 정상 기준표 |
| `validation_report.json`, `validation_score_hist.png` | AUC, 유형별 탐지율, 오탐률, 점수 분포 |
| `eda_hourly_stop_move.csv/png`, `eda_type_timeband_summary.csv` | 시간대별 EDA |
| `shap_input_iforest.csv`, `models/iforest.joblib` | SHAP 분석용 |
| `models/navis_anomaly_detector.joblib` | 학습된 탐지기 전체 |
| `features_vpass.csv`, `sequences_vpass.npz` | 원본에서 변환한 특징·시퀀스 (재실행용) |

SHAP (4번 참고):
```python
import joblib, pandas as pd, shap
m = joblib.load("outputs_vpass/models/iforest.joblib")
X = pd.read_csv("outputs_vpass/shap_input_iforest.csv").drop(columns=["vessel_id", "trip_id", "anomaly_score"])
sv = shap.TreeExplainer(m).shap_values(X)   # 음수일수록 이상 쪽 기여
```

## 파일 구조

```
run_submission.py           ★ 제출 파일 생성 (매칭 → 분류 → 시각별 이상점수 → xlsx)
run_vpass.py                V-Pass 원본에서 실행
run_anomaly.py              1번 특징 테이블에서 실행
make_demo_data.py           데모용 가상 데이터
navis_anomaly/
  config.py                 컬럼명·특징·위험지표·가중치·임계값 설정
  vpass_raw.py              V-Pass 원본 → 특징·시퀀스·인위적 이상 궤적 (6시간대 단위)
  hourly_raw.py             V-Pass 원본 → 시간 단위 5시간 창 특징·인위적 사고·시간별 통계
  classifier.py             조업형태 분류 (선박 단위 특징 + LightGBM)
  match_accidents.py        사고 목록 ↔ V-Pass 선박 매칭
  test_targets.py           테스트 엑셀 읽기
  raw_sources.py            zip/csv 안의 월별 원본 찾기
  io_utils.py               데이터 로드, 시퀀스 변환
  eda.py                    시간대별 EDA
  baseline.py               업종×시간대 정상 기준, 꼬리확률, 이상 사유
  models.py                 Isolation Forest, 시퀀스 오토인코더, 점수 보정
  pipeline.py               학습·점수·임계값·검증
  runner.py                 전체 실행·저장
  features_min.py           최소 특징 계산기 (데모용)
```
