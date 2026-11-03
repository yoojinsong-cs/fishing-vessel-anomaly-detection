# NAVIS 시각화 (viz)

이상탐지 결과를 지도·그래프·표로 보여 주는 대시보드와 사건 지도를 만드는 코드.
서버나 DB 없이, 저장소 안의 **결과 파일을 읽어 HTML 한 개로** 만든다.

## 데이터 흐름

```
[팀이 만든 결과 파일]              [시각화 스크립트]                 [화면]
out/NAVIS_anomaly/*.csv, *.json ─┐
out/NAVIS_anomaly/test_scoring/out/testB_* ─┼─▶ viz/build_dashboard.py ─▶ out/NAVIS_anomaly/dashboard.html
viz/korea_land.json (해안선)  ─┘    (집계 · 좌표 변환)          out/NAVIS_anomaly/dashboard_testB.html

out/NAVIS_anomaly/alarm_events.csv ─▶ viz/build_event_map.py ─▶ out/NAVIS_anomaly/event_map.html
out/NAVIS_anomaly/alarm_rows.csv   ─┘
```

- 스크립트가 파일을 읽어 집계한 뒤 **데이터를 HTML 안에 직접 넣는다.** 그래서 HTML만 열어도 보인다.
- 원본 항적 데이터는 읽지 않는다. 입력은 `run_all.py`(이상탐지 `NAVIS_anomaly_final`, step7~9)가 만든 결과 파일이다. 경로는 `NAVIS_anomaly_final/navis_paths.py`를 따른다(기본 `out/NAVIS_anomaly/`).
  결과 CSV를 새로 만들어 교체하면 대시보드도 같이 바뀐다.

## 실행

```bash
# 저장소 루트에서
python viz/build_dashboard.py            # 대시보드 생성 (파일로 바로 열어도 됨)
python viz/build_dashboard.py --serve    # 로컬 서버 + 자동 갱신
python viz/build_event_map.py            # 사건 지도 생성
```

| 옵션 | 설명 |
| --- | --- |
| `--serve` | `http://127.0.0.1:8000/out/NAVIS_anomaly/dashboard.html` 로 열림. 입력 파일이 바뀌면 5초 안에 다시 만들고, 열린 화면은 15초마다 새 데이터를 가져옴 |
| `--port 8000` | 서버 포트 |
| `--anon` | 선박 번호 뒤쪽을 `*`로 가림 (외부 공유용) |

필요한 패키지: `pandas`, `numpy`, `pyarrow`, `pyproj`, `openpyxl` (테스트B 화면의 xlsx 읽기).
`make_land.py`만 `shapely`를 추가로 쓴다.

## 화면 1. 전체 분석 결과 (`dashboard.html`)

| 카드 | 읽는 파일 | 쓰는 값 |
| --- | --- | --- |
| 상단 숫자 | `alarm_events.csv`, `alarm_vessels.csv`, `final_thresholds.json` | 경보·전체 사건 수, 선박 수, 하루 경보 건수 |
| 결과 1 혼동행렬 | `vessel_mismatch.csv` | 등록 업종(`code`)과 예측 업종(`pred_code`)을 교차 집계 |
| 결과 2 이상점수 | `alarm_rows.csv`, `alarm_events.csv`, `final_thresholds.json` | 시간별 점수, 사건 구간, 주의·경보 기준선 |
| 결과 3 위험 해역 | `alarm_events.csv`, `alarm_rows.csv` | 사건 위치를 0.25° 격자(약 25km)로 집계 |
| 결과 4 목록 | `alarm_vessels.csv` | 경보 사건이 많은 선박 30척 (상습 단절·정지 선박 제외) |
| 보고서 팝업 | `completion_check.csv` | 목표 대비 실제 결과 표 |

## 화면 2. 테스트B 제출 결과 (`dashboard_testB.html`)

테스트셋 전체 결과(`run_all.py`의 `full`·`submission` 단계)가 있으면 그것을, 없으면 9단계(제출샘플 선박) 결과를 쓴다.

| 입력 | 테스트셋 전체 (우선) | 9단계 (제출샘플 선박) |
| --- | --- | --- |
| 사건 | `test_scoring/full/full_testB_events.csv` | `test_scoring/out/testB_events.csv` |
| 위치 | `test_scoring/full/full_testB_scores.parquet` | `test_scoring/out/testB_scores.parquet` |
| 제출 시트 | `out/submission.xlsx` | `test_scoring/out/testB_submission.xlsx` |
| 근거 | `test_scoring/full/full_testB_근거.csv` | `test_scoring/out/testB_submission_근거.csv` |

선박이 많으면 목록·시계열에는 사고시각 점수가 높은 300척(`TOP_B`)만 담는다. 지도에는 모든 선박의 사고 지점을 찍는다.

아래 표는 각 카드가 쓰는 값이다 (파일 이름은 9단계 기준).

| 카드 | 읽는 파일 |
| --- | --- |
| 결과 1 | `testB_submission.xlsx` 의 테스트A 시트, `testB_submission_근거.csv` |
| 결과 2 | `testB_submission.xlsx` 의 테스트B 시트 (사고 시각 + 전 24시간 점수) |
| 결과 3 | `testB_events.csv`, `testB_scores.parquet` (좌표) |
| 결과 4 | `testB_submission_근거.csv`, `testB_events.csv` |

- 테스트B에는 임계값 파일이 없어서 경보 기준선은 `out/NAVIS_anomaly/final_thresholds.json` 의 값을 그대로 쓴다 (같은 모델이라는 가정).
- 테스트A 시트가 비어 있으면 분류 결과 대신 업종 구성만 보여 주고 "미작성"으로 표시한다.

## 지도와 외부 연결

- **해안선**: `korea_land.json` (Natural Earth, 퍼블릭 도메인)을 HTML에 직접 넣는다. 지도 타일 서버를 쓰지 않아서 `file://` 로 열어도 막히지 않는다. 다시 만들려면 `make_land.py`.
- **좌표**: 원본이 EPSG:5179(미터)라서 `pyproj` 로 위경도(EPSG:4326)로 바꾼다.
- **인터넷에서 불러오는 것**: Chart.js 4.4.1, Leaflet 1.9.4 (cdnjs). 데이터를 밖으로 보내는 곳은 없다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `build_dashboard.py` | 대시보드 두 화면 생성, `--serve` 자동 갱신 |
| `dashboard_template.html` | 대시보드 화면 틀 |
| `build_event_map.py`, `event_map_template.html` | 사건 지도 |
| `korea_land.json`, `make_land.py` | 해안선 데이터와 생성 스크립트 |
| `expected_dashboard.html` | 가상 값으로 만든 예시 화면 (실제 결과 아님) |

## 알아 둘 점

- **데이터에 없던 값, 시각화에서 정한 것**: 위험 격자 크기(0.25°)와 상위 10%·30% 등급 기준, 선박 순위(경보 건수순), 변수 이름을 한글로 풀어 쓴 탐지 사유 표현.
- **결과 1(혼동행렬)은 최종 분류 모델이 아니다.** 이상탐지 5단계가 입력으로 받는 1차 분류 결과(`vessel_mismatch.csv`)이며, 화면에 "잠정"으로 표시한다.
- **선박 번호가 그대로 들어간다.** 공개할 때는 `--anon` 으로 만들 것.
- 결과 파일은 `run_all.py`로 만든 최신 결과(V2.5 최종, 1~6월 전체)를 쓴다.
