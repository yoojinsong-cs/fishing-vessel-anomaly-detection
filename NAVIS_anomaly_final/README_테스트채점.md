# 테스트셋 채점 (V2.5 기준)

테스트 채점 파일은 이제 `NAVIS_anomaly_final` 폴더 안에 함께 들어 있다. 폴더 경로는 `navis_paths.py`에서 가져온다 (README.md의 '경로 설정').

| 파일 | 하는 일 |
| --- | --- |
| `test_config.py` | 추적 시간(24h), 항구 안 처리 여부, 점수 보정 (폴더 경로는 navis_paths.py) |
| `filter_testB.py` | 테스트셋 CSV(약 7GB)에서 테스트B 대상 100척만 뽑기 (2~3분) |
| `step9_testB.py` | V2.5 모델로 선박·시간별 점수와 사건 계산 (3~10분). 학습 데이터에 없는 선박의 업종은 분류 모델 함수(`cls_bridge.predict_vessels`)로 예측 |
| `step9b_submission.py` | 제출 파일(테스트B) 만들기 (1분 미만) |

실행 (가상환경을 켠 터미널, 코드 폴더에서):

```
python filter_testB.py
python step9_testB.py
python step9b_submission.py
```


결과: `NAVIS_anomaly/test_scoring/out/` 의 `testB_submission.xlsx`, `testB_submission_근거.csv`, `testB_scores.parquet`, `testB_events.csv`
