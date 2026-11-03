# -*- coding: utf-8 -*-
"""
최종 제출 파일 만들기 — 제출샘플.xlsx와 같은 형식(시트 '테스트A', '테스트B', 같은 칸)으로 하나의 엑셀에 합친다.

대상 선박: 테스트셋 csv에 있는 '모든' 선박 (NAVIS_anomaly_final/step10_full.py 결과)
  테스트A (시트1): rfid_id, 예측 조업형태, 예측조업형태 확률           ← full/full_testA.csv
  테스트B (시트2): rfid_id, 사고시각, 이상항적 점수, 시각1~3 + 점수     ← full/full_testB.json
  칸 이름·칸 수·값 형식(rfid 숫자/문자 등)은 제출샘플.xlsx를 그대로 따른다.

  python make_submission.py            → out/submission.xlsx
  python make_submission.py --out 경로
"""
import argparse
import json
import sys
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "NAVIS_anomaly_final"))
import navis_paths as P  # noqa: E402
import test_config as T  # noqa: E402

FULL = Path(T.TEST_DIR) / "full"
FULL_A = FULL / "full_testA.csv"
FULL_B = FULL / "full_testB.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "out" / "submission.xlsx"))
    args = ap.parse_args()

    for p, what in [(P.SUBMIT_SAMPLE, "제출샘플.xlsx"), (FULL_A, "테스트A 전체 결과 (run_all.py --from full)"),
                    (FULL_B, "테스트B 전체 결과 (run_all.py --from full)")]:
        if not Path(p).exists():
            sys.exit(f"[중단] {what} 파일이 없습니다: {p}")

    sample = openpyxl.load_workbook(P.SUBMIT_SAMPLE, read_only=True)
    head = {n: [c for c in next(sample[n].iter_rows(max_row=1, values_only=True)) if c is not None]
            for n in ("테스트A", "테스트B")}
    # rfid_id 형식(숫자/문자)도 제출샘플을 따른다
    id_num = {n: isinstance(next(sample[n].iter_rows(min_row=2, max_row=2, values_only=True))[0], (int, float))
              for n in ("테스트A", "테스트B")}
    conv = {n: (lambda v: int(v)) if id_num[n] else (lambda v: str(v)) for n in id_num}

    A = pd.read_csv(FULL_A, dtype={"rfid_id": str})
    rowsB = json.load(open(FULL_B, encoding="utf-8"))

    out = openpyxl.Workbook()
    wa = out.active
    wa.title = "테스트A"
    wa.append(head["테스트A"])
    for r in A.itertuples(index=False):
        wa.append([conv["테스트A"](r.rfid_id), r.pred_type, round(float(r.pred_prob), 4)])

    wb = out.create_sheet("테스트B")
    wb.append(head["테스트B"])
    nB = len(head["테스트B"])                 # 제출샘플 칸 수까지만 (시각1~3)
    for r in rowsB:
        row = [conv["테스트B"](r[0])]
        for k, v in enumerate(r[1:nB], start=1):
            row.append(round(float(v), 4) if k % 2 == 0 else str(v))   # 짝수 칸 = 점수, 홀수 칸 = 시각
        wb.append(row)

    idsA = [str(x) for x in A["rfid_id"]]
    idsB = [str(r[0]) for r in rowsB]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"[제출] 저장: {args.out}")
    print(f"  테스트A {len(idsA):,}척 (중복 {len(idsA) - len(set(idsA))}) | "
          f"테스트B {len(idsB):,}척 (중복 {len(idsB) - len(set(idsB))})")
    if set(idsA) != set(idsB):
        print(f"  [주의] 테스트A와 테스트B 선박 목록이 다름: A만 {len(set(idsA) - set(idsB))}척, B만 {len(set(idsB) - set(idsA))}척")


if __name__ == "__main__":
    main()
