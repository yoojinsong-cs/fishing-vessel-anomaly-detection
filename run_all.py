# -*- coding: utf-8 -*-
"""
NAVIS 전체 실행 — 명령 하나로 분류 모델(testA)과 이상탐지(V2.5 최종)를 순서대로 돌린다.

  python run_all.py                  전체 실행 (분류 → 이상탐지 0~8단계 → 테스트셋 전체 채점 → 최종 제출 파일 out/submission.xlsx)
  python run_all.py --list           실행할 단계 목록과 데이터 경로만 확인 (아무것도 실행하지 않음)
  python run_all.py --from step6     중간에 멈췄을 때 그 단계부터 다시
  python run_all.py --only classifier          분류만
  python run_all.py --skip classifier          이상탐지만

데이터: 환경 변수 NAVIS_DATA (학습 원본 csv 6개 + 사고현황 xlsx). 그 밖의 경로는 NAVIS_anomaly_final/navis_paths.py 참고.
로그: out/run_all_log.txt (화면 출력과 같은 내용)
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLS_DIR = ROOT / "NAVIS_classifier"
ANO_DIR = ROOT / "NAVIS_anomaly_final"

# (이름, 실행 폴더, 명령, 설명, 테스트 데이터 필요 여부)
STEPS = [
    ("classifier", CLS_DIR, ["run_submission.py"], "분류 모델 학습 + testA/testB 제출 파일 (약 10분)", False),
    ("features",   ANO_DIR, ["navis_features.py"], "이상탐지용 특징 추출 (끝난 달은 건너뜀)", False),
    ("step0",      ANO_DIR, ["step0_endpoints.py"], "항적 시작·끝 좌표", False),
    ("step1",      ANO_DIR, ["step1_eda.py"], "정상 기준, 모항", False),
    ("step2",      ANO_DIR, ["step2_iforest.py"], "Isolation Forest 점수", False),
    ("step3",      ANO_DIR, ["step3_seq_ae.py"], "시퀀스 오토인코더 점수", False),
    ("step4",      ANO_DIR, ["step4_combine.py"], "점수 결합", False),
    ("step5",      ANO_DIR, ["step5_mismatch.py"], "업종 불일치 (분류 모델 연결)", False),
    ("step6",      ANO_DIR, ["step6_validate.py"], "인위적 이상 검증 (20~30분)", False),
    ("step7",      ANO_DIR, ["step7_events.py"], "임계값 자동 선택 + 경보 목록", False),
    ("step8",      ANO_DIR, ["step8_review.py"], "상위 경보 20건 지도·검토표", False),
    ("full",       ANO_DIR, ["step10_full.py"], "테스트셋 전체 선박: 테스트A 업종 예측 + 테스트B 이상점수 (1~2시간)", True),
    ("submission", ROOT,    ["make_submission.py"], "최종 제출 파일 out/submission.xlsx (테스트셋 전체 선박, 제출샘플 형식)", True),
    ("dashboard",  ROOT,    ["viz/build_dashboard.py"], "대시보드 out/NAVIS_anomaly/dashboard.html, dashboard_testB.html", False),
    ("eventmap",   ROOT,    ["viz/build_event_map.py"], "사건 지도 out/NAVIS_anomaly/event_map.html", False),
]
NAMES = [s[0] for s in STEPS]


class Tee:
    """화면과 로그 파일에 같이 쓰기"""
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def check_packages(log):
    missing = []
    for mod in ["numpy", "pandas", "sklearn", "scipy", "lightgbm", "torch", "openpyxl", "pyarrow", "pyproj", "shapely"]:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        log(f"[중단] 설치되지 않은 패키지: {', '.join(missing)}")
        log(f"       pip install -r {ROOT / 'requirements.txt'}")
        sys.exit(1)


def paths_info():
    sys.path.insert(0, str(ANO_DIR))
    import navis_paths as P
    return P


def run_step(name, cwd, cmd, log):
    t0 = time.time()
    log(f"\n{'=' * 70}\n[{name}] {' '.join(cmd)}  (폴더: {cwd.name})\n{'=' * 70}")
    p = subprocess.Popen([sys.executable, "-u", *cmd], cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    for line in p.stdout:
        log(line.rstrip("\n"))
    p.wait()
    dt = time.time() - t0
    return p.returncode, dt


def main():
    ap = argparse.ArgumentParser(description="NAVIS 분류 + 이상탐지 전체 실행")
    ap.add_argument("--list", action="store_true", help="단계 목록과 경로만 보여 주고 끝냄")
    ap.add_argument("--from", dest="start", choices=NAMES, help="이 단계부터 실행")
    ap.add_argument("--only", nargs="+", choices=NAMES, help="이 단계들만 실행")
    ap.add_argument("--skip", nargs="+", choices=NAMES, default=[], help="이 단계들은 건너뜀")
    args = ap.parse_args()

    log = Tee(ROOT / "out" / "run_all_log.txt")
    log(f"\n##### NAVIS 전체 실행 시작 {time.strftime('%Y-%m-%d %H:%M:%S')} #####")
    log(f"[Python] {sys.executable}")

    P = paths_info()
    raw = P.raw_files()
    log(f"[데이터] NAVIS_DATA = {os.environ.get('NAVIS_DATA') or '(설정 안 됨)'}")
    log(f"[데이터] 학습 원본 csv {len(raw)}개: " + ", ".join(os.path.basename(f) for f in raw))
    has_test = os.path.exists(P.TEST_RAW_DIR) and os.path.exists(P.SUBMIT_SAMPLE)
    log(f"[데이터] 테스트셋 {'있음' if os.path.exists(P.TEST_RAW_DIR) else '없음'}: {P.TEST_RAW_DIR}")
    log(f"[데이터] 제출샘플 {'있음' if os.path.exists(P.SUBMIT_SAMPLE) else '없음'}: {P.SUBMIT_SAMPLE}")

    steps = STEPS
    if args.only:
        steps = [s for s in steps if s[0] in args.only]
    if args.start:
        steps = steps[[s[0] for s in steps].index(args.start):] if args.start in [s[0] for s in steps] else []
    steps = [s for s in steps if s[0] not in args.skip]
    skipped_test = [s[0] for s in steps if s[4] and not has_test]
    steps = [s for s in steps if not (s[4] and not has_test)]

    log("\n[실행 계획]")
    for name, cwd, cmd, desc, _ in steps:
        log(f"  - {name:<10} {cwd.name}/{cmd[0]:<22} {desc}")
    if skipped_test:
        log(f"  (건너뜀: {', '.join(skipped_test)} — 제출샘플.xlsx가 없어서. "
            f"NAVIS_DATA 폴더에 제출샘플.xlsx를 넣으면 실행됨)")

    if len(raw) != 6:
        log(f"\n[경고] 학습 원본 csv가 6개가 아니라 {len(raw)}개입니다. NAVIS_DATA 폴더를 확인하세요.")
        if not args.list:
            sys.exit(1)
    if args.list:
        return
    check_packages(log)

    T0 = time.time()
    done = []
    for name, cwd, cmd, desc, _ in steps:
        code, dt = run_step(name, cwd, cmd, log)
        if code != 0:
            log(f"\n[실패] {name} 단계에서 멈춤 (종료 코드 {code}, {dt / 60:.1f}분)")
            log(f"       원인을 고친 뒤 이어서 실행: python run_all.py --from {name}")
            sys.exit(code)
        done.append((name, dt))
        log(f"[완료] {name} ({dt / 60:.1f}분)")

    log(f"\n##### 전체 완료: {(time.time() - T0) / 60:.1f}분 #####")
    for name, dt in done:
        log(f"  {name:<10} {dt / 60:6.1f}분")
    log(f"\n결과 위치\n  최종 제출 파일: {ROOT / 'out' / 'submission.xlsx'}"
        f"\n  대시보드: {Path(P.OUT_DIR) / 'dashboard.html'}"
        f"\n  분류 상세: {CLS_DIR / 'results'}\n  이상탐지 상세: {P.OUT_DIR}")
    if skipped_test:
        log("  ※ 테스트셋 전체 채점과 제출 파일은 제출샘플.xlsx(또는 테스트셋)가 없어 실행하지 않았습니다.")


if __name__ == "__main__":
    main()
