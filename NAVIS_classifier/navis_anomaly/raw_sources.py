"""원본 데이터 찾기: csv / zip (여러 csv, 하위 폴더, zip 안의 zip) → 월별 csv를 하나씩 꺼냄"""
import io
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

USECOLS = ["part_dt", "rfid_id", "rfid_time", "geom", "fishing_code"]
REQUIRED = ["part_dt", "rfid_id", "rfid_time", "geom"]          # fishing_code는 테스트 데이터엔 없을 수 있음


def _fix_name(n: str) -> str:
    """윈도우에서 만든 zip의 한글 파일명 깨짐 복구"""
    try:
        raw = n.encode("cp437")
    except UnicodeEncodeError:
        return n                      # 이미 정상적으로 읽힌 이름
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return n


def _header_ok(opener) -> bool:
    """V-Pass 원본 csv인지 첫 줄로 확인 (코드표·설명 csv 등은 건너뜀)"""
    with opener() as f:
        line = f.readline()
    for enc in ("utf-8-sig", "cp949"):
        try:
            head = line.decode(enc)
            return all(c in head for c in REQUIRED)
        except UnicodeDecodeError:
            continue
    return False


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    깨진 행 제거: 일부 월 파일(예: 6월)은 geom이 중간에 잘려 줄이 밀린 행이 섞여 있음
      → part_dt가 8자리 날짜, rfid_time이 0~23, geom이 MULTILINESTRING(...)인 행만 사용
    """
    n0 = len(df)
    pdt = df["part_dt"].astype(str).str.strip()
    t = pd.to_numeric(df["rfid_time"], errors="coerce")
    g = df["geom"].astype(str)
    ok = pdt.str.fullmatch(r"\d{8}") & t.between(0, 23) & g.str.startswith("MULTILINESTRING") & g.str.rstrip().str.endswith("))")
    ok &= ~g.str.contains(r"[\r\n]", regex=True)          # 줄바꿈이 끼어 잘린 geom 제외
    ok &= pd.to_datetime(pdt.where(ok), format="%Y%m%d", errors="coerce").notna()
    df = df[ok].copy()
    df["part_dt"] = pdt[ok]
    df["rfid_time"] = t[ok].astype(int)
    df["rfid_id"] = df["rfid_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if "fishing_code" in df.columns:
        df["fishing_code"] = df["fishing_code"].astype(str).str.strip()
    if n0 - len(df):
        print(f"[정리] 형식이 깨진 행 {n0 - len(df):,}개 제거 (전체 {n0:,}행)")
    return df


def _read(opener) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            with opener() as f:
                df = pd.read_csv(f, encoding=enc, dtype=str, usecols=lambda c: c in USECOLS,
                                 on_bad_lines="skip", engine="c")
            return _clean(df)
        except UnicodeDecodeError:
            continue
    raise ValueError("인코딩 확인 필요")


def _iter_zip(z: zipfile.ZipFile, prefix: str):
    for n in sorted(z.namelist(), key=_fix_name):
        label = f"{prefix}/{_fix_name(n)}"
        if _in_test_dir(str(Path(_fix_name(n)).parent)):         # zip 안의 테스트셋 폴더
            if n.lower().endswith((".csv", ".zip")):
                print(f"[건너뜀] 테스트셋 폴더: {label}")
            continue
        if n.lower().endswith(".csv"):
            opener = (lambda z=z, n=n: z.open(n))
            if _header_ok(opener):
                yield label, opener
            else:
                print(f"[건너뜀] V-Pass 형식 아님: {label}")
        elif n.lower().endswith(".zip"):          # zip 안의 zip (월별 압축)
            inner = zipfile.ZipFile(io.BytesIO(z.read(n)))
            yield from _iter_zip(inner, label)


def _in_test_dir(rel_dir: str) -> bool:
    """하위 폴더 경로에 '테스트'가 들어 있으면 테스트셋 폴더 (Mac zip 한글 파일명(NFD)도 맞춰 비교)"""
    return "테스트" in unicodedata.normalize("NFC", rel_dir)


def expand_paths(paths, exts=(".zip", ".csv", ".xlsx", ".xls"), skip_test_dirs=False):
    """경로 목록 → 실제 파일 목록. 파일 / 폴더(하위 폴더 포함) / 와일드카드(*) 모두 허용, 상대경로 가능
    skip_test_dirs=True: 폴더를 뒤질 때 경로에 '테스트'가 든 하위 폴더(테스트셋)는 건너뜀.
                         파일을 직접 지정한 경우는 그대로 사용 (이상탐지 9단계가 테스트 csv를 직접 넘김)"""
    import glob
    out = []
    for p in paths:
        p = str(Path(p).expanduser())
        hits = glob.glob(p, recursive=True) if any(c in p for c in "*?[") else [p]
        for h in hits:
            hp = Path(h)
            if hp.is_dir():
                found = [f for f in sorted(hp.rglob("*")) if f.is_file() and f.suffix.lower() in exts]
                if skip_test_dirs:
                    test = [f for f in found if _in_test_dir(str(f.parent.relative_to(hp)))]
                    if test:
                        print(f"[건너뜀] 테스트셋 폴더의 파일 {len(test)}개 (학습·사고 매칭에 쓰지 않음): "
                              f"{unicodedata.normalize('NFC', str(test[0].parent.relative_to(hp)))}")
                    found = [f for f in found if f not in test]
                out += [str(f) for f in found]
            else:
                out.append(str(hp))
    return list(dict.fromkeys(out))


def default_data_paths():
    """--data 를 안 줬을 때 찾는 위치: 환경변수 NAVIS_DATA → ./data → 이 프로그램 옆 data 폴더 → 현재 폴더"""
    import os
    here = Path(__file__).resolve().parent.parent
    cands = [os.environ.get("NAVIS_DATA"), Path.cwd() / "data", here / "data", Path.cwd()]
    for c in cands:
        if c and Path(c).exists() and expand_paths([str(c)], (".zip", ".csv")):
            return [str(c)]
    return []


def iter_raw_sources(paths):
    """csv / zip(여러 csv, zip 안의 zip 포함)을 월별 csv 단위로 하나씩 꺼냄 → 메모리 절약"""
    for p in expand_paths(paths, (".zip", ".csv"), skip_test_dirs=True):
        if p.lower().endswith(".zip"):
            yield from _iter_zip(zipfile.ZipFile(p), Path(p).name)
        else:
            opener = (lambda p=p: open(p, "rb"))
            if _header_ok(opener):
                yield Path(p).name, opener
            else:
                print(f"[건너뜀] V-Pass 형식 아님: {p}")




def read_source(opener) -> pd.DataFrame:
    return _read(opener)


def iter_xlsx(paths):
    """zip 안/밖의 xlsx (테스트 대상 목록 등)"""
    for p in expand_paths(paths, (".zip", ".xlsx", ".xls")):
        if p.lower().endswith(".zip"):
            z = zipfile.ZipFile(p)
            for n in z.namelist():
                if n.lower().endswith((".xlsx", ".xls")):
                    yield _fix_name(n), (lambda z=z, n=n: io.BytesIO(z.read(n)))
        elif p.lower().endswith((".xlsx", ".xls")):
            yield Path(p).name, (lambda p=p: open(p, "rb"))
