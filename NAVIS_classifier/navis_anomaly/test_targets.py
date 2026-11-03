"""
테스트 대상 엑셀 읽기 (26년 1~6월 어선 사고현황(테스트 데이터).xlsx)
  → A: testA 대상 선박 (조업형태 분류)
  → B: testB 대상 선박 (+ 사고 날짜가 있으면 target_date)

시트/컬럼 이름이 바뀌어도 최대한 알아서 찾도록:
  - 선박 ID: 컬럼명에 'rfid' 또는 '선박'이 들어가거나, 8자리 숫자가 대부분인 컬럼
  - 날짜: 컬럼명에 '일자/일시/날짜/date/시각/발생' 이 들어가고 날짜로 읽히는 컬럼
  - A/B 구분: 시트 이름에 'A'/'B' (testA/testB) → 없으면 '구분' 컬럼 값 → 그래도 없으면 날짜 없는 쪽 A, 있는 쪽 B
"""
import re

import numpy as np
import pandas as pd


def _find_id_col(d: pd.DataFrame):
    for c in d.columns:
        if re.search(r"rfid|선박\s*(id|번호)", str(c), re.I):
            return c
    for c in d.columns:
        s = d[c].dropna().astype(str).str.replace(r"\.0$", "", regex=True)
        if len(s) and s.str.fullmatch(r"\d{8}").mean() > 0.8:
            return c
    return None


def _find_date_col(d: pd.DataFrame):
    for c in d.columns:
        if re.search(r"일자|일시|날짜|date|시각|발생", str(c), re.I):
            v = pd.to_datetime(d[c].astype(str).str.replace(r"\.0$", "", regex=True).str[:8],
                               format="%Y%m%d", errors="coerce")
            if v.notna().mean() < 0.5:
                v = pd.to_datetime(d[c], errors="coerce")
            if v.notna().mean() > 0.5:
                return c, v.dt.normalize()
    return None, None


def _clean_ids(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _as_accidents(d: pd.DataFrame):
    """rfid 없이 사고 발생일시·위경도가 있는 '사고현황' 형식이면 사고 목록으로 변환"""
    tc = next((c for c in d.columns if re.search(r"발생일시|일시|datetime", str(c))), None)
    la = next((c for c in d.columns if re.search(r"위도|lat", str(c), re.I)), None)
    lo = next((c for c in d.columns if re.search(r"경도|lon", str(c), re.I)), None)
    if not (tc and la and lo):
        return None
    lic = next((c for c in d.columns if re.search(r"어업면허|업종|fishing", str(c))), None)
    acc = pd.DataFrame({
        "acc_id": np.arange(len(d)),
        "acc_time": pd.to_datetime(d[tc], errors="coerce").values,
        "lat": pd.to_numeric(d[la], errors="coerce").values,
        "lon": pd.to_numeric(d[lo], errors="coerce").values,
        "license": d[lic].astype(str).str.strip().values if lic else "",
    })
    for c in d.columns:                       # 원본 컬럼도 같이 보관 (발생유형 등 분석용)
        if c not in (tc, la, lo):
            acc[f"원본_{c}"] = d[c].values
    return acc.dropna(subset=["acc_time", "lat", "lon"]).reset_index(drop=True)


def load_targets(xlsx_sources) -> dict:
    out = {"A": None, "B": None, "raw": {}, "accidents": None}
    found = []
    for name, opener in xlsx_sources:
        book = pd.read_excel(opener(), sheet_name=None)
        for sh, d in book.items():
            d = d.dropna(how="all")
            if d.empty:
                continue
            # 머리글이 첫 행이 아닌 경우(제목 행 등) 처리: rfid가 들어간 행을 머리글로
            if _find_id_col(d) is None:
                for i in range(min(10, len(d))):
                    if d.iloc[i].astype(str).str.contains("rfid", case=False).any():
                        d.columns = d.iloc[i].astype(str).values
                        d = d.iloc[i + 1:]
                        break
            idc = _find_id_col(d)
            if idc is None:
                acc = _as_accidents(d)
                if acc is not None:
                    out["accidents"] = acc if out["accidents"] is None else pd.concat([out["accidents"], acc])
                    print(f"[테스트] {name}/{sh}: 사고 목록 {len(acc):,}건 (rfid 없음 → 위치·시각으로 선박 매칭)")
                else:
                    print(f"[테스트] {name}/{sh}: 선박 ID 컬럼을 못 찾음 → 건너뜀")
                continue
            dc, dates = _find_date_col(d)
            t = pd.DataFrame({"rfid_id": _clean_ids(d[idc])})
            if dates is not None:
                t["target_date"] = dates.values
            # '구분' 컬럼이 있으면 그 값으로 A/B 분리
            gcol = next((c for c in d.columns if re.search(r"구분|test|유형", str(c), re.I)
                         and d[c].astype(str).str.contains(r"[AB]$|test", case=False).any()), None)
            out["raw"][f"{name}/{sh}"] = d
            found.append((str(sh), t, gcol, d))
            print(f"[테스트] {name}/{sh}: {len(t):,}행, ID='{idc}', 날짜='{dc}', 구분='{gcol}'")
    for sh, t, gcol, d in found:
        key = "A" if re.search(r"A\b|testA|_A", sh, re.I) else "B" if re.search(r"B\b|testB|_B", sh, re.I) else None
        if gcol is not None and key is None:
            g = d[gcol].astype(str).values
            a = t[np.char.endswith(g.astype(str), "A")]
            b = t[np.char.endswith(g.astype(str), "B")]
            out["A"] = a if len(a) else out["A"]
            out["B"] = b if len(b) else out["B"]
            continue
        if key is None:
            key = "B" if "target_date" in t.columns else "A"
            if out[key] is not None:
                key = "B" if key == "A" else "A"
        out[key] = t if out[key] is None else pd.concat([out[key], t], ignore_index=True)
    if out["A"] is None and out["B"] is not None:
        # 사고 선박 목록만 있으면: 같은 선박들을 testA(분류)·testB(이상점수) 둘 다에 사용
        out["A"] = out["B"][["rfid_id"]].drop_duplicates()
        print("[테스트] testA 시트가 따로 없어 testB 선박 목록으로 분류도 수행")
    if out["B"] is None and out["A"] is not None:
        out["B"] = out["A"].copy()
        print("[테스트] testB 시트가 따로 없어 testA 선박 목록으로 이상점수도 계산")
    return out
