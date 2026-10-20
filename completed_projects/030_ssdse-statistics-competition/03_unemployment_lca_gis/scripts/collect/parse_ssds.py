#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
社会生活統計指標－都道府県－ から都道府県レベルの共変量／distal フォールバックを整形する。

入力（data/raw/estat/、いずれも e-Stat の BIFF .xls / OLE2、engine='xlrd' で読む）:
  - ssds_a001_jinko.xls   (A 人口・世帯,  statInfId=000040133615, sheet 'A')
  - ssds_a006_roudou.xls  (F 労働,        statInfId=000040133620, sheet 'F')
  - ssds_b001_shizen.xls  (B 自然環境,    statInfId=000040133616, sheet 'B')

各シートの構造（実証済み）:
  - 都道府県名は列8。データ行は 11=全国, 12=北海道, ... , 58=沖縄県（全国＋47県＝48行）。
  - ヘッダ帯: 行6='ID:xxxx', 行7=指標名(日), 行9=指標コード(F/A/B…)＋単位, 行10=参照年。
  - 各指標は「3列」に横並びで、3列それぞれが別の参照年（例: 2010/2015/2020 や 2015/2020/2022）。
    行9の指標コードは3列のうち先頭列にのみ入る。行10の参照年は3列すべてに入る。
    → (コード, 年) を指定すると一意の値列が決まる。

算出:
  完全失業率  = 完全失業者数 / (就業者数 + 完全失業者数) × 100        （2020 国勢調査ベース）
  有効求人倍率 = 月間有効求人数(一般) / 月間有効求職者数(一般)        （2022 年度・職業安定業務統計）
  人口密度    = 総人口 / 総面積(北方地域及び竹島を除く)               （2020, 人/km²）
  高齢化率    = 65歳以上人口 / 総人口 × 100                          （2020 国勢調査ベース）

出力: data/processed/pref_covariates.csv（UTF-8, 冒頭にソース注記コメント行）。

実行: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/collect/parse_ssds.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw" / "estat"
OUT = ROOT / "data" / "processed" / "pref_covariates.csv"

CODE_ROW = 9      # 指標コード（先頭列のみ）
YEAR_ROW = 10     # 参照年（3列すべて）
PREF_COL = 8      # 都道府県名
DATA_ROW0 = 11    # 全国
DATA_ROW1 = 59    # 沖縄県の次（range 排他）


def load(fname: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(RAW / fname, sheet_name=sheet, header=None, engine="xlrd")


def find_col(df: pd.DataFrame, code: str, year: int) -> int:
    """指標コード code の3列ブロックの中から、参照年 year の値列インデックスを返す。"""
    base = None
    for c in range(df.shape[1]):
        v = df.iat[CODE_ROW, c]
        if isinstance(v, str) and v.strip() == code:
            base = c
            break
    if base is None:
        raise KeyError(f"コード {code} が {YEAR_ROW=} 付近に見つかりません")
    for c in (base, base + 1, base + 2):
        yv = df.iat[YEAR_ROW, c]
        if pd.notna(yv) and int(float(yv)) == year:
            return c
    raise KeyError(f"コード {code} に参照年 {year} の列がありません "
                   f"(候補年: {[df.iat[YEAR_ROW, base+i] for i in range(3)]})")


def series(df: pd.DataFrame, code: str, year: int) -> pd.Series:
    c = find_col(df, code, year)
    prefs = [str(df.iat[r, PREF_COL]).strip() for r in range(DATA_ROW0, DATA_ROW1)]
    vals = [pd.to_numeric(df.iat[r, c], errors="coerce") for r in range(DATA_ROW0, DATA_ROW1)]
    return pd.Series(vals, index=prefs, name=code)


def main() -> None:
    F = load("ssds_a006_roudou.xls", "F")
    A = load("ssds_a001_jinko.xls", "A")
    B = load("ssds_b001_shizen.xls", "B")

    # --- 各原指標の値（都道府県名 index）---
    employed  = series(F, "F1102", 2020)   # 就業者数
    unemploy  = series(F, "F1107", 2020)   # 完全失業者数
    seekers   = series(F, "F3102", 2022)   # 月間有効求職者数（一般）
    openings  = series(F, "F3103", 2022)   # 月間有効求人数（一般）
    pop_total = series(A, "A1101", 2020)   # 総人口
    pop_65    = series(A, "A1303", 2020)   # 65歳以上人口
    area      = series(B, "B1101", 2020)   # 総面積（北方地域及び竹島を除く）km²

    # index（都道府県名）が3ファイルで一致していることを確認
    assert list(employed.index) == list(pop_total.index) == list(area.index), "都道府県index不一致"

    out = pd.DataFrame(index=employed.index)
    out.index.name = "prefecture"
    out["完全失業率"]  = (unemploy / (employed + unemploy) * 100).round(2)
    out["有効求人倍率"] = (openings / seekers).round(3)
    out["人口密度"]    = (pop_total / area).round(1)
    out["高齢化率"]    = (pop_65 / pop_total * 100).round(2)
    out["ref_year_note"] = ("失業率=2020国勢調査/求人倍率=2022年度/"
                            "人口密度=2020(総人口÷総面積)/高齢化率=2020国勢調査")

    header = [
        "# 社会生活統計指標－都道府県－ 由来の都道府県レベル共変量（distal フォールバック用）",
        "# 完全失業率(%)   = F1107 完全失業者数 / (F1102 就業者数 + F1107) *100  [2020 国勢調査]",
        "# 有効求人倍率     = F3103 月間有効求人数(一般) / F3102 月間有効求職者数(一般)  [2022 年度・職業安定業務統計]",
        "# 人口密度(人/km2) = A1101 総人口 / B1101 総面積(北方地域及び竹島を除く)  [2020]",
        "# 高齢化率(%)      = A1303 65歳以上人口 / A1101 総人口 *100  [2020 国勢調査]",
        "# 出典: e-Stat 社会生活統計指標 A(統計ID:000040133615)/F(000040133620)/B(000040133616)",
        "# 行: 全国＋47都道府県（48行）。空欄は原データ欠損。",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        out.reset_index().to_csv(f, index=False)

    # --- 検証出力 ---
    print(f"出力: {OUT}")
    print(f"行数: {len(out)}（全国含む）, 都道府県数: {len(out) - 1}")
    print(f"欠損セル数: {int(out.drop(columns='ref_year_note').isna().sum().sum())}")
    print("\n=== 全国 & 抜粋 ===")
    show = out.drop(columns="ref_year_note")
    print(show.loc[["全国", "東京都", "北海道", "沖縄県", "秋田県"]].to_string())
    print("\n=== 各指標レンジ（都道府県のみ, 全国除く）===")
    pref_only = show.drop(index="全国")
    for col in pref_only.columns:
        s = pref_only[col]
        print(f"  {col}: min={s.min()} ({s.idxmin()}) / max={s.max()} ({s.idxmax()}) / mean={s.mean():.2f}")


if __name__ == "__main__":
    main()
