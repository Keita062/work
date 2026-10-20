#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
賃金構造基本統計調査「（参考表）都道府県別…（47都道府県一覧）」を整形する。

背景（03テーマの比率補完）:
  賃金・勤続年数・超過労働時間は「全国×産業」しか無いため、
    県×産業 = 全国×産業 × (県×全産業 / 全国×全産業)
  で都道府県×産業へ按分する。本スクリプトはその分母「県×全産業」断面値を用意する。

対象表: 一般労働者・産業計・企業規模計・男女計・学歴計・年齢計 の都道府県別断面。
  各年 e-Stat「賃金構造基本統計調査 一般労働者 都道府県別」ファイルページの
  「（参考表）都道府県別きまって支給する現金給与額、所定内給与額及び
   年間賞与その他特別給与額（47都道府県一覧）」を file-download で取得。

取得メカニズム（実証済み）:
  https://www.e-stat.go.jp/stat-search/file-download?statInfId=<ID>&fileKind=<K>
  - R3〜R7 は fileKind=4 の .xlsx（PKzip, openpyxl）
  - R1・R2 は fileKind=0 の .xls（OLE2/BIFF, xlrd）
  シート「男女計」を使用。0始まり列インデックスは両形式で共通:
    都道府県=2, 年齢=4, 勤続年数=5, 所定内実労働時間数=6,
    超過実労働時間数=7, きまって支給する現金給与額=8, 所定内給与額=9,
    年間賞与=10, 労働者数=11
  ※単位: 所定内給与額=千円, 勤続年数=年, 超過実労働時間数=時間

出力: data/processed/pref_wage_allindustry.csv
  列: prefecture, 所定内給与額, 勤続年数, 超過実労働時間数, year
  行: (全国 + 47都道府県) × 取得年
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw" / "estat"
OUT = ROOT / "data" / "processed" / "pref_wage_allindustry.csv"

# statInfId は本ファイル冒頭のメモ通り。ファイル名(R#)→調査（西暦）年。
# R1=令和元年=2019 … R7=令和7年=2025
FILES = [
    ("wage_pref_R1_ref47.xls", 2019),
    ("wage_pref_R2_ref47.xls", 2020),
    ("wage_pref_R3_ref47.xlsx", 2021),
    ("wage_pref_R4_ref47.xlsx", 2022),
    ("wage_pref_R5_ref47.xlsx", 2023),
    ("wage_pref_R6_ref47.xlsx", 2024),
    ("wage_pref_R7_ref47.xlsx", 2025),
]

# 47都道府県（JIS順）。表は接尾辞なしの短縮名（青森, 東京, 大阪…）で並ぶため base→full を用意。
PREF_FULL = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
    "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
    "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
    "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
]


def _base(name: str) -> str:
    """full名 → 表の短縮名（末尾の都/府/県を落とす。北海道は道を残す）。"""
    if name == "北海道":
        return "北海道"
    return name[:-1] if name[-1] in "都府県" else name


BASE2FULL = {_base(f): f for f in PREF_FULL}
VALID_KEYS = set(BASE2FULL) | {"全国"}

# 0始まり列インデックス（男女計シート・両形式共通）
COL_PREF = 2
COL_KINZOKU = 5      # 勤続年数（年）
COL_CHOKA = 7        # 超過実労働時間数（時間）
COL_SHOTEINAI = 9    # 所定内給与額（千円）


def _norm(v) -> str:
    """セル文字列を正規化（全角/半角空白を除去）。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return re.sub(r"\s|　", "", str(v))


def parse_one(path: Path, year: int) -> pd.DataFrame:
    engine = "xlrd" if path.suffix == ".xls" else "openpyxl"
    df = pd.read_excel(path, sheet_name="男女計", header=None, engine=engine)

    rows = []
    for i in range(len(df)):
        key = _norm(df.iat[i, COL_PREF]) if df.shape[1] > COL_PREF else ""
        if key not in VALID_KEYS:
            continue
        pref = "全国" if key == "全国" else BASE2FULL[key]

        def num(c):
            if df.shape[1] <= c:
                return None
            x = df.iat[i, c]
            if pd.isna(x):
                return None
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        rows.append({
            "prefecture": pref,
            "所定内給与額": num(COL_SHOTEINAI),
            "勤続年数": num(COL_KINZOKU),
            "超過実労働時間数": num(COL_CHOKA),
            "year": year,
        })
    out = pd.DataFrame(rows).drop_duplicates(subset=["prefecture"], keep="first")
    return out


def main():
    frames = []
    for fname, year in FILES:
        p = RAW / fname
        if not p.exists():
            print(f"[skip] 未取得: {fname}")
            continue
        d = parse_one(p, year)
        n_pref = (d["prefecture"] != "全国").sum()
        has_zen = (d["prefecture"] == "全国").any()
        # 欠損チェック
        miss = {c: int(d[c].isna().sum()) for c in ["所定内給与額", "勤続年数", "超過実労働時間数"]}
        print(f"[{fname}] year={year} rows={len(d)} 全国={'有' if has_zen else '無'} "
              f"47都道府県={n_pref} 欠損={miss}")
        frames.append(d)

    if not frames:
        print("取得できたファイルが無い。")
        sys.exit(1)

    result = pd.concat(frames, ignore_index=True)
    # 並び: year昇順 → 全国を先頭 → JIS順
    order = {"全国": -1, **{f: i for i, f in enumerate(PREF_FULL)}}
    result["_o"] = result["prefecture"].map(order)
    result = result.sort_values(["year", "_o"]).drop(columns="_o").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n出力: {OUT}")
    print(f"  行数={len(result)} 年={sorted(result['year'].unique())} "
          f"都道府県数(全国除く)={result.loc[result.prefecture!='全国','prefecture'].nunique()}")
    print(result.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
