#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雇用動向調査（平成28年 / 2016）都道府県別表の整形。

入力（data/raw/estat/、いずれも e-Stat file-download fileKind=0 の BIFF .xls）:
  - koyoudoukou_table5_rishoku.xls  … 都道府県別表５ 離職率  (statInfId=000031612648)
  - koyoudoukou_table2_nyushoku.xls … 都道府県別表２ 入職率  (statInfId=000031612645)
  ※ 両表は同一調査（2016年）・同一レイアウト。表4=離職者数, 表1=入職者数（率ではない）。

シート構造（表2・表5 共通, shape=(88,51)）:
  - 行5: 都道府県ヘッダ。列3='計'(全国), 列4..50 = 47都道府県（北海道..沖縄）。
  - 性別×企業規模ブロックが縦に3つ: 計（男女計）→ 計（男）→ 計（女）。
    各ブロックは 'ＧＴ 計（男女計）' 行（=全産業総数）＋企業規模別5行、続いて
    産業大分類 C..R の行（列1に全角の産業記号, 列2に産業名）が並ぶ。
    産業行のうち一部（E製造業）は直下に企業規模別サブ行を持つが、それらは列1が空。
  - よって「企業規模ＧＴ×性別＝計（男女計）」の各産業大分類の率＝
    計（男女計）ブロック内で列1が産業記号(C..R)の行の値。

産業大分類→9業種マッピング（本プロジェクト定義）:
  農・林・漁・鉱業 = A農業林業 + B漁業 + C鉱業
    → 雇用動向調査は A/B を調査対象外（Cから開始）。本表では C 鉱業のみで代表する。
  建設業=D, 製造業=E, 情報通信業=G, 運輸業=H, 卸売業，小売業=I,
  金融・保険業=J, 不動産業=K,
  サービス業他 = F電気ガス + L学術 + M宿泊飲食 + N生活娯楽 + O教育 + P医療福祉 + Q複合 + Rサービス他
  ターゲット3業種 = 製造業(E) / 卸売業，小売業(I) / サービス業他(集約)。

集約方法:
  - 率(離職率・入職率)は業種内の複数大分類を単純平均で集約する。
    （雇用動向調査の都道府県別表には常用労働者数ウェイトが同一表に無いため、
      従業者数加重ではなく単純平均を採用。欠損('-')は平均から除外。）
  - 農・林・漁・鉱業は C 単独のため実質そのままの値。

出力: data/processed/pref_industry_turnover.csv
  列: prefecture, industry(9業種名), 離職率, 入職率, year  / UTF-8, long形式, year=2016
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw" / "estat"
OUT = ROOT / "data" / "processed" / "pref_industry_turnover.csv"
YEAR = 2016

RISHOKU = RAW / "koyoudoukou_table5_rishoku.xls"
NYUSHOKU = RAW / "koyoudoukou_table2_nyushoku.xls"

# 雇用動向調査の都道府県別表は都道府県名を短縮表記（青森・東京・京都…）で持つ。
# 本プロジェクトの既存 processed データ（pref_covariates.csv 等）はフル表記
# （青森県・東京都・京都府・北海道…）なので、それに揃えるための正規化。
#   北海道 → そのまま / 東京 → 東京都 / 大阪・京都 → ○府 / それ以外 → ○県
PREF_SUFFIX = {"東京": "都", "大阪": "府", "京都": "府"}


def normalize_pref(name):
    """短縮都道府県名をフル表記へ。北海道はそのまま。"""
    name = name.strip()
    if name == "北海道":
        return "北海道"
    if name in PREF_SUFFIX:
        return name + PREF_SUFFIX[name]
    return name + "県"


# 全角の産業記号 C..R（雇用動向調査の都道府県別表で使われる大分類）
IND_LETTERS = list("ＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲ")

# 大分類記号 → 9業種名
NINE = {
    "Ｃ": "農・林・漁・鉱業",
    "Ｄ": "建設業",
    "Ｅ": "製造業",
    "Ｆ": "サービス業他",
    "Ｇ": "情報通信業",
    "Ｈ": "運輸業",
    "Ｉ": "卸売業，小売業",
    "Ｊ": "金融・保険業",
    "Ｋ": "不動産業",
    "Ｌ": "サービス業他",
    "Ｍ": "サービス業他",
    "Ｎ": "サービス業他",
    "Ｏ": "サービス業他",
    "Ｐ": "サービス業他",
    "Ｑ": "サービス業他",
    "Ｒ": "サービス業他",
}

# 出力する9業種の並び（産業大分類順）
NINE_ORDER = [
    "農・林・漁・鉱業", "建設業", "製造業", "情報通信業", "運輸業",
    "卸売業，小売業", "金融・保険業", "不動産業", "サービス業他",
]


def to_num(v):
    """数値セルは float、'-' や文字列・欠損は NaN。"""
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
        return float(v)
    return np.nan


def extract_rates(path, sheet):
    """計（男女計）ブロックから {(prefecture, letter): rate} と都道府県順リストを返す。"""
    df = pd.read_excel(path, engine="xlrd", sheet_name=sheet, header=None)

    # 都道府県ヘッダ（行5, 列4..末尾。列3='計'(全国)は除外）
    # 短縮表記→フル表記へ正規化して既存 processed データと揃える。
    pref_cols = []
    for c in range(4, df.shape[1]):
        h = df.iat[5, c]
        if isinstance(h, str) and h.strip():
            pref_cols.append((c, normalize_pref(h)))

    # 計（男女計）ブロックの範囲を特定：col1=='ＧＴ' の出現で区切る
    gt_rows = [r for r in range(df.shape[0])
               if isinstance(df.iat[r, 1], str) and df.iat[r, 1].strip() == "ＧＴ"]
    # 先頭が男女計、次が男 … 男女計ブロック = [gt_rows[0], gt_rows[1])
    start = gt_rows[0]
    end = gt_rows[1] if len(gt_rows) > 1 else df.shape[0]
    # 念のため先頭行が男女計か確認
    assert isinstance(df.iat[start, 2], str) and "男女計" in df.iat[start, 2], \
        f"想定した男女計ブロック開始行ではない: {df.iat[start,2]!r}"

    records = {}
    for r in range(start, end):
        c1 = df.iat[r, 1]
        if isinstance(c1, str) and c1.strip() in IND_LETTERS:
            letter = c1.strip()
            for c, pname in pref_cols:
                records[(pname, letter)] = to_num(df.iat[r, c])

    pref_order = [p for _, p in pref_cols]
    return records, pref_order


def aggregate(records, pref_order):
    """(pref, letter) の率を9業種へ単純平均で集約 → {(pref, nine): rate}。"""
    out = {}
    for pref in pref_order:
        for nine in NINE_ORDER:
            letters = [L for L in IND_LETTERS if NINE[L] == nine]
            vals = [records.get((pref, L), np.nan) for L in letters]
            vals = [v for v in vals if not np.isnan(v)]
            out[(pref, nine)] = round(float(np.mean(vals)), 2) if vals else np.nan
    return out


def main():
    ris, pref_order = extract_rates(RISHOKU, "都道府県別表５")
    nyu, pref_order2 = extract_rates(NYUSHOKU, "都道府県別表２")
    assert pref_order == pref_order2, "離職率と入職率で都道府県の並びが不一致"

    ris_agg = aggregate(ris, pref_order)
    nyu_agg = aggregate(nyu, pref_order)

    rows = []
    for pref in pref_order:
        for nine in NINE_ORDER:
            rows.append({
                "prefecture": pref,
                "industry": nine,
                "離職率": ris_agg[(pref, nine)],
                "入職率": nyu_agg[(pref, nine)],
                "year": YEAR,
            })
    df = pd.DataFrame(rows, columns=["prefecture", "industry", "離職率", "入職率", "year"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8", na_rep="")

    # レポート用サマリ
    n = len(df)
    npref = df["prefecture"].nunique()
    nind = df["industry"].nunique()
    miss_ris = int(df["離職率"].isna().sum())
    miss_nyu = int(df["入職率"].isna().sum())
    print(f"OUT: {OUT}")
    print(f"rows={n}  prefectures={npref}  industries={nind}  year={YEAR}")
    print(f"missing 離職率={miss_ris}  missing 入職率={miss_nyu}")
    print("industries:", list(df['industry'].unique()))
    print("\n--- 全国(計)は含めず47都道府県のみ。先頭12行プレビュー ---")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
