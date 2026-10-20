#!/usr/bin/env python3
"""
Phase 0（データ非依存）: 全国×業種の「真の完全失業率」を算出して
`unemployment_rate_by_industry.csv` の空欄（就業者数・労働力人口・完全失業率）を埋める。

分母（労働力人口）＝ 就業者数 ＋ 失業者数。
就業者数は国勢調査（県×産業）を全国集計して用いる（不詳補完値ベース）。
失業年 → 対応する国勢調査年（既存の census_year 列）で就業者数を代用する。

要件定義書_改訂版 §3.6。本値は distal outcome（LCA指標に含めない）。
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CENSUS = ROOT / "data" / "raw" / "census_industry_prefecture.csv"
RATE = ROOT / "data" / "processed" / "unemployment_rate_by_industry.csv"

# 産業大分類コード → 9業種（データ辞書のマッピング。190電気ガスはサービス業他。公務320/分類不能330は除外）
CODE_TO_IND = {
    "120": "農・林・漁・鉱業", "130": "農・林・漁・鉱業", "150": "農・林・漁・鉱業",
    "160": "建設業", "170": "製造業", "200": "情報通信業", "210": "運輸業",
    "220": "卸売業，小売業", "230": "金融・保険業", "240": "不動産業",
    "190": "サービス業他", "250": "サービス業他", "260": "サービス業他",
    "270": "サービス業他", "280": "サービス業他", "290": "サービス業他",
    "300": "サービス業他", "310": "サービス業他",
}
# 国勢調査年 → 就業者数の列インデックス（不詳補完値）
CENSUS_YEAR_COL = {2015: 14, 2020: 15, 2005: 10, 2010: 11}


def to_num(s):
    s = (s or "").strip().replace(",", "")
    return 0.0 if s in ("", "-") else float(s)


# --- 国勢調査を全国集計：national_emp[census_year][industry] = 就業者数合計 ---
rows = list(csv.reader(open(CENSUS, encoding="cp932")))
data = rows[15:]  # ヘッダは行15(1-indexed)、データは行16以降
national_emp = {y: {} for y in CENSUS_YEAR_COL}
for r in data:
    if len(r) <= 15:
        continue
    gender_code = r[3].strip()
    ind_code = r[6].strip()
    if gender_code != "100" or ind_code not in CODE_TO_IND:  # 総数のみ・9業種のみ
        continue
    ind = CODE_TO_IND[ind_code]
    for y, col in CENSUS_YEAR_COL.items():
        national_emp[y][ind] = national_emp[y].get(ind, 0.0) + to_num(r[col])

print("=== 国勢調査 全国集計（就業者数, 千人単位表示） ===")
for y in (2015, 2020):
    tot = sum(national_emp[y].values())
    print(f"  {y}年: 9業種合計 {tot/1e4:.0f}万人")

# --- unemployment_rate ファイルを読み、空欄を埋める ---
with open(RATE, encoding="utf-8-sig") as f:
    rate_rows = list(csv.DictReader(f))
    fields = ["year", "industry", "unemployment_count", "employment_count",
              "labor_force", "unemployment_rate", "census_year", "substitution_flag"]

# 単位を万人に統一（失業者数の原単位＝万人。就業者数は人→万人に換算）
for row in rate_rows:
    cy = int(row["census_year"])
    ind = row["industry"]
    unemp_man = float(row["unemployment_count"])          # 万人（前職産業別完全失業者数）
    emp_persons = national_emp.get(cy, {}).get(ind)        # 人（国勢調査就業者）
    if emp_persons is None:
        continue
    emp_man = emp_persons / 1e4                            # 人→万人
    lf_man = emp_man + unemp_man                           # 労働力人口（万人）
    row["employment_count"] = round(emp_man, 1)            # 万人
    row["labor_force"] = round(lf_man, 1)                  # 万人
    row["unemployment_rate"] = round(unemp_man / lf_man * 100, 3) if lf_man > 0 else ""

with open(RATE, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in rate_rows:
        w.writerow({k: row.get(k, "") for k in fields})

print(f"\n✓ 保存: {RATE.name}")
print("  ※ 就業者数・労働力人口・失業者数はすべて『万人』単位")
print("\n=== サンプル（2020年・業種別完全失業率） ===")
for row in rate_rows:
    if row["year"] == "2020":
        print(f"  {row['industry']}: 失業{row['unemployment_count']}万人 / 労働力{row['labor_force']}万人"
              f" → {row['unemployment_rate']}%")
