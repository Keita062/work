#!/usr/bin/env python3
"""
Phase 1: パネル構築（都道府県×業種×年）

ターゲット3業種（製造業／卸売業，小売業／サービス業他）について、取得済みデータを
結合し、構成概念別の観測変数を都道府県×年で組み立てる。

- 賃金・勤続・超過労働は「全国×産業」しか無いため比率補完:
    県×産業 = 全国×産業(industry_master) × 県全産業(pref_wage) / 全国全産業(pref_wage 全国)
- 国勢調査系（女性比率・就業シェア・高齢/若年比率）は県×産業の実データ（2020）。
- 離職率・入職率は県×産業（2016）。有効求人倍率・人口密度・高齢化率は県（共変量）。
- 完全失業率（県, distal）はフォールバック（国勢調査は失業者を産業別に分類しないため）。

出力:
  panel_2020.csv    … 2020年断面（LCA本体用, 47県×3業種）
  panel_annual.csv  … 2019-2025 年次（LTA用, 賃金系のみ時変・他は保持）

実行: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/preprocess/build_panel.py
"""
import csv
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

TARGETS = ["製造業", "卸売業，小売業", "サービス業他"]
IND_CODES = {  # 産業大分類コード → ターゲット業種
    "170": "製造業", "220": "卸売業，小売業",
    "190": "サービス業他", "250": "サービス業他", "260": "サービス業他",
    "270": "サービス業他", "280": "サービス業他", "290": "サービス業他",
    "300": "サービス業他", "310": "サービス業他",
}


def num(s):
    s = (s or "").strip().replace(",", "")
    return 0.0 if s in ("", "-") else float(s)


# ============================================================
# 1) 国勢調査2020: 女性比率・就業シェア（県×業種）
# ============================================================
rows = list(csv.reader(open(RAW / "census_industry_prefecture.csv", encoding="cp932")))[15:]
# 集計: pref -> {(ind, gender): emp2020}, pref -> total(総数,全産業)
emp = {}          # (pref, ind, gender) -> 就業者数
pref_total = {}   # pref -> 全産業総数(コード100, 性別総数)
for r in rows:
    if len(r) <= 15:
        continue
    pref, gcode, icode = r[2].strip(), r[3].strip(), r[6].strip()
    v = num(r[15])  # 2020年_不詳補完値
    if icode == "100" and gcode == "100":
        pref_total[pref] = v
    ind = IND_CODES.get(icode)
    if ind is None:
        continue
    for g in ("100", "110", "120"):
        pass
    key = (pref, ind, gcode)
    emp[key] = emp.get(key, 0.0) + v

prefs = sorted(pref_total)
cen_rows = []
for pref in prefs:
    for ind in TARGETS:
        tot = emp.get((pref, ind, "100"), 0.0)
        fem = emp.get((pref, ind, "120"), 0.0)
        cen_rows.append({
            "prefecture": pref, "industry": ind,
            "女性比率": round(fem / tot * 100, 3) if tot else None,
            "就業シェア": round(tot / pref_total[pref] * 100, 3) if pref_total[pref] else None,
        })
df_cen = pd.DataFrame(cen_rows)

# ============================================================
# 2) 年齢構成（県×業種, 2020）・離職/入職（2016）・共変量（県）
# ============================================================
df_age = pd.read_csv(PROC / "census_age_industry_pref.csv", comment="#")[
    ["prefecture", "industry", "高齢就業者比率", "若年就業者比率"]]
df_turn = pd.read_csv(PROC / "pref_industry_turnover.csv", comment="#")[
    ["prefecture", "industry", "離職率", "入職率"]]
df_cov = pd.read_csv(PROC / "pref_covariates.csv", comment="#")
df_cov = df_cov[df_cov["prefecture"] != "全国"][
    ["prefecture", "有効求人倍率", "人口密度", "高齢化率", "完全失業率"]]

# ============================================================
# 3) 賃金・勤続・超過労働の比率補完（県×業種×年）
# ============================================================
nat = pd.read_csv(PROC / "industry_master.csv")  # 全国×業種
wage = pd.read_csv(PROC / "pref_wage_allindustry.csv", comment="#")  # 県全産業（+全国）
WAGE_VARS = ["所定内給与額", "勤続年数", "超過実労働時間数"]


def impute_wage(year):
    naty = nat[nat.year == year][["industry"] + WAGE_VARS]
    naty = naty[naty["industry"].isin(TARGETS)]
    wy = wage[wage.year == year]
    nat_all = wy[wy.prefecture == "全国"][WAGE_VARS].iloc[0]
    wp = wy[wy.prefecture != "全国"]
    out = []
    for _, pr in wp.iterrows():
        for _, ni in naty.iterrows():
            rec = {"prefecture": pr["prefecture"], "industry": ni["industry"], "year": year}
            for v in WAGE_VARS:
                factor = pr[v] / nat_all[v] if nat_all[v] else 1.0
                rec[v] = round(ni[v] * factor, 3)  # 比率補完
            out.append(rec)
    return pd.DataFrame(out)


# ============================================================
# 4) 断面パネル(2020) と 年次パネル(2019-2025)
# ============================================================
def assemble(year, wageyear):
    w = impute_wage(wageyear)
    p = (w.merge(df_cen, on=["prefecture", "industry"], how="left")
          .merge(df_age, on=["prefecture", "industry"], how="left")
          .merge(df_turn, on=["prefecture", "industry"], how="left")
          .merge(df_cov, on="prefecture", how="left"))
    p["year"] = year
    p["wage_year"] = wageyear
    return p


panel_2020 = assemble(2020, 2020)
panel_2020.to_csv(PROC / "panel_2020.csv", index=False, encoding="utf-8")

annual = pd.concat([assemble(y, y) for y in range(2019, 2026)], ignore_index=True)
annual.to_csv(PROC / "panel_annual.csv", index=False, encoding="utf-8")

# ============================================================
# 5) サマリ
# ============================================================
ind_vars = ["所定内給与額", "勤続年数", "超過実労働時間数", "女性比率", "就業シェア",
            "高齢就業者比率", "若年就業者比率", "離職率", "入職率", "有効求人倍率"]
print("=== panel_2020.csv ===")
print(f"行数 {len(panel_2020)}（{panel_2020.prefecture.nunique()}県 × {panel_2020.industry.nunique()}業種）")
print("観測変数と欠損率:")
for v in ind_vars:
    miss = panel_2020[v].isna().mean() * 100
    print(f"  {v:12s} 欠損 {miss:4.1f}%  例(製造業/北海道)="
          f"{panel_2020[(panel_2020.industry=='製造業')&(panel_2020.prefecture=='北海道')][v].values}")
print(f"\n=== panel_annual.csv === 行数 {len(annual)}（年 {sorted(annual.year.unique())}）")
print("✓ 保存: panel_2020.csv, panel_annual.csv")
