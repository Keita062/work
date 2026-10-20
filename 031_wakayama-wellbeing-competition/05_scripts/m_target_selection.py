"""ステップ2：ターゲット市区町村を決める（記述統計パート）

和歌山6市について、幸福度・生活満足度・主観24分野の記述統計を出す。
どの市を分析のターゲットに据えるかを、数字の側から判断するための材料。

  1. データ量        … 年度数・回答者数（何年分・何人の回答で語れるか）
  2. 水準            … 幸福度/生活満足度/主観24分野平均の市別平均と、全国678地域の中での順位
  3. ばらつき・推移  … 年度ごとの値と、最初の年度から最後の年度への変化
  4. 分野別          … 24分野それぞれの市別平均と全国パーセンタイル
  5. 寄与上位5分野   … ステップ3で効き目が大きいと分かった5分野に絞った市別の並び

入力: 04_data/digital_wellbeing_shikuchoson_2022_2026.xlsx
出力: 04_data/processed/target_selection.json
"""
import json

import numpy as np
import pandas as pd

from wb_common import PROC, WAKAYAMA, categories, load_wb, m0_sample

# ステップ3（旧ステップ2）で寄与の大きかった5分野
TOP5 = ["健康状態", "自己効力感", "住宅環境", "公共空間", "地域とのつながり"]

subj, obj = load_wb()
CATS = categories(subj)
S = [f"主観_{c}" for c in CATS]
m = m0_sample(subj, obj)
m["主観24分野平均"] = m[S].mean(axis=1)

wk = m[m["都道府県"] == "和歌山県"].copy()
res = {"対象行数": {"全国": len(m), "和歌山": len(wk)}}


def pct_rank(col, v):
    """全国678地域×年度 2,116行の中で、値 v が下から何%の位置か"""
    return round(float((m[col] < v).mean() * 100), 1)


# ---------------------------------------------------------------- 1. データ量
vol = []
for c in WAKAYAMA:
    g = wk[wk["市区町村"] == c]
    vol.append({
        "市": c,
        "年度数": int(g["年度"].nunique()),
        "年度": sorted(int(y) for y in g["年度"].unique()),
        "回答者数合計": int(g["回答者数"].sum()),
        "回答者数の年平均": int(round(g["回答者数"].mean())),
    })
res["データ量"] = vol

# ---------------------------------------------------------------- 2. 水準と全国順位
KEY = ["幸福度", "生活満足度", "主観24分野平均"]
lev = []
for c in WAKAYAMA:
    g = wk[wk["市区町村"] == c]
    row = {"市": c}
    for k in KEY:
        mu = float(g[k].mean())
        row[k] = round(mu, 2)
        row[f"{k}_全国パーセンタイル"] = pct_rank(k, mu)
    lev.append(row)
# 県平均（回答者数加重）と全国平均も並べる
w = wk["回答者数"]
pref = {"市": "和歌山県（6市加重平均）"}
for k in KEY:
    mu = float((wk[k] * w).sum() / w.sum())
    pref[k] = round(mu, 2)
    pref[f"{k}_全国パーセンタイル"] = pct_rank(k, mu)
lev.append(pref)
nat = {"市": "全国678地域の平均"}
for k in KEY:
    nat[k] = round(float(m[k].mean()), 2)
    nat[f"{k}_全国パーセンタイル"] = 50.0
lev.append(nat)
res["水準"] = lev

res["全国の分布"] = {
    k: {q: round(float(m[k].quantile(q / 100)), 2) for q in (10, 25, 50, 75, 90)} for k in KEY
}

# ---------------------------------------------------------------- 3. 年度ごとの推移
trend = {}
for k in KEY:
    t = {}
    for c in WAKAYAMA:
        g = wk[wk["市区町村"] == c].set_index("年度")[k]
        ys = sorted(g.index)
        t[c] = {
            "年度別": {int(y): round(float(g.loc[y]), 2) for y in ys},
            "最初から最後への変化": round(float(g.loc[ys[-1]] - g.loc[ys[0]]), 2),
            "標準偏差": round(float(g.std(ddof=0)), 2) if len(ys) > 1 else None,
        }
    trend[k] = t
res["推移"] = trend

# ---------------------------------------------------------------- 4. 分野別の市別平均
byc = {}
for c in CATS:
    col = f"主観_{c}"
    row = {}
    for city in WAKAYAMA:
        g = wk[wk["市区町村"] == city]
        mu = float(g[col].mean())
        row[city] = {"平均": round(mu, 1), "全国パーセンタイル": pct_rank(col, mu)}
    row["_6市平均"] = round(float(np.mean([row[x]["平均"] for x in WAKAYAMA])), 1)
    row["_全国平均"] = round(float(m[col].mean()), 1)
    byc[c] = row
res["分野別"] = byc

# ---------------------------------------------------------------- 5. 寄与上位5分野に絞った並び
top5 = {}
for city in WAKAYAMA:
    g = wk[wk["市区町村"] == city]
    vals = {c: round(float(g[f"主観_{c}"].mean()), 1) for c in TOP5}
    pcts = {c: pct_rank(f"主観_{c}", g[f"主観_{c}"].mean()) for c in TOP5}
    top5[city] = {
        "分野別": vals,
        "全国パーセンタイル": pcts,
        "5分野平均": round(float(np.mean(list(vals.values()))), 1),
        "下位25%に入る分野数": int(sum(1 for v in pcts.values() if v < 25)),
        "最も低い分野": min(pcts, key=pcts.get),
    }
res["寄与上位5分野"] = top5

# ---------------------------------------------------------------- 6. 24分野の中での市ごとの弱点（絶対水準）
weak = {}
for city in WAKAYAMA:
    g = wk[wk["市区町村"] == city]
    s = pd.Series({c: float(g[f"主観_{c}"].mean()) for c in CATS}).sort_values()
    weak[city] = {
        "低い順5分野": [{"分野": k, "平均": round(v, 1),
                    "全国パーセンタイル": pct_rank(f"主観_{k}", v)} for k, v in s.head(5).items()],
        "高い順3分野": [{"分野": k, "平均": round(v, 1)} for k, v in s.tail(3)[::-1].items()],
    }
res["市ごとの弱点"] = weak

# ---------------------------------------------------------------- 7. 人口規模など（判断の補助）
pop = {}
kpi = pd.read_csv(PROC / "kpi_municipal.csv", dtype={"code": str})
COLS = {"pop2020": "人口（2020国勢調査）", "aging_rate": "高齢化率", "fiscal_index": "財政力指数"}
for city in WAKAYAMA:
    r = kpi[(kpi["pref"] == "和歌山県") & (kpi["name"] == city)]
    if len(r) == 1:
        r = r.iloc[0]
        pop[city] = {ja: (round(float(r[en]), 3) if pd.notna(r[en]) else None)
                     for en, ja in COLS.items()}
res["市の規模"] = pop

(PROC / "target_selection.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("→ 04_data/processed/target_selection.json")
for r in res["水準"]:
    print(f"{r['市']:<20} 幸福度 {r['幸福度']:>5} (全国{r['幸福度_全国パーセンタイル']:>5}%ile)"
          f"  満足度 {r['生活満足度']:>5}  主観24平均 {r['主観24分野平均']:>5}")
