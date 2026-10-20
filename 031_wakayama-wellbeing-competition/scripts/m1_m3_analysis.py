"""全国での立ち位置・M0〜M3 の計算

  立ち位置 … 和歌山県の全国順位（幸福度・生活満足度・主観/客観24分野）と、4年揃う4市に固定した版
  M0      … 対象の抽出と、主観×客観の年度の結合方法（同年度 / 前年度）の比較
  M1      … 幸福度への寄与度（LMG＝足す順番の平均。ブートストラップで安定度、Elastic Net・生活満足度と突き合わせ）
  M2      … 和歌山6市の下振れ（行内z化した主観−客観から全国平均を引く）
  M3      … 主観を客観で説明できるか（同分野のみ／客観24分野。地域単位の交差検証R²）

入力: data/digital_wellbeing_shikuchoson_2022_2026.xlsx
出力: data/processed/m1_m3_results.json
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import GroupKFold

from wb_common import PROC, WAKAYAMA, categories, load_wb, m0_sample

RNG = np.random.default_rng(20260922)
N_PERM, N_BOOT, N_PERM_BOOT = 3000, 60, 300

subj, obj = load_wb()
CATS = categories(subj)
S = [f"主観_{c}" for c in CATS]
O = [f"客観_{c}" for c in CATS]
m = m0_sample(subj, obj)
res = {}

# ================================================================ 全国での立ち位置
ward = subj["市区町村"].str.contains(r"市.+区$")                   # 政令市の区（全域行と二重になる）
sp = subj[(subj["区分"] == "全体") & (subj["回答者数"] >= 30) & ~ward & subj["年度"].between(2023, 2026)]
op = obj[~obj["市区町村"].str.contains(r"市.+区$") & obj["年度"].between(2023, 2025)]


def wmean(g, cols):
    w = g["回答者数"]
    return (g[cols].mul(w, axis=0)).sum() / w.sum()


def pref_ranks(sub, cities=None):
    """都道府県別の値→順位（1位が最良）。主観は回答者数加重、客観は等重み"""
    out = {}
    for y in range(2023, 2027):
        s = sub[sub["年度"] == y]
        ps = s.groupby("都道府県").apply(lambda g: wmean(g, S + ["幸福度", "生活満足度"]), include_groups=False)
        o = op[op["年度"] == min(y, 2025)].groupby("都道府県")[O].mean()
        tbl = pd.DataFrame({"幸福度": ps["幸福度"], "生活満足度": ps["生活満足度"],
                            "主観24分野平均": ps[S].mean(axis=1), "客観24分野平均": o.mean(axis=1)})
        for c in CATS:
            tbl[f"主観_{c}"] = ps[f"主観_{c}"]
            tbl[f"客観_{c}"] = o[f"客観_{c}"]
        rk = tbl.rank(ascending=False, method="min")
        out[y] = {k: int(v) for k, v in rk.loc["和歌山県"].items() if pd.notna(v)}
        out[y]["_n_cities"] = int((s["都道府県"] == "和歌山県").sum())
    return out


res["position"] = pref_ranks(sp)
four = ["和歌山市", "橋本市", "紀の川市", "岩出市"]
sp4 = sp[(sp["都道府県"] != "和歌山県") | sp["市区町村"].isin(four)]
res["position_4cities"] = pref_ranks(sp4)

# ================================================================ M0 年度の結合方法
def join_corr(lag):
    o = obj.copy()
    o["年度"] = o["年度"] + lag
    a = subj[(subj["区分"] == "全体") & subj["年度"].between(2023, 2026) & (subj["回答者数"] >= 30)]
    a = a[~a["市区町村"].str.contains(r"区$")]
    j = a.merge(o[["年度", "地域ID"] + O], on=["年度", "地域ID"]).dropna(subset=O)
    return float(np.mean([np.corrcoef(j[s], j[oc])[0, 1] for s, oc in zip(S, O)])), int(len(j))


res["m0"] = {"rows": int(len(m)), "regions": int(m["地域ID"].nunique()),
             "same_year": join_corr(0), "prev_year": join_corr(1)}

# ================================================================ M1 寄与度
def lmg(C, n_perm, rng):
    """相関行列 C（0列目が目的変数）から LMG。R²(部分集合) を順番を変えて積み上げた増分の平均"""
    k = C.shape[0] - 1
    cache = {}

    def r2(mask):
        if mask not in cache:
            idx = [i + 1 for i in range(k) if mask >> i & 1]
            if not idx:
                cache[mask] = 0.0
            else:
                r = C[0, idx]
                cache[mask] = float(r @ np.linalg.solve(C[np.ix_(idx, idx)], r))
        return cache[mask]

    share = np.zeros(k)
    for _ in range(n_perm):
        mask, prev = 0, 0.0
        for i in rng.permutation(k):
            mask |= 1 << i
            cur = r2(mask)
            share[i] += cur - prev
            prev = cur
    return share / n_perm


def corr_of(df, y):
    return np.corrcoef(df[[y] + S].values.T)


def group_cv_r2(X, y, groups, k=5):
    pred = np.empty(len(y))
    for tr, te in GroupKFold(k).split(X, y, groups):
        pred[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


X, yh, g = m[S].values, m["幸福度"].values, m["地域ID"].values
r2_full = float(LinearRegression().fit(X, yh).score(X, yh))
lmg_h = lmg(corr_of(m, "幸福度"), N_PERM, RNG)
lmg_s = lmg(corr_of(m, "生活満足度"), N_PERM, RNG)

# ブートストラップ：地域を選び直す（同じ地域の年度はまとめて入る）
regions = m["地域ID"].unique()
by_reg = m.groupby("地域ID").indices
top5 = np.zeros(len(CATS))
for b in range(N_BOOT):
    pick = RNG.choice(regions, len(regions), replace=True)
    bm = m.iloc[np.concatenate([by_reg[r] for r in pick])]
    sh = lmg(corr_of(bm, "幸福度"), N_PERM_BOOT, RNG)
    top5[np.argsort(-sh)[:5]] += 1

Xz = (X - X.mean(0)) / X.std(0)
folds = list(GroupKFold(5).split(Xz, yh, g))
en = ElasticNetCV(l1_ratio=[.1, .5, .9], cv=folds, random_state=0).fit(Xz, yh)
rank_lmg = pd.Series(lmg_h, CATS).rank(ascending=False)
rank_en = pd.Series(np.abs(en.coef_), CATS).rank(ascending=False)
rank_ls = pd.Series(lmg_s, CATS).rank(ascending=False, method="min")

res["m1"] = {
    "r2": r2_full, "cv_r2": group_cv_r2(X, yh, g),
    "share": {c: float(v / lmg_h.sum()) for c, v in zip(CATS, lmg_h)},
    "stability_top5": {c: float(v / N_BOOT) for c, v in zip(CATS, top5)},
    "rank_life_satisfaction": {c: int(v) for c, v in rank_ls.items()},
    "top_life_satisfaction": CATS[int(np.argmax(lmg_s))],
    "spearman_en_lmg": float(spearmanr(rank_lmg, rank_en).statistic),
    "spearman_ls_lmg": float(spearmanr(lmg_h, lmg_s).statistic),
    "en_coef": {c: float(v) for c, v in zip(CATS, en.coef_)},
    "en_top4": [CATS[i] for i in np.argsort(-np.abs(en.coef_))[:4]],
}

# ================================================================ M2 和歌山6市の下振れ
def rowz(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1, ddof=0), axis=0)


d = rowz(m[S]).values - rowz(m[O]).values
nat = d.mean(axis=0)
adj = pd.DataFrame(d - nat, columns=CATS)
adj["年度"], adj["市区町村"], adj["都道府県"] = m["年度"], m["市区町村"], m["都道府県"]
wk = adj[adj["都道府県"] == "和歌山県"]
top5_cats = list(pd.Series(lmg_h, CATS).sort_values(ascending=False).index[:5])
res["m2"] = {
    "national_mean_gap": {c: float(v) for c, v in zip(CATS, nat)},
    "city_mean": {c: {k: float(v) for k, v in wk[wk["市区町村"] == c][CATS].mean().items()} for c in WAKAYAMA},
    "pref_mean": {k: float(v) for k, v in wk[CATS].mean().items()},
    "city_years": {c: sorted(int(y) for y in wk[wk["市区町村"] == c]["年度"]) for c in WAKAYAMA},
    "by_year": {f"{c}×{k}": {int(r["年度"]): float(r[k]) for _, r in wk[wk["市区町村"] == c].iterrows()}
                for c, k in [("田辺市", "自己効力感"), ("橋本市", "健康状態")]},
    "happiness_wakayama": float(wmean(m[m["都道府県"] == "和歌山県"], ["幸福度"]).iloc[0]),
    "happiness_national": float(wmean(m, ["幸福度"]).iloc[0]),
}

# ================================================================ M3 説明力
m3 = {}
for c in top5_cats:
    y = m[f"主観_{c}"].values
    m3[c] = {
        "same_only": group_cv_r2(m[[f"客観_{c}"]].values, y, g),
        "all24": group_cv_r2(m[O].values, y, g),
        "corr": float(np.corrcoef(m[f"主観_{c}"], m[f"客観_{c}"])[0, 1]),
    }
    m3[c]["route"] = "B候補" if m3[c]["same_only"] < 0.1 and m3[c]["all24"] < 0.35 else "A候補"
res["m3"] = m3
res["top5"] = top5_cats

(PROC / "m1_m3_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- 画面確認用
print("M0", res["m0"])
print(f"M1 R2={r2_full:.3f} CV={res['m1']['cv_r2']:.3f} EN-LMG ρ={res['m1']['spearman_en_lmg']:.2f} "
      f"LS-LMG ρ={res['m1']['spearman_ls_lmg']:.2f} 生活満足度首位={res['m1']['top_life_satisfaction']}")
for c in pd.Series(res["m1"]["share"]).sort_values(ascending=False).index[:10]:
    print(f"  {c:<10} {res['m1']['share'][c]*100:5.1f}%  安定度{res['m1']['stability_top5'][c]*100:4.0f}%  "
          f"満足度{res['m1']['rank_life_satisfaction'][c]}位")
print("EN上位4", res["m1"]["en_top4"])
print("M2", json.dumps(res["m2"]["city_mean"], ensure_ascii=False))
print("   ", res["m2"]["by_year"], res["m2"]["happiness_wakayama"], res["m2"]["happiness_national"])
print("M3", json.dumps(m3, ensure_ascii=False))
for y, r in res["position"].items():
    print(y, {k: r[k] for k in ["幸福度", "生活満足度", "主観24分野平均", "客観24分野平均", "_n_cities"]})

# ================================================================ 都道府県ごとの比較
BLOCKS = {
    "北海道・東北": "北海道 青森県 岩手県 宮城県 秋田県 山形県 福島県",
    "関東": "茨城県 栃木県 群馬県 埼玉県 千葉県 東京都 神奈川県",
    "中部": "新潟県 富山県 石川県 福井県 山梨県 長野県 岐阜県 静岡県 愛知県",
    "近畿": "三重県 滋賀県 京都府 大阪府 兵庫県 奈良県 和歌山県",
    "中国": "鳥取県 島根県 岡山県 広島県 山口県",
    "四国": "徳島県 香川県 愛媛県 高知県",
    "九州・沖縄": "福岡県 佐賀県 長崎県 熊本県 大分県 宮崎県 鹿児島県 沖縄県",
}
blk = {p: b for b, ps in BLOCKS.items() for p in ps.split()}

# --- M0 カバー率：条件を満たす市区町村の数と割合
allm = subj[~subj["市区町村"].str.contains(r"区$")].groupby("都道府県")["地域ID"].nunique()
cov = m.groupby("都道府県").agg(rows=("地域ID", "size"), regions=("地域ID", "nunique"))
cov["all"] = allm
cov["rate"] = cov["regions"] / cov["all"] * 100
res["coverage"] = {
    "national": {"regions": int(cov["regions"].sum()), "all": int(allm.sum()),
                 "rate": float(cov["regions"].sum() / allm.sum() * 100),
                 "median_regions": float(cov["regions"].median()), "median_rate": float(cov["rate"].median())},
    "pref": {p: {"regions": int(r["regions"]), "all": int(r["all"]), "rate": float(r["rate"]),
                 "rows": int(r["rows"]), "rank_regions": int(cov["regions"].rank(ascending=True)[p])}
             for p, r in cov.iterrows()},
}

# --- M1 ブロック別のLMG（都道府県単独では行数が足りないため、8地方ブロックで計算）
mb = m.assign(block=m["都道府県"].map(blk))
m1b = {}
for b in BLOCKS:
    sub = mb[mb["block"] == b]
    sh_b = lmg(corr_of(sub, "幸福度"), N_PERM, RNG)
    m1b[b] = {"rows": int(len(sub)), "regions": int(sub["地域ID"].nunique()),
              "share": {c: float(v / sh_b.sum()) for c, v in zip(CATS, sh_b)},
              "top3": [CATS[i] for i in np.argsort(-sh_b)[:3]],
              "spearman_vs_national": float(spearmanr(lmg_h, sh_b).statistic)}
res["m1_block"] = m1b

# --- M1 和歌山だけ（21行なので24分野の回帰は解けない。単相関と、全国モデルのずれで見る）
wm = m[m["都道府県"] == "和歌山県"]
pred_nat = LinearRegression().fit(X, yh).predict(wm[S].values)
res["m1_wakayama"] = {
    "rows": int(len(wm)),
    "corr_national": {c: float(np.corrcoef(m[f"主観_{c}"], m["幸福度"])[0, 1]) for c in CATS},
    "corr_wakayama": {c: float(np.corrcoef(wm[f"主観_{c}"], wm["幸福度"])[0, 1]) for c in CATS},
    "resid_mean": float((wm["幸福度"] - pred_nat).mean()),
    "resid_by_city": {c: float((wm["幸福度"] - pred_nat)[wm["市区町村"] == c].mean()) for c in WAKAYAMA},
}

# --- M2 都道府県別の下振れ（行の単純平均）
adj["block"] = adj["都道府県"].map(blk)
pref_gap = adj.groupby("都道府県")[CATS].mean()
res["m2_pref"] = {
    "rank_wakayama": {c: int(pref_gap[c].rank(ascending=True)["和歌山県"]) for c in top5_cats},
    "value_wakayama": {c: float(pref_gap[c]["和歌山県"]) for c in top5_cats},
    "worst5": {c: [{"pref": p, "v": float(v)} for p, v in pref_gap[c].nsmallest(5).items()] for c in top5_cats},
    "best3": {c: [{"pref": p, "v": float(v)} for p, v in pref_gap[c].nlargest(3).items()] for c in top5_cats},
}
# 市町村単位（4年平均）で見たときの全国順位
city_gap = adj.groupby(["都道府県", "市区町村"])[CATS].mean()
res["m2_city_rank"] = {}
for city, c in [("田辺市", "自己効力感"), ("橋本市", "健康状態"), ("和歌山市", "地域とのつながり")]:
    r_ = city_gap[c].rank(ascending=True)
    res["m2_city_rank"][f"{city}×{c}"] = {"rank": int(r_[("和歌山県", city)]), "n": int(len(r_)),
                                          "value": float(city_gap[c][("和歌山県", city)]),
                                          "worst3": [{"pref": p, "city": ct, "v": float(v)}
                                                     for (p, ct), v in city_gap[c].nsmallest(3).items()]}

# --- M3 都道府県別の説明力（同分野の客観との相関。15行以上の都道府県のみ）
res["m3_pref"] = {}
big = cov[cov["rows"] >= 15].index
for c in top5_cats:
    rs = {}
    for p in big:
        sub = m[m["都道府県"] == p]
        rs[p] = float(np.corrcoef(sub[f"主観_{c}"], sub[f"客観_{c}"])[0, 1])
    ser = pd.Series(rs)
    res["m3_pref"][c] = {
        "n_pref": int(len(ser)), "median": float(ser.median()),
        "positive_share": float((ser > 0).mean()),
        "wakayama": float(ser["和歌山県"]) if "和歌山県" in ser else None,
        "rank_wakayama": int(ser.rank(ascending=False)["和歌山県"]) if "和歌山県" in ser else None,
        "national": r["m3"][c]["corr"] if isinstance(r := res, dict) else None,
    }

# 和歌山6市×24分野が、全国678地域の中でどの位置にあるか（1位＝最も下振れ）
rank_all = city_gap.rank(ascending=True)
res["m2"]["city_rank_all"] = {c: {k: int(rank_all.loc[("和歌山県", c), k]) for k in CATS} for c in WAKAYAMA}
res["m2"]["n_regions"] = int(len(city_gap))

(PROC / "m1_m3_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nカバー率", res["coverage"]["national"], "和歌山", res["coverage"]["pref"]["和歌山県"])
for b, v in m1b.items():
    print(f"  {b:<8} n={v['rows']:4d} 上位3={v['top3']} ρ={v['spearman_vs_national']:.2f}")
print("和歌山 残差", res["m1_wakayama"]["resid_mean"], res["m1_wakayama"]["resid_by_city"])
print("M2 和歌山順位", res["m2_pref"]["rank_wakayama"], res["m2_pref"]["value_wakayama"])
print("M2 worst", json.dumps(res["m2_pref"]["worst5"], ensure_ascii=False)[:900])
print("M2 city", json.dumps(res["m2_city_rank"], ensure_ascii=False))
print("M3 pref", json.dumps(res["m3_pref"], ensure_ascii=False))
