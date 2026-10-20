"""ステップ0〜4・改善候補①②を、年代×性別の行で計算し直す（10-01）

分析対象（前処理）：主観データの年代×性別の行・2023〜26年度・男女・区を除く・同年度の客観あり・回答者数5人以上
年代・性別の構成の違いは「属性調整値」で取り除く：
    調整値 = 値 − 同じ年度×年代×性別の全国平均（回答者数加重）＋全国平均
市町村単位の値は、調整値の回答者数加重平均。

出力：data/processed/m8_agesex_results.json
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from wb_common import PROC, load_wb
from wb_typology import VARS, typology

MIN_N = 5
RNG = np.random.default_rng(20261001)
CELL = ["年度", "年代", "性別"]
TARGETS = ["和歌山市", "田辺市"]
WAKAYAMA = ["田辺市", "和歌山市", "橋本市", "岩出市", "海南市", "紀の川市"]

# 分野ごとの個別KPI（列名, 表示名, 向き：大きいほど望ましい＝+1）
KPIS = {
    "医療・福祉": [("medical_walk_cover", "医療施設徒歩圏人口カバー率", 1), ("welfare_walk_cover", "福祉施設徒歩圏人口カバー率", 1)],
    "健康状態": [("healthy_life_m", "健康寿命（男性・近似）", 1), ("healthy_life_f", "健康寿命（女性・近似）", 1)],
    "移動・交通": [("station_bus_walk_cover", "駅・バス停徒歩圏人口カバー率", 1), ("small_car_km_per_capita", "人口あたり小型車走行キロ", -1)],
    "公共空間": [("park_walk_cover", "公園緑地徒歩圏人口カバー率", 1), ("park_area_per_capita", "人口あたり公園面積", 1),
             ("sidewalk_rate", "歩道設置率", 1), ("walkable_city", "ウォーカブル推進都市", 1)],
    "初等・中等教育": [("elem_school_per_hab", "可住地面積あたり小学校数", 1), ("jhs_per_hab", "可住地面積あたり中学校数", 1),
                ("hs_per_hab", "可住地面積あたり高等学校数", 1)],
    "買物・飲食": [("commercial_walk_cover", "商業施設徒歩圏人口カバー率", 1), ("restaurant_per_hab", "可住地面積あたり飲食店数", 1)],
    "子育て": [("kinder_per_hab", "可住地面積あたり幼稚園数", 1), ("nursery_walk_cover", "保育所徒歩圏0〜4歳人口カバー率", 1),
            ("edu_exp_share", "歳出総額の教育費割合", 1)],
    "自己効力感": [("net_migration_rate", "転入超過割合", 1), ("unemployed_per_pop", "完全失業者数（人口比）", -1)],
    "地域とのつながり": [("suicide_per_100k", "人口あたり自殺者数", -1), ("extended_family_share", "拡大家族世帯割合", 1),
                 ("married_share", "既婚者割合", 1), ("elderly_single_share", "高齢単身世帯割合", -1),
                 ("political_org_per_10k", "人口あたり政治団体等の数", 1), ("religious_est_per_10k", "人口あたり宗教の事業所数", 1),
                 ("npo_per_10k", "人口あたりNPOの数", 1)],
    "地域行政": [("debt_service_ratio", "実質公債費比率", -1)],
    "雇用・所得": [("unemployment_rate", "完全失業率", -1)],
    "住宅環境": [("floor_area_per_dwelling", "住宅当たり延べ面積", 1), ("rent_per_m2", "専用住宅面積あたり家賃", -1),
             ("detached_owner_share", "一戸建の持ち家の割合", 1)],
    "遊び・娯楽": [("leisure_est_per_10k", "人口あたり生活関連サービス業・娯楽業事業所数", 1)],
}


def wmean(x, w):
    return float(np.average(x, weights=w))


def pct(series, v):
    """series の中で v より低い割合（%）。同値は半分数える"""
    s = series.dropna()
    return float(((s < v).mean() + (s == v).mean() / 2) * 100)


# ================================================================ 前処理
def load():
    cache = os.environ.get("WB_CACHE")
    if cache:
        return pd.read_pickle(f"{cache}/subj.pkl"), pd.read_pickle(f"{cache}/obj.pkl")
    return load_wb()


subj, obj = load()
cats = [c[3:] for c in subj.columns if c.startswith("主観_")]
SC = [f"主観_{c}" for c in cats]
OC = [f"客観_{c}" for c in cats]

funnel = []
d = subj
funnel.append(("収集（全区分・全年度）", len(d), d["地域ID"].nunique()))
d = d[d["区分"] == "年代×性別"]
funnel.append(("年代×性別の行", len(d), d["地域ID"].nunique()))
d = d[d["年度"].between(2023, 2026)]
funnel.append(("2023〜26年度", len(d), d["地域ID"].nunique()))
d = d[d["性別"].isin(["男性", "女性"]) & (d["回答者数"] > 0)]
d = d[~d["市区町村"].str.contains(r"区$")]
funnel.append(("区を除く", len(d), d["地域ID"].nunique()))
d = d.merge(obj[["年度", "地域ID"] + OC], on=["年度", "地域ID"]).dropna(subset=OC + SC + ["幸福度"])
funnel.append(("客観データあり", len(d), d["地域ID"].nunique()))
d = d[d["回答者数"] >= MIN_N].reset_index(drop=True)
funnel.append((f"回答者数{MIN_N}人以上", len(d), d["地域ID"].nunique()))
print(funnel)

w = d["回答者数"].astype(float)
d["name"] = d["市区町村"].str.replace("全域", "", regex=False)

# 属性調整値
YV = SC + ["幸福度"]
cellmean = d[YV].mul(w, axis=0).groupby([d[c] for c in CELL]).transform("sum").div(
    w.groupby([d[c] for c in CELL]).transform("sum"), axis=0)
grand = {v: wmean(d[v], w) for v in YV}
adj = d[YV] - cellmean + pd.Series(grand)
adj.columns = [f"adj_{c}" for c in YV]
d = pd.concat([d, adj], axis=1)
d["adj_avg24"] = d[[f"adj_{c}" for c in SC]].mean(axis=1)
d["obj_avg24"] = d[OC].mean(axis=1)

# 市町村パネル（1行＝1市町村）
kpi = pd.read_csv(PROC / "kpi_municipal.csv", dtype={"code": str})
kx = pd.read_csv(PROC / "kpi_extra.csv", dtype={"code": str})
kpi = kpi.merge(kx, on="code", how="left")
g = d.groupby("地域ID")
cols = [f"adj_{c}" for c in YV] + ["adj_avg24"] + OC + ["obj_avg24"]
panel = d[cols].mul(w, axis=0).groupby(d["地域ID"]).sum().div(w.groupby(d["地域ID"]).sum(), axis=0)
panel["pref"] = g["都道府県"].first()
panel["name"] = g["name"].first()
panel["n_resp"] = w.groupby(d["地域ID"]).sum()
panel = panel.reset_index().merge(kpi, on=["pref", "name"], how="left", suffixes=("", "_k"))
miss = panel[panel["code"].isna()]
print("コードが付かない地域", len(miss), miss[["pref", "name"]].values.tolist()[:10])
panel = panel.dropna(subset=["code"]).reset_index(drop=True)
d = d[d["地域ID"].isin(panel["地域ID"])].reset_index(drop=True)
w = d["回答者数"].astype(float)

# ================================================================ ステップ0：類型
types, Zs, struct, elb = typology(panel["code"].values)
panel["類型"] = types.values
d = d.merge(panel[["地域ID", "類型"]], on="地域ID")
TYPES = ["都市部", "地方の中心都市", "農山村の小都市"]
st = struct.reset_index().assign(類型=types.values, name=panel["name"].values, pref=panel["pref"].values)
s0 = {"types": [], "elbow": elb}
for t in TYPES:
    x = st[st["類型"] == t]
    s0["types"].append({"type": t, "n": int(len(x)), "pop_med": float(x.pop2020.median()),
                        "dens_med": float(x.pop_density.median()), "aging_med": float(x.aging_rate.median()),
                        "fiscal_med": float(x.fiscal_index.median()), "primary_med": float(x.primary_share.median()),
                        "hab_med": float(x.habitable_share.median()),
                        "wakayama": [c for c in WAKAYAMA if ((x.pref == "和歌山県") & (x.name == c)).any()],
                        "examples": (x.nlargest(4, "pop2020").pref + x.nlargest(4, "pop2020").name).tolist()})
peers = {}
for c in WAKAYAMA:
    i = st.index[(st.pref == "和歌山県") & (st.name == c)][0]
    t = st.loc[i, "類型"]
    same = st[(st["類型"] == t) & (st.index != i)]
    dist = np.sqrt(((Zs.values[same.index] - Zs.values[i]) ** 2).sum(axis=1))
    near = same.assign(dist=dist).nsmallest(8, "dist")
    r = st.loc[i]
    peers[c] = {"type": t, "pop": float(r.pop2020), "dens": float(r.pop_density), "aging": float(r.aging_rate),
                "fiscal": float(r.fiscal_index), "primary": float(r.primary_share), "hab": float(r.habitable_share),
                "type_n": int((st["類型"] == t).sum()),
                "kinki": int(((st["類型"] == t) & st.pref.isin(["滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"])).sum()),
                "same_pref": [o for o in WAKAYAMA if o != c and ((st.pref == "和歌山県") & (st.name == o) & (st["類型"] == t)).any()],
                "near": [(p + n, float(pp)) for p, n, pp in zip(near.pref, near.name, near.pop2020)]}
s0["peers"] = peers


def r2_type(y, wts, typ):
    X = pd.get_dummies(typ, drop_first=True).astype(float)
    X = sm.add_constant(X)
    return float(sm.WLS(y, X, weights=wts).fit().rsquared)


s0["type_r2"] = {c: r2_type(panel[f"adj_主観_{c}"], panel["n_resp"], panel["類型"]) for c in cats}
s0["type_r2_avg"] = float(np.mean(list(s0["type_r2"].values())))
s0["type_r2_happy"] = r2_type(panel["adj_幸福度"], panel["n_resp"], panel["類型"])
s0["type_means"] = {c: {t: wmean(panel.loc[panel["類型"] == t, f"adj_主観_{c}"], panel.loc[panel["類型"] == t, "n_resp"])
                        for t in TYPES} for c in cats}
s0["type_happy"] = {t: wmean(panel.loc[panel["類型"] == t, "adj_幸福度"], panel.loc[panel["類型"] == t, "n_resp"]) for t in TYPES}
# 行単位：年代×性別で説明できる割合（同じ年度の中で）＝グループ平均で説明できる分散の割合（加重）
def cell_r2(v):
    gm = d[v].mul(w).groupby([d[c] for c in CELL]).transform("sum") / w.groupby([d[c] for c in CELL]).transform("sum")
    mu = wmean(d[v], w)
    return float(1 - (w * (d[v] - gm) ** 2).sum() / (w * (d[v] - mu) ** 2).sum())


s0["agesex_r2_avg"] = float(np.mean([cell_r2(v) for v in SC]))
s0["agesex_r2_happy"] = cell_r2("幸福度")

# ================================================================ ステップ1：立ち位置
s1 = {}
ranks = {"幸福度": {}, "主観平均": {}, "客観平均": {}}
for y, gdf in d.groupby("年度"):
    ww = gdf["回答者数"]
    for key, col in [("幸福度", "adj_幸福度"), ("主観平均", "adj_avg24"), ("客観平均", "obj_avg24")]:
        pm = (gdf[col] * ww).groupby(gdf["都道府県"]).sum() / ww.groupby(gdf["都道府県"]).sum()
        ranks[key][int(y)] = {"rank": int(pm.rank(ascending=False)["和歌山県"]), "n": int(len(pm))}
s1["pref_ranks"] = ranks
wk = d["都道府県"] == "和歌山県"
for key, col in [("avg24", "adj_avg24"), ("happy", "adj_幸福度")]:
    nat = wmean(d[col], w)
    act = wmean(d.loc[wk, col], w[wk])
    tmean = {t: wmean(d.loc[d["類型"] == t, col], w[d["類型"] == t]) for t in TYPES}
    wt = w[wk].groupby(d.loc[wk, "類型"]).sum()
    exp = float(sum(tmean[t] * wt[t] for t in wt.index) / wt.sum())
    s1[key] = {"national": nat, "expected": exp, "actual": act,
               "share_type": float((nat - exp) / (nat - act)) if nat != act else None}
cities = {}
for c in WAKAYAMA:
    r = panel[(panel.pref == "和歌山県") & (panel.name == c)].iloc[0]
    same = panel[panel["類型"] == r["類型"]]
    cities[c] = {"type": r["類型"], "n_resp": float(r.n_resp),
                 "avg24": float(r.adj_avg24), "avg24_type": pct(same.adj_avg24, r.adj_avg24), "avg24_nat": pct(panel.adj_avg24, r.adj_avg24),
                 "happy": float(r.adj_幸福度), "happy_type": pct(same.adj_幸福度, r.adj_幸福度), "happy_nat": pct(panel.adj_幸福度, r.adj_幸福度),
                 "cats": {k: {"v": float(r[f"adj_主観_{k}"]), "type": pct(same[f"adj_主観_{k}"], r[f"adj_主観_{k}"]),
                              "nat": pct(panel[f"adj_主観_{k}"], r[f"adj_主観_{k}"])} for k in cats}}
s1["cities"] = cities

# ================================================================ ステップ3：寄与度（LMG）
def lmg(df, ycol, xcols, wts, perms):
    """加重相関行列から部分集合のR²を出し、順番をランダムに変えてR²の増分を平均する"""
    M = df[[ycol] + xcols].values
    ww = wts.values / wts.sum()
    mu = (M * ww[:, None]).sum(0)
    C = ((M - mu) * ww[:, None]).T @ (M - mu)
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    ry, Rxx = R[0, 1:], R[1:, 1:]
    cache = {(): 0.0}

    def r2(S):
        S = tuple(sorted(S))
        if S not in cache:
            idx = list(S)
            cache[S] = float(ry[idx] @ np.linalg.solve(Rxx[np.ix_(idx, idx)], ry[idx]))
        return cache[S]
    k = len(xcols)
    share = np.zeros(k)
    for _ in range(perms):
        order = RNG.permutation(k)
        prev, S = 0.0, []
        for j in order:
            S.append(j)
            cur = r2(S)
            share[j] += cur - prev
            prev = cur
    share /= perms
    return float(r2(range(k))), share


def demean(df, by, cols, wts):
    m = df[cols].mul(wts, axis=0).groupby([df[b] for b in by]).transform("sum").div(
        wts.groupby([df[b] for b in by]).transform("sum"), axis=0)
    return df[cols] - m


s3 = {}
specs = {"類型を揃えた全国": (d, ["類型"] + CELL), "都市部": (d[d["類型"] == "都市部"], CELL),
         "農山村の小都市": (d[d["類型"] == "農山村の小都市"], CELL), "全国（参考）": (d, CELL)}
for name, (df, by) in specs.items():
    ww = df["回答者数"].astype(float)
    dm = demean(df, by, SC + ["幸福度"], ww)
    tot, sh = lmg(dm, "幸福度", SC, ww, 3000 if name == "全国（参考）" else 1500)
    rk = (-sh).argsort().argsort() + 1
    s3[name] = {"r2": tot, "rows": int(len(df)), "share": {cats[i]: float(sh[i] / tot) for i in range(len(cats))},
                "rank": {cats[i]: int(rk[i]) for i in range(len(cats))}}
    print("LMG", name, round(tot, 3))

# ================================================================ ステップ4：下振れ
zs = d[SC].sub(d[SC].mean(axis=1), axis=0).div(d[SC].std(axis=1, ddof=0), axis=0)
zo = d[OC].sub(d[OC].mean(axis=1), axis=0).div(d[OC].std(axis=1, ddof=0), axis=0)
gap = pd.DataFrame(zs.values - zo.values, columns=cats, index=d.index)
gd = pd.concat([d[["地域ID", "類型"] + CELL], gap], axis=1)
gap_type = demean(gd, ["類型"] + CELL, cats, w)
gap_nat = demean(gd, CELL, cats, w)
ws = w.groupby(d["地域ID"]).sum()
gt = gap_type.mul(w, axis=0).groupby(d["地域ID"]).sum().div(ws, axis=0)
gn = gap_nat.mul(w, axis=0).groupby(d["地域ID"]).sum().div(ws, axis=0)
typ_of = panel.set_index("地域ID")["類型"]
s4 = {"pairs": {}, "type_mean_nat": {}}
for c in WAKAYAMA:
    rid = panel.loc[(panel.pref == "和歌山県") & (panel.name == c), "地域ID"].iat[0]
    t = typ_of[rid]
    same = typ_of.index[typ_of == t]
    for k in cats:
        s4["pairs"][f"{c}|{k}"] = {
            "type_gap": float(gt.loc[rid, k]), "type_rank": int((gt.loc[same, k] < gt.loc[rid, k]).sum() + 1), "type_n": int(len(same)),
            "nat_gap": float(gn.loc[rid, k]), "nat_rank": int((gn[k] < gn.loc[rid, k]).sum() + 1), "nat_n": int(len(gn))}
for t in TYPES:
    same = typ_of.index[typ_of == t]
    s4["type_mean_nat"][t] = {k: float(np.average(gn.loc[same, k], weights=ws[same])) for k in cats}

# ================================================================ 改善候補①：環境整備で上がる分野か
def cv_r2(X, y, reps=20, k=5):
    X, y = np.asarray(X, float), np.asarray(y, float)
    out = []
    for r in range(reps):
        pred = np.empty_like(y)
        for tr, te in KFold(k, shuffle=True, random_state=r).split(X):
            pred[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
        out.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out))


TD = pd.get_dummies(panel["類型"], drop_first=True).astype(float)
z = lambda s: (s - s.mean()) / s.std(ddof=0)
c1 = {}
for cat, items in KPIS.items():
    kc = [c for c, _, _ in items]
    m = panel[kc].notna().all(axis=1)
    y = panel.loc[m, f"adj_主観_{cat}"]
    K = panel.loc[m, kc].apply(z)
    base = cv_r2(TD[m], y)
    full = cv_r2(pd.concat([TD[m], K], axis=1), y)
    own = cv_r2(panel.loc[m, [f"客観_{cat}"]], y)
    o24 = cv_r2(panel.loc[m, OC], y)
    # KPIごとの向きと有意性（類型＋分野のKPIの重回帰、HC1）
    X = sm.add_constant(pd.concat([TD[m], K], axis=1))
    fit = sm.OLS(y, X).fit(cov_type="HC1")
    kp = {c: {"b": float(fit.params[c]), "p": float(fit.pvalues[c]), "ok": bool(np.sign(fit.params[c]) == s and fit.pvalues[c] < 0.05)}
          for c, _, s in items}
    c1[cat] = {"n": int(m.sum()), "type_only": base, "type_kpi": full, "gain": full - base, "own_obj": own, "obj24": o24, "kpi": kp}
    print("候補①", cat, round(base, 3), round(full, 3))

# ================================================================ 改善候補②：伸ばす数値の候補
allk = [(c, lab, s, cat) for cat, items in KPIS.items() for c, lab, s in items]
c2 = {"kpi": {}}
for c, lab, s, cat in allk:
    m = panel[c].notna()
    X = sm.add_constant(pd.concat([TD[m], z(panel.loc[m, c]).rename(c)], axis=1))
    fit = sm.WLS(panel.loc[m, "adj_幸福度"], X, weights=panel.loc[m, "n_resp"]).fit(cov_type="HC1")
    c2["kpi"][c] = {"label": lab, "cat": cat, "sign": s, "b_sd": float(fit.params[c]), "p": float(fit.pvalues[c]),
                    "happy_ok": bool(np.sign(fit.params[c]) == s and fit.pvalues[c] < 0.05),
                    "subj_ok": c1[cat]["kpi"][c]["ok"], "sd": float(panel.loc[m, c].std(ddof=0)), "n": int(m.sum())}
use = [c for c, _, _, _ in allk if panel[c].isna().mean() < 0.10]
m = panel[use].notna().all(axis=1)
r2_t = float(sm.WLS(panel.loc[m, "adj_幸福度"], sm.add_constant(TD[m]), weights=panel.loc[m, "n_resp"]).fit().rsquared)
r2_tk = float(sm.WLS(panel.loc[m, "adj_幸福度"], sm.add_constant(pd.concat([TD[m], panel.loc[m, use].apply(z)], axis=1)),
                     weights=panel.loc[m, "n_resp"]).fit().rsquared)
c2["all"] = {"n_kpi": len(use), "n_mun": int(m.sum()), "r2_type": r2_t, "r2_type_kpi": r2_tk}
c2["cities"] = {}
for city in TARGETS:
    r = panel[(panel.pref == "和歌山県") & (panel.name == city)].iloc[0]
    same = panel[panel["類型"] == r["類型"]]
    rows = []
    for c, lab, s, cat in allk:
        if pd.isna(r[c]):
            continue
        p = pct(same[c] * s, r[c] * s)
        k = c2["kpi"][c]
        med = float(same[c].median())
        gapv = (med - r[c]) * s
        rows.append({"col": c, "label": lab, "cat": cat, "value": float(r[c]), "median": med, "pct": p,
                     "subj_ok": k["subj_ok"], "happy_ok": k["happy_ok"],
                     "happy_gain": float(abs(k["b_sd"]) * gapv / k["sd"]) if (k["happy_ok"] and gapv > 0) else None})
    c2["cities"][city] = sorted(rows, key=lambda x: x["pct"])
# 予測との差（類型＋分野のKPI → 実感）
c2["resid"] = {}
for city, cat in [("田辺市", "自己効力感"), ("和歌山市", "地域とのつながり"), ("和歌山市", "移動・交通"), ("和歌山市", "公共空間")]:
    kc = [c for c, _, _ in KPIS[cat]]
    m = panel[kc].notna().all(axis=1)
    X = pd.concat([TD[m], panel.loc[m, kc].apply(z)], axis=1)
    lr = LinearRegression().fit(X, panel.loc[m, f"adj_主観_{cat}"])
    res = panel.loc[m, f"adj_主観_{cat}"] - lr.predict(X)
    i = panel.index[(panel.pref == "和歌山県") & (panel.name == city)][0]
    c2["resid"][f"{city}|{cat}"] = {"actual": float(panel.loc[i, f"adj_主観_{cat}"]), "pred": float(panel.loc[i, f"adj_主観_{cat}"] - res[i]),
                                    "resid": float(res[i]), "nat_pct": pct(res, res[i])}

# KPIの欠測
c2["missing"] = {c: int(panel[c].isna().sum()) for c, _, _, _ in allk if panel[c].isna().any()}

out = {"funnel": funnel, "n_rows": int(len(d)), "n_mun": int(len(panel)), "agesex_rows_wakayama": int(wk.sum()),
       "s0": s0, "s1": s1, "s3": s3, "s4": s4, "c1": c1, "c2": c2}
(PROC / "m8_agesex_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
panel.to_csv(PROC / "m8_panel.csv", index=False, encoding="utf-8-sig")
print("saved")
