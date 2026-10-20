"""M4 個別KPIによる説明力の確認（上位5分野）

問い: 主観スコアの地域差を、分野の個別KPIでどこまで説明できるか。
      説明できるなら、どのKPIを動かせばよいか（ルートA）。できないなら何が足りないか（ルートB）。

入力: data/digital_wellbeing_shikuchoson_2022_2026.xlsx（主観・客観スコア）
      data/processed/kpi_municipal.csv（build_kpi_panel.py の出力）
出力: data/processed/m4_results.json（アーティファクトに載せる数字）
      data/processed/m4_panel.csv（分析に使った 1行＝1市町村 の表）

分析単位: 市町村。個別KPIが2020〜2024年のどれか1時点しかないため、
主観スコアは2023〜26年度の回答者数加重平均にして、1市町村1行にする。
対象はM0と同じ（主観「全体」行・回答者数30以上・客観あり・区を除く）678市町村。
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from wb_common import PROC, WAKAYAMA, load_wb, m0_sample

RNG = np.random.default_rng(20260922)

# 分野ごとの個別KPI。sign は「値が大きいほど望ましい」なら +1（ガイドブックの向き）
KPIS = {
    "健康状態": [
        ("healthy_life_m", "健康寿命（男性・近似）", +1),
        ("healthy_life_f", "健康寿命（女性・近似）", +1),
    ],
    "自己効力感": [
        ("net_migration_rate", "転入超過割合", +1),
        ("unemployed_per_pop", "完全失業者数（人口比）", -1),
    ],
    "住宅環境": [
        ("floor_area_per_dwelling", "住宅当たり延べ面積", +1),
        ("rent_per_m2", "専用住宅面積あたり家賃", -1),
        ("detached_owner_share", "一戸建の持ち家の割合", +1),
    ],
    "地域とのつながり": [
        ("suicide_per_100k", "人口あたり自殺者数", -1),
        ("extended_family_share", "拡大家族世帯割合", +1),
        ("married_share", "既婚者割合", +1),
        ("elderly_single_share", "高齢単身世帯割合", -1),
        ("political_org_per_10k", "人口あたり政治団体等の数", +1),
        ("religious_est_per_10k", "人口あたり宗教の事業所数", +1),
        ("npo_per_10k", "人口あたりNPOの数", +1),
    ],
    "公共空間": [
        ("park_walk_cover", "公園緑地徒歩圏人口カバー率", +1),
        ("park_area_per_capita", "人口あたり公園面積", +1),
        ("sidewalk_rate", "歩道設置率", +1),
        ("walkable_city", "ウォーカブル推進都市", +1),
    ],
}
ENV = [("log_pop", "人口（対数）"), ("aging_rate", "高齢化率"), ("fiscal_index", "財政力指数")]
FOCUS = [("田辺市", "自己効力感"), ("橋本市", "健康状態")]


# ---------------------------------------------------------------- データ
subj, obj = load_wb()
m = m0_sample(subj, obj)

cats = list(KPIS)
w = m["回答者数"]
agg = {}
for c in [f"主観_{k}" for k in cats] + [f"客観_{k}" for k in cats] + ["幸福度"]:
    agg[c] = (m[c] * w).groupby(m["地域ID"]).sum() / w.groupby(m["地域ID"]).sum()
panel = pd.DataFrame(agg)
first = m.drop_duplicates("地域ID").set_index("地域ID")
panel["pref"] = first["都道府県"]
panel["name"] = first["市区町村"].str.replace("全域", "", regex=False)
panel["n_resp"] = w.groupby(m["地域ID"]).sum()
panel["n_years"] = m.groupby("地域ID").size()

kpi = pd.read_csv(PROC / "kpi_municipal.csv", dtype={"code": str})
kpi["log_pop"] = np.log(kpi["pop2020"])
panel = panel.reset_index().merge(kpi, on=["pref", "name"], how="left", validate="1:1")
assert panel["code"].notna().all()
panel.to_csv(PROC / "m4_panel.csv", index=False, encoding="utf-8-sig")


def z(df):
    return (df - df.mean()) / df.std(ddof=0)


def cv_r2(X, y, reps=20, k=5):
    """5分割交差検証のR²（分割を20回変えて平均）。1市町村1行なので市町村単位の分割になる"""
    X, y = X.values, y.values
    out = []
    for r in range(reps):
        pred = np.empty_like(y)
        for tr, te in KFold(k, shuffle=True, random_state=r).split(X):
            pred[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
        out.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out))


def boot_coef(X, y, B=500):
    n = len(y)
    Xv, yv = X.values, y.values
    bs = np.array([LinearRegression().fit(Xv[i], yv[i]).coef_ for i in (RNG.integers(0, n, n) for _ in range(B))])
    return np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


env_cols = [c for c, _ in ENV]
res = {"n_municipalities": int(len(panel)), "categories": {}}
for cat, items in KPIS.items():
    cols = [c for c, _, _ in items]
    d = panel.dropna(subset=cols + env_cols + [f"主観_{cat}", f"客観_{cat}"]).copy()
    y = z(d[f"主観_{cat}"])
    Z = z(d[cols + env_cols + [f"客観_{cat}"]])

    # 個別KPIが公式の客観スコアをどれだけ再現しているか（取得データの妥当性の確認）
    r2_obj_from_kpi = LinearRegression().fit(Z[cols], z(d[f"客観_{cat}"])).score(Z[cols], z(d[f"客観_{cat}"]))

    fits = {
        "env": cv_r2(Z[env_cols], y),
        "env_obj": cv_r2(Z[env_cols + [f"客観_{cat}"]], y),
        "kpi": cv_r2(Z[cols], y),
        "env_kpi": cv_r2(Z[cols + env_cols], y),
    }
    lr = LinearRegression().fit(Z[cols + env_cols], y)
    lo, hi = boot_coef(Z[cols + env_cols], y)
    kp = []
    for i, (c, lab, sign) in enumerate(items):
        # 環境変数を除いた偏相関（KPIと主観の両方から環境変数の影響を取り除いた残差どうしの相関）
        ex = LinearRegression().fit(Z[env_cols], Z[c]); ey = LinearRegression().fit(Z[env_cols], y)
        pc = np.corrcoef(Z[c] - ex.predict(Z[env_cols]), y - ey.predict(Z[env_cols]))[0, 1]
        kp.append({
            "col": c, "label": lab, "sign": sign,
            "r_subj": float(np.corrcoef(d[c], d[f"主観_{cat}"])[0, 1]),
            "r_obj": float(np.corrcoef(d[c], d[f"客観_{cat}"])[0, 1]),
            "partial_r_subj": float(pc),
            "beta": float(lr.coef_[i]), "ci": [float(lo[i]), float(hi[i])],
            # 主観スコア（偏差値）の変化 ÷ KPI 1単位の変化（環境変数・他KPIを固定）
            "slope": float(lr.coef_[i] * d[f"主観_{cat}"].std(ddof=0) / d[c].std(ddof=0)),
            "median": float(d[c].median()),
            "significant": bool(lo[i] > 0 or hi[i] < 0),
            "sign_ok": bool(np.sign(lr.coef_[i]) == sign),
        })
    env_beta = [{"label": lab, "beta": float(lr.coef_[len(cols) + j]),
                 "ci": [float(lo[len(cols) + j]), float(hi[len(cols) + j])]} for j, (_, lab) in enumerate(ENV)]

    # 和歌山6市：各KPIの全国順位（望ましい側が上位＝パーセンタイルが高い）
    wk = {}
    for city in WAKAYAMA:
        row = panel[(panel["pref"] == "和歌山県") & (panel["name"] == city)]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = {}
        for c, lab, sign in items:
            col = panel[c].dropna()
            v = row[c]
            if pd.isna(v):
                vals[c] = None
                continue
            pct = float(((col * sign) < (v * sign)).mean() * 100 + ((col * sign) == (v * sign)).mean() * 50)
            vals[c] = {"value": float(v), "pct": pct}
        # 環境変数＋KPIのモデルが予測する主観との差（実際 − 予測、偏差値ポイント）
        hit = d.index[(d["pref"] == "和歌山県") & (d["name"] == city)]
        resid = None
        if len(hit):
            pred_z = lr.predict(Z.loc[hit, cols + env_cols])[0]
            resid = float(row[f"主観_{cat}"] - (d[f"主観_{cat}"].mean() + pred_z * d[f"主観_{cat}"].std(ddof=0)))
        wk[city] = {"kpi": vals, "subj": float(row[f"主観_{cat}"]), "obj": float(row[f"客観_{cat}"]), "resid": resid}

    res["categories"][cat] = {
        "n": int(len(d)), "r2_obj_from_kpi": float(r2_obj_from_kpi), "cv_r2": fits,
        "kpi_gain": fits["env_kpi"] - fits["env"], "obj_gain": fits["env_obj"] - fits["env"],
        "kpis": kp, "env_beta": env_beta, "wakayama": wk,
    }

# 焦点の2組：比べる相手＝環境変数が近い全国の市町村（マハラノビス距離の近い順に30）
foc = {}
E = panel.dropna(subset=env_cols)
Ez = z(E[env_cols])
VI = np.linalg.inv(np.cov(Ez.values.T))
for city, cat in FOCUS:
    idx = E.index[(E["pref"] == "和歌山県") & (E["name"] == city)][0]
    dv = Ez.values - Ez.loc[idx].values
    dist = np.sqrt(np.einsum("ij,jk,ik->i", dv, VI, dv))
    peers = E.assign(dist=dist).drop(idx).nsmallest(30, "dist")
    items = KPIS[cat]
    foc[f"{city}×{cat}"] = {
        "city": {c: (None if pd.isna(E.loc[idx, c]) else float(E.loc[idx, c])) for c, _, _ in items} | {
            "subj": float(E.loc[idx, f"主観_{cat}"]), "obj": float(E.loc[idx, f"客観_{cat}"])},
        "peers_median": {c: float(peers[c].median()) for c, _, _ in items} | {
            "subj": float(peers[f"主観_{cat}"].median()), "obj": float(peers[f"客観_{cat}"].median())},
        "peers_top": peers.nlargest(5, f"主観_{cat}")[["pref", "name", f"主観_{cat}", f"客観_{cat}"] + [c for c, _, _ in items]]
        .round(3).to_dict("records"),
        "subj_rank_in_peers": int((peers[f"主観_{cat}"] > E.loc[idx, f"主観_{cat}"]).sum() + 1),
    }
res["focus"] = foc

(PROC / "m4_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

for cat, r in res["categories"].items():
    f_ = r["cv_r2"]
    print(f"\n## {cat} n={r['n']}  KPI→客観R2={r['r2_obj_from_kpi']:.2f}  "
          f"CV R2 env={f_['env']:.3f} env+客観={f_['env_obj']:.3f} kpi={f_['kpi']:.3f} env+kpi={f_['env_kpi']:.3f}")
    for k in r["kpis"]:
        print(f"  {k['label']:<22} r主観={k['r_subj']:+.2f} r客観={k['r_obj']:+.2f} 偏r={k['partial_r_subj']:+.2f} "
              f"β={k['beta']:+.2f} [{k['ci'][0]:+.2f},{k['ci'][1]:+.2f}] {'*' if k['significant'] else ' '} 向き{'○' if k['sign_ok'] else '×'}")
    for e in r["env_beta"]:
        print(f"  (環境) {e['label']:<10} β={e['beta']:+.2f} [{e['ci'][0]:+.2f},{e['ci'][1]:+.2f}]")
print(json.dumps(res["focus"], ensure_ascii=False, indent=1)[:4000])
