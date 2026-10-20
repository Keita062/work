"""ステップ5（改善候補を絞る）・ステップ6（改善余地の確認）

設計：01_docs/分析の流れ_1001_ステップ1-4結果と5-6設計.md（西村氏 0929step5-7提案・0930残論点）
- ステップ5：自治体×属性×年度のパネルで固定効果モデルを推定する。
    ① 重点カテゴリーの主観 = β1·X + セル効果 + 年度効果
    ② 幸福度             = β2·X + セル効果 + 年度効果
  X は年度で値が変わる客観スコア（個別KPIは1時点のため使えない：設計書 §3-1 の選択肢 c）。
  セル＝地域×年代×性別。回答者数で重みづけ、標準誤差は地域でクラスタリング。
  0930案どおり年代×性別の行で、回答者数 n≥5/10/20/30/50 の感度分析を行う。参考に「全体」行 n≥30 も推定する。
- ステップ6：残った X について、ターゲットの市の現状値を同じ類型の分布と比べ、改善余地を出す。

出力：04_data/processed/m7_step5_6_results.json
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import wb_typology
from wb_common import PROC, ROOT, load_wb

YEARS = [2023, 2024, 2025]          # 客観の2026年度は2025年度の再掲なので使わない
THRESHOLDS = [5, 10, 20, 30, 50]
FOCUS = {"地域とのつながり": "和歌山市", "自己効力感": "田辺市", "健康状態": "和歌山市・田辺市"}
TARGETS = {"和歌山市": "都市部", "田辺市": "農山村の小都市"}


def load():
    """WB_CACHE（pickle置き場）があればそちらを読む。Excelの読込は数分かかる"""
    cache = os.environ.get("WB_CACHE")
    if cache:
        return pd.read_pickle(f"{cache}/subj.pkl"), pd.read_pickle(f"{cache}/obj.pkl")
    return load_wb()


# ---------------------------------------------------------------- ステップ0の類型（再現）
def typology():
    """ステップ0の類型（wb_typology）。対象は m8 と同じ年代×性別・回答者数5人以上の680市町村"""
    p = pd.read_csv(PROC / "m8_panel.csv", dtype={"code": str})
    t, _, _, _ = wb_typology.typology(p["code"].values)
    return pd.DataFrame({"地域ID": p["地域ID"].values, "name": p["name"].values, "類型": t.values})


# ---------------------------------------------------------------- ステップ5
def fe_fit(df, y, x, w="回答者数", cell="cell", cluster="地域ID"):
    """セル固定効果を加重平均除去で吸収し、年度ダミーを入れた WLS（FWL）。地域クラスタSE"""
    df = df[[y, x, w, cell, cluster, "年度"]].dropna()
    df = df[df.groupby(cell)[cell].transform("size") >= 2]     # 1年しかないセルは識別に寄与しない
    if df[cluster].nunique() < 30:
        return None
    yd = pd.get_dummies(df["年度"], prefix="y", drop_first=True).astype(float)
    cols = pd.concat([df[[y, x]], yd], axis=1)
    ww = df[w].astype(float)
    m = cols.mul(ww, axis=0).groupby(df[cell]).transform("sum").div(ww.groupby(df[cell]).transform("sum"), axis=0)
    dm = cols - m
    X = dm.drop(columns=[y])
    if dm[x].abs().max() < 1e-9:
        return None
    r = sm.WLS(dm[y], X, weights=ww).fit(cov_type="cluster", cov_kwds={"groups": df[cluster].values})
    lo, hi = r.conf_int().loc[x]
    return {"b": float(r.params[x]), "se": float(r.bse[x]), "p": float(r.pvalues[x]),
            "lo": float(lo), "hi": float(hi), "rows": int(len(df)), "regions": int(df[cluster].nunique())}


def step5(subj, obj, xcols):
    oy = obj[obj["年度"].isin(YEARS)][["年度", "地域ID"] + xcols]
    base = subj[subj["年度"].isin(YEARS) & ~subj["市区町村"].str.contains(r"区$") & (subj["回答者数"] > 0)]
    out = {}
    specs = {f"年代×性別 n≥{t}": (base[(base["区分"] == "年代×性別") & (base["回答者数"] >= t)
                                      & base["性別"].isin(["男性", "女性"])], t) for t in THRESHOLDS}
    specs["全体 n≥30（参考）"] = (base[(base["区分"] == "全体") & (base["回答者数"] >= 30)], 30)
    for spec, (d, _) in specs.items():
        d = d.merge(oy, on=["年度", "地域ID"])
        d["cell"] = d["地域ID"].astype(str) + "|" + d["年代"].fillna("全") + "|" + d["性別"].fillna("全")
        res = {}
        for ycol in [f"主観_{k}" for k in FOCUS] + ["幸福度"]:
            rows = []
            for x in xcols:
                r = fe_fit(d, ycol, x)
                if r:
                    rows.append({"x": x[3:], **r})
            if rows:
                q = multipletests([r["p"] for r in rows], method="fdr_bh")[1]
                for r, qq in zip(rows, q):
                    r["q"] = float(qq)
            res[ycol.replace("主観_", "")] = rows
        out[spec] = {"rows": int(len(d)), "regions": int(d["地域ID"].nunique()), "cells": int(d["cell"].nunique()),
                     "est": res}
        print(spec, len(d), d["地域ID"].nunique(), flush=True)
    return out


def classify(est, focus, alpha=0.05):
    """① 重点カテゴリーの主観、② 幸福度の両方で β>0 かつ p<alpha を最終候補とする"""
    a = {r["x"]: r for r in est[focus]}
    b = {r["x"]: r for r in est["幸福度"]}
    rows = []
    for x in a:
        if x not in b:
            continue
        s1 = a[x]["b"] > 0 and a[x]["p"] < alpha
        s2 = b[x]["b"] > 0 and b[x]["p"] < alpha
        rows.append({"x": x, "b1": a[x]["b"], "p1": a[x]["p"], "q1": a[x]["q"], "lo1": a[x]["lo"], "hi1": a[x]["hi"],
                     "b2": b[x]["b"], "p2": b[x]["p"], "q2": b[x]["q"], "lo2": b[x]["lo"], "hi2": b[x]["hi"],
                     "both": bool(s1 and s2), "only1": bool(s1 and not s2), "only2": bool(s2 and not s1)})
    return rows


# ---------------------------------------------------------------- ステップ6
def step6(obj, typ, xcols, cands):
    o = obj[obj["年度"].isin([2023, 2025])][["年度", "地域ID"] + xcols]
    o25 = o[o["年度"] == 2025].merge(typ, on="地域ID")
    o23 = o[o["年度"] == 2023].set_index("地域ID")
    chg = (o[o["年度"] == 2025].set_index("地域ID")[xcols] - o23[xcols]).loc[typ["地域ID"]]
    out = {}
    for city, tname in TARGETS.items():
        grp = o25[o25["類型"] == tname]
        me = o25[o25["name"] == city].iloc[0]
        rows = []
        for x in cands:
            c = f"客観_{x}"
            v = float(me[c])
            p75 = float(grp[c].quantile(.75))
            rows.append({"x": x, "value": v, "median": float(grp[c].median()), "p75": p75,
                         "pct": float((grp[c] < v).mean() * 100), "room": max(0.0, p75 - v),
                         "own_change": float(chg.loc[me["地域ID"], c]),
                         "nat_p90_change": float(chg[c].quantile(.9)),
                         "type_n": int(len(grp))})
        out[city] = rows
    return out


def main():
    subj, obj = load()
    typ = typology()
    sizes = typ["類型"].value_counts().to_dict()
    print(sizes)
    xcols = [c for c in obj.columns if c.startswith("客観_")]
    o = obj[obj["年度"].isin(YEARS)]
    within = {c: float(((o[c] - o.groupby("地域ID")[c].transform("mean")).var()) / o[c].var()) for c in xcols}
    xvar = [c for c in xcols if within[c] >= 1e-3]  # 3年間動かない4分野と、ほぼ動かない教育機会の豊かさを除く
    s5 = step5(subj, obj, xvar)
    sel = {}
    for spec, v in s5.items():
        sel[spec] = {f: classify(v["est"], f) for f in FOCUS}
    cands = sorted({r["x"] for spec in sel.values() for f in spec.values() for r in f if r["both"]})
    s6 = step6(obj, typ, xcols, sorted(set(cands) | {k for k in FOCUS}))
    res = {"typology_sizes": sizes, "within_share": {k[3:]: v for k, v in within.items()},
           "x_used": [x[3:] for x in xvar], "step5": s5, "selection": sel, "step6": s6}
    (PROC / "m7_step5_6_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    main()
