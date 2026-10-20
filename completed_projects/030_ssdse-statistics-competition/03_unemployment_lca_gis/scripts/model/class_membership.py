#!/usr/bin/env python3
"""
Phase 3b: 潜在クラスの都道府県別「所属割合」（事後確率）＋完全失業率課題の可視化

- industry_lca.py と同一の GMM（K=5, diag, n_init=50, random_state=0）を業種別に再推定
  （決定的なので lca_class_assignments.csv とクラス番号は一致）。
- 各 都道府県×年 の事後確率 predict_proba を取得。
- 都道府県ごとに、年（2019-2025）平均の事後確率＝「クラス所属割合」を算出（合計1）。
- distal（クラス別 完全失業率）を重みに使い、
  「所属割合×クラス失業率」の加重＝各県の“失業タイプ寄与”を算出（課題の浮き彫り）。

入力: panel_pca_scores.csv, distal_outcome_summary.csv
出力:
  class_membership_pref_year.csv … 県×業種×年 の事後確率（p1..p5）
  class_membership_by_pref.csv   … 県×業種 の所属割合（年平均）＋支配クラス＋加重失業タイプ
  figures/class_membership_<industry>.png … 県別 所属割合の積み上げ棒（失業寄与降順）

実行: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/model/class_membership.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

matplotlib.rcParams["font.sans-serif"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent.parent
PROC = ROOT / "data" / "processed"
FIG = PROC / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TARGETS = ["製造業", "卸売業，小売業", "サービス業他"]
K = 5
# dataviz 既定カテゴリ配色（light, 隣接CVD ΔE9.1 で検証済）。クラス1..5。
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def class_unemp_means() -> dict:
    """distal_outcome_summary.csv から業種別 class{c}_mean_unadj を読む。"""
    d = pd.read_csv(PROC / "distal_outcome_summary.csv")
    out = {}
    for _, r in d[d["term"].str.startswith("class")].iterrows():
        c = int(r["term"].replace("class", "").split("_")[0])
        out.setdefault(r["industry"], {})[c] = float(r["coef"])
    return out


def main():
    df = pd.read_csv(PROC / "panel_pca_scores.csv", encoding="utf-8")
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    unemp = class_unemp_means()

    py_rows, pref_rows = [], []
    for ind in TARGETS:
        d = df[df["industry"] == ind].reset_index(drop=True)
        X = d[pc_cols].to_numpy(dtype=float)
        gm = GaussianMixture(n_components=K, covariance_type="diag",
                             n_init=50, max_iter=500, random_state=0)
        gm.fit(X)
        proba = gm.predict_proba(X)          # 県×年 の事後確率
        pcols = [f"p{c+1}" for c in range(K)]
        dd = d[["prefecture", "year"]].copy()
        dd[pcols] = proba.round(4)
        dd["industry"] = ind
        py_rows.append(dd)

        # 県ごとに年平均＝所属割合
        agg = dd.groupby("prefecture")[pcols].mean().round(4)
        cmean = unemp.get(ind, {})
        w = np.array([cmean.get(c + 1, np.nan) for c in range(K)])
        agg["支配クラス"] = agg[pcols].to_numpy().argmax(axis=1) + 1
        agg["支配割合"] = agg[pcols].max(axis=1).round(3)
        # 加重 失業タイプ寄与 = Σ 所属割合 × クラス失業率
        agg["加重失業率"] = (agg[pcols].to_numpy() * w).sum(axis=1).round(3)
        agg["industry"] = ind
        pref_rows.append(agg.reset_index())

        # ---- 図: 県別 所属割合の積み上げ棒（加重失業率の降順で上位ほど高リスク）----
        a = agg.reset_index().sort_values("加重失業率", ascending=True)  # barh下から上へ
        y = np.arange(len(a))
        left = np.zeros(len(a))
        fig, ax = plt.subplots(figsize=(9, 12))
        for c in range(K):
            vals = a[f"p{c+1}"].to_numpy()
            ax.barh(y, vals, left=left, height=0.72, color=CLASS_COLORS[c],
                    edgecolor="#fcfcfb", linewidth=1.2, label=f"クラス{c+1}")
            left += vals
        ax.set_yticks(y)
        ax.set_yticklabels([f"{p}  ({u:.1f}%)" for p, u in
                            zip(a["prefecture"], a["加重失業率"])], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("クラス所属割合（年平均の事後確率）", fontsize=10)
        ax.set_title(f"{ind}：都道府県別 潜在クラス所属割合\n"
                     f"（右ラベル＝所属割合×クラス失業率の加重, 上ほど高リスク）",
                     fontsize=12, pad=12)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(FIG / f"class_membership_{ind}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ 図: figures/class_membership_{ind}.png")

    pd.concat(py_rows).to_csv(PROC / "class_membership_pref_year.csv",
                              index=False, encoding="utf-8")
    out = pd.concat(pref_rows)
    cols = ["industry", "prefecture", "p1", "p2", "p3", "p4", "p5",
            "支配クラス", "支配割合", "加重失業率"]
    out[cols].to_csv(PROC / "class_membership_by_pref.csv",
                     index=False, encoding="utf-8")
    print("\n✓ 保存: class_membership_pref_year.csv, class_membership_by_pref.csv")

    # 検証: 支配クラスが lca_class_assignments(2020) と整合するか
    chk = pd.read_csv(PROC / "lca_class_assignments.csv")
    chk = chk[chk["year"] == 2020][["prefecture", "industry", "class"]]
    m = out.reset_index()[["industry", "prefecture", "支配クラス"]].merge(
        chk, on=["industry", "prefecture"])
    agree = (m["支配クラス"] == m["class"]).mean() * 100
    print(f"※ 支配クラス vs 2020ハード割当の一致率 = {agree:.1f}%（決定的再推定の整合確認）")


if __name__ == "__main__":
    main()
