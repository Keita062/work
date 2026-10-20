#!/usr/bin/env python3
"""
Step 0'-b（M2先行プロトタイプ・可視化）: ソフト化GMMで「層状の所属確率」を試作

soften_lca_experiment.py の結果を踏まえ、分離度を保ちつつ層化する設定
（cov=full, reg_covar=0.1 を既定・要Step4'確定）で所属確率を再推定し、
現行(diag,1e-6)の一峰版と比較できる layered 所属確率レーダー（製造業デモ）を出力する。

出力:
  data/processed/class_membership_softened_by_pref.csv（3業種・年平均の所属確率）
  data/processed/figures/pref_membership_radar_softened_製造業.png
実行: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/model/soften_lca_apply.py
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

TARGETS = ["製造業", "卸売業，小売業", "サービス業他"]
K = 5
COV, REG = "full", 0.1  # 層化と分離のバランス（要Step4'確定）
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
PCOLS = [f"p{c+1}" for c in range(K)]
ang5 = np.linspace(0, 2 * np.pi, K, endpoint=False)
ang5c = np.append(ang5, ang5[0])


def main():
    df = pd.read_csv(PROC / "panel_pca_scores.csv", encoding="utf-8")
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    all_pref = []
    for ind in TARGETS:
        d = df[df["industry"] == ind].reset_index(drop=True)
        X = d[pc_cols].to_numpy(dtype=float)
        gm = GaussianMixture(n_components=K, covariance_type=COV, reg_covar=REG,
                             n_init=30, max_iter=500, random_state=0)
        gm.fit(X)
        proba = gm.predict_proba(X)
        py = d[["prefecture", "year"]].copy()
        py[PCOLS] = proba.round(4)
        agg = py.groupby("prefecture")[PCOLS].mean().round(4)
        agg["支配クラス"] = agg[PCOLS].to_numpy().argmax(axis=1) + 1
        agg["支配割合"] = agg[PCOLS].max(axis=1).round(3)
        agg["industry"] = ind
        all_pref.append(agg.reset_index())
        share = float((proba.max(axis=1) < 0.9).mean())
        print(f"{ind}: 層化(所属確率<0.9)={share:.2f} / 平均支配割合={agg['支配割合'].mean():.2f}")

    out = pd.concat(all_pref)
    out.to_csv(PROC / "class_membership_softened_by_pref.csv", index=False, encoding="utf-8")

    # 製造業デモ：層状の所属確率レーダー（支配割合の低い＝層状の県を上位に）
    demo = out[out.industry == "製造業"].sort_values("支配割合").reset_index(drop=True)
    n = len(demo); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.5 * nrow), subplot_kw={"polar": True})
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off"); continue
        r = demo.iloc[i]
        vals = np.append(r[PCOLS].to_numpy(float), r[PCOLS].to_numpy(float)[0])
        col = CLASS_COLORS[int(r["支配クラス"]) - 1]
        ax.plot(ang5c, vals, color=col, linewidth=1.7); ax.fill(ang5c, vals, color=col, alpha=0.30)
        ax.set_ylim(0, 1); ax.set_xticks(ang5)
        ax.set_xticklabels([f"C{c+1}" for c in range(K)], fontsize=6.5, color="#52514e")
        ax.set_yticklabels([]); ax.grid(color="#e1e0d9", linewidth=0.5)
        ax.set_title(f"{r['prefecture']}\n主C{int(r['支配クラス'])}={r['支配割合']:.2f}", fontsize=7.5, pad=4)
    handles = [plt.Line2D([0], [0], color=CLASS_COLORS[c], lw=6, label=f"クラス{c+1}") for c in range(K)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"製造業（M2ソフト化プロトタイプ: cov={COV}, reg_covar={REG}）\n"
                 f"都道府県別 クラス所属確率レーダー（支配割合の低い＝層状の県を上位配置）",
                 fontsize=13, y=1.004)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    fig.savefig(FIG / "pref_membership_radar_softened_製造業.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("✓ 図: figures/pref_membership_radar_softened_製造業.png")
    print("✓ 保存: class_membership_softened_by_pref.csv")


if __name__ == "__main__":
    main()
