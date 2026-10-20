#!/usr/bin/env python3
"""
Phase 3: 業種別LCA（実装は連続主成分のGaussian混合＝LPA、呼称はLCAで統一）

入力: 構成概念PCAスコアのパネル（prefecture, industry, year, PC1.., PC2..,..）
処理: ターゲット業種ごとに、主成分スコアで K=2〜7 の Gaussian混合を推定。
      AIC/BIC/相対エントロピーを算出し、K=5採否ルールを適用（要件定義書_改訂版 §5, 第11章④）。
出力: クラス割当CSV・K別統計量CSV・クラス×主成分のレーダー図。

実行: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/model/industry_lca.py [panel.csv]
"""
import sys
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
K_FIXED = 5           # レーダー目的で固定
K_RANGE = range(2, 8)  # 2〜7で統計量を報告
ENTROPY_MIN = 0.5     # K=5採用の相対エントロピー目安


def relative_entropy(proba: np.ndarray) -> float:
    """相対エントロピー En = 1 - sum(-p log p) / (N log K)。1に近いほど分離が明瞭。"""
    n, k = proba.shape
    if k < 2:
        return 1.0
    p = np.clip(proba, 1e-12, 1.0)
    e = -np.sum(p * np.log(p)) / (n * np.log(k))
    return float(1.0 - e)


def fit_industry(df_ind: pd.DataFrame, pc_cols: list[str]):
    X = df_ind[pc_cols].to_numpy(dtype=float)
    stats = []
    models = {}
    for k in K_RANGE:
        gm = GaussianMixture(n_components=k, covariance_type="diag",
                             n_init=50, max_iter=500, random_state=0)
        gm.fit(X)
        proba = gm.predict_proba(X)
        labels = gm.predict(X)
        stats.append({
            "K": k, "AIC": round(gm.aic(X), 1), "BIC": round(gm.bic(X), 1),
            "converged": bool(gm.converged_),
            "min_class_n": int(np.min(np.bincount(labels, minlength=k))),
            "rel_entropy": round(relative_entropy(proba), 3),
        })
        models[k] = (gm, labels, proba)
    stats_df = pd.DataFrame(stats)
    # K=5採否判定
    row5 = stats_df[stats_df["K"] == K_FIXED].iloc[0]
    adopt_k5 = bool(row5["converged"] and row5["min_class_n"] > 0
                    and row5["rel_entropy"] >= ENTROPY_MIN)
    k_star = int(stats_df.loc[stats_df["BIC"].idxmin(), "K"])
    return stats_df, models, adopt_k5, k_star


def radar(profile: pd.DataFrame, pc_cols: list[str], title: str, path: Path):
    labels = pc_cols
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for cls, row in profile.iterrows():
        vals = row[pc_cols].tolist()
        vals += vals[:1]
        ax.plot(ang, vals, label=f"クラス{cls}", linewidth=2)
        ax.fill(ang, vals, alpha=0.08)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=13, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    panel_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROC / "panel_pca_scores.csv"
    df = pd.read_csv(panel_path, encoding="utf-8")
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise SystemExit(f"主成分列(PC*)が見つかりません: {panel_path}")
    print(f"パネル: {len(df)}行, 主成分列={pc_cols}")

    all_assign, all_stats = [], []
    for ind in TARGETS:
        d = df[df["industry"] == ind].reset_index(drop=True)
        if len(d) < 10:
            print(f"[skip] {ind}: レコード{len(d)}件（少なすぎ）")
            continue
        stats_df, models, adopt_k5, k_star = fit_industry(d, pc_cols)
        print(f"\n=== {ind}（{len(d)}レコード）===")
        print(stats_df.to_string(index=False))
        print(f"K=5採用可={adopt_k5} / BIC最小K*={k_star}")
        k_use = K_FIXED
        gm, labels, proba = models[k_use]
        d = d.assign(**{"class": labels + 1, "post_max": proba.max(axis=1).round(3)})
        all_assign.append(d[["prefecture", "industry", "year", "class", "post_max"]])
        stats_df.insert(0, "industry", ind)
        stats_df["adopt_K5"] = adopt_k5
        stats_df["K_star_BIC"] = k_star
        all_stats.append(stats_df)
        # レーダー（クラス×主成分平均）
        prof = d.groupby("class")[pc_cols].mean()
        radar(prof, pc_cols, f"{ind}：LCAクラス別 主成分プロファイル（K={k_use}）",
              FIG / f"lca_radar_{ind}.png")
        print(f"  → 図: figures/lca_radar_{ind}.png")

    if all_assign:
        pd.concat(all_assign).to_csv(PROC / "lca_class_assignments.csv",
                                     index=False, encoding="utf-8")
        pd.concat(all_stats).to_csv(PROC / "lca_k_selection.csv",
                                    index=False, encoding="utf-8")
        print("\n✓ 保存: lca_class_assignments.csv, lca_k_selection.csv")


if __name__ == "__main__":
    main()
