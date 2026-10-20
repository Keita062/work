#!/usr/bin/env python3
"""
Step 0'-b（M2先行プロトタイプ）: 所属確率を「層」にするためのモデルソフト化 実験

課題C8：連続主成分に対する GMM(diag, K=5) は分離が鋭く、各都道府県の所属確率が
ほぼ単一クラスに集中（一峰）して「層」にならない。
本スクリプトは既存パネル（panel_pca_scores.csv, データ非依存）で、
  covariance_type ∈ {diag, tied, full} × reg_covar ∈ {既定〜強め}
を掃引し、各設定の「層化（支配確率<0.9の割合）」と「分離度（相対エントロピー）」の
トレードオフを比較する。→ Step 4' で採用設定を確定する材料。

出力: data/processed/soften_lca_experiment.csv（業種×設定の比較表）
実行: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/model/soften_lca_experiment.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent.parent
PROC = ROOT / "data" / "processed"

TARGETS = ["製造業", "卸売業，小売業", "サービス業他"]
K = 5
COV_TYPES = ["diag", "tied", "full"]
REG_COVARS = [1e-6, 1e-2, 5e-2, 1e-1, 3e-1]  # 既定(1e-6)→段階的に強める＝境界を緩める


def relative_entropy(proba: np.ndarray) -> float:
    n, k = proba.shape
    if k < 2:
        return 1.0
    p = np.clip(proba, 1e-12, 1.0)
    e = -np.sum(p * np.log(p)) / (n * np.log(k))
    return float(1.0 - e)


def main():
    df = pd.read_csv(PROC / "panel_pca_scores.csv", encoding="utf-8")
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    rows = []
    for ind in TARGETS:
        d = df[df["industry"] == ind].reset_index(drop=True)
        X = d[pc_cols].to_numpy(dtype=float)
        for cov in COV_TYPES:
            for reg in REG_COVARS:
                gm = GaussianMixture(n_components=K, covariance_type=cov,
                                     reg_covar=reg, n_init=30, max_iter=500,
                                     random_state=0)
                try:
                    gm.fit(X)
                except Exception as e:
                    rows.append({"industry": ind, "cov": cov, "reg_covar": reg,
                                 "converged": False, "note": type(e).__name__})
                    continue
                proba = gm.predict_proba(X)
                labels = gm.predict(X)
                maxpost = proba.max(axis=1)
                rows.append({
                    "industry": ind, "cov": cov, "reg_covar": reg,
                    "converged": bool(gm.converged_),
                    "min_class_n": int(np.min(np.bincount(labels, minlength=K))),
                    "rel_entropy": round(relative_entropy(proba), 3),
                    "mean_max_post": round(float(maxpost.mean()), 3),   # 低いほどソフト
                    "share_layered<0.9": round(float((maxpost < 0.9).mean()), 3),  # 高いほど層状
                    "share_layered<0.8": round(float((maxpost < 0.8).mean()), 3),
                })
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "soften_lca_experiment.csv", index=False, encoding="utf-8")

    # 業種別に、層化(share<0.9)の高い順で上位を表示
    pd.set_option("display.unicode.east_asian_width", True)
    for ind in TARGETS:
        sub = out[(out.industry == ind) & (out.get("converged", False) != False)]
        sub = sub[sub["min_class_n"] > 0].sort_values("share_layered<0.9", ascending=False)
        print(f"\n=== {ind}：層化↑順（min_class_n>0 のみ）===")
        cols = ["cov", "reg_covar", "min_class_n", "rel_entropy",
                "mean_max_post", "share_layered<0.9", "share_layered<0.8"]
        print(sub[cols].head(8).to_string(index=False))
    # 既定(diag,1e-6)との対比
    base = out[(out.cov == "diag") & (out.reg_covar == 1e-6)]
    print("\n=== 既定 diag/1e-6（現行）===")
    print(base[["industry", "rel_entropy", "mean_max_post", "share_layered<0.9"]].to_string(index=False))
    print("\n✓ 保存: soften_lca_experiment.csv")


if __name__ == "__main__":
    main()
