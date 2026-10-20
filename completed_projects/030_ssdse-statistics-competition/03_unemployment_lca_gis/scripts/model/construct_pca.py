#!/usr/bin/env python3
"""
Phase 2: 構成概念PCA（block PCA）

構成概念ごとに、その構成概念の観測変数だけを標準化してPCA→第1主成分（構成概念スコア）
に丸める。5構成概念 → 5主成分。要件定義書_改訂版 §3.2 / 方針書 第4章。

取得できた観測変数（都道府県×業種×年）でマッピングを構成（設計の25変数のうち、
公的統計で県×産業が取れたもの）。単一変数の構成概念はPC1＝標準化変数（自明）。

入力: panel_2020.csv（既定）/ 任意のパネルCSV
出力: panel_pca_scores.csv（prefecture, industry, year, PC1_..〜PC5_..）
      pca_loadings.csv（構成概念別 負荷量・寄与率）

実行: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/model/construct_pca.py [panel.csv]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent.parent
PROC = ROOT / "data" / "processed"

# 構成概念 → (観測変数, 向き符号)。向き=+1はそのまま、-1は反転（高いほどその概念が強い向きに）
CONSTRUCTS = {
    "PC1_賃金処遇":   [("所定内給与額", +1)],
    "PC2_雇用安定性": [("勤続年数", +1), ("離職率", -1), ("入職率", -1)],
    "PC3_労働需給":   [("有効求人倍率", +1)],
    "PC4_労働時間負荷": [("超過実労働時間数", +1)],
    "PC5_労働力構成": [("女性比率", +1), ("高齢就業者比率", +1),
                     ("若年就業者比率", +1), ("就業シェア", +1)],
}


def main():
    panel = Path(sys.argv[1]) if len(sys.argv) > 1 else PROC / "panel_2020.csv"
    df = pd.read_csv(panel, encoding="utf-8")
    keys = df[["prefecture", "industry", "year"]].copy()
    scores = {}
    loadings = []
    for cname, members in CONSTRUCTS.items():
        cols = [c for c, s in members if c in df.columns]
        signs = [s for c, s in members if c in df.columns]
        if not cols:
            print(f"[skip] {cname}: 変数なし")
            continue
        X = df[cols].to_numpy(dtype=float) * np.array(signs)  # 向き揃え
        Xz = StandardScaler().fit_transform(X)
        pca = PCA(n_components=1)
        pc = pca.fit_transform(Xz)[:, 0]
        # 符号の向き: 第1主成分が最初の変数と正相関になるよう調整
        if np.corrcoef(pc, Xz[:, 0])[0, 1] < 0:
            pc = -pc
            comp = -pca.components_[0]
        else:
            comp = pca.components_[0]
        scores[cname] = pc
        evr = pca.explained_variance_ratio_[0]
        for col, ld in zip(cols, comp):
            loadings.append({"construct": cname, "variable": col,
                             "loading": round(float(ld), 3),
                             "explained_var_ratio": round(float(evr), 3),
                             "n_vars": len(cols)})
        print(f"{cname}: 変数{cols} 第1主成分寄与率={evr:.2f}")

    out = pd.concat([keys, pd.DataFrame(scores)], axis=1)
    out.to_csv(PROC / "panel_pca_scores.csv", index=False, encoding="utf-8")
    pd.DataFrame(loadings).to_csv(PROC / "pca_loadings.csv", index=False, encoding="utf-8")
    print(f"\n✓ 保存: panel_pca_scores.csv（{len(out)}行, 主成分{list(scores)}）, pca_loadings.csv")


if __name__ == "__main__":
    main()
