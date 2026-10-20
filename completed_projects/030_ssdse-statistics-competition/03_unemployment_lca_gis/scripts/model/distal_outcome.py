#!/usr/bin/env python3
"""
Phase 5: distal outcome（完全失業率）とLCAクラスの関連 ＋ 交絡調整

- 完全失業率は県レベル（全産業, 2020断面）をフォールバックとして使用
  （国勢調査は失業者を産業別に分類しないため。要件定義書_改訂版 §3.6・決着⑤）。
- クラスは業種別LCAの割当（2020断面, 47県/業種）。
- 交絡（人口密度・高齢化率・有効求人倍率）を共変量として調整（指標には混ぜない）。
  → 調整前（クラス別平均）と、線形回帰による調整済みのクラス効果を報告。
- 観察データ・単一断面のため「調整済みの関連」まで（実験的因果は主張しない）。

実行: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/model/distal_outcome.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
PROC = ROOT / "data" / "processed"

panel = pd.read_csv(PROC / "panel_2020.csv")
assign = pd.read_csv(PROC / "lca_class_assignments.csv")
assign = assign[assign["year"] == 2020]

df = panel.merge(assign[["prefecture", "industry", "class"]],
                 on=["prefecture", "industry"], how="inner")

COVS = ["人口密度", "高齢化率", "有効求人倍率"]
out_rows = []
for ind, d in df.groupby("industry"):
    print(f"\n=== {ind} ===")
    # 調整前: クラス別 完全失業率 平均
    g = d.groupby("class")["完全失業率"].agg(["mean", "count"]).round(3)
    print("クラス別 完全失業率（調整前, %）:")
    print(g.to_string())
    # 調整済み: 完全失業率 ~ classダミー(ref=最小クラス番号) + 共変量
    classes = sorted(d["class"].unique())
    ref = classes[0]
    X_cols = []
    X = [np.ones(len(d))]
    X_cols.append("切片")
    for c in classes[1:]:
        X.append((d["class"] == c).astype(float).to_numpy())
        X_cols.append(f"class{c}(vs{ref})")
    for cov in COVS:
        X.append(d[cov].to_numpy(dtype=float))
        X_cols.append(cov)
    Xm = np.column_stack(X)
    y = d["完全失業率"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    print("調整済み クラス効果（完全失業率への寄与, %ポイント; 共変量調整後）:")
    for name, b in zip(X_cols, beta):
        if name.startswith("class"):
            print(f"  {name}: {b:+.3f}")
    for name, b in zip(X_cols, beta):
        if name in COVS:
            out_rows.append({"industry": ind, "term": name, "coef": round(float(b), 4)})
    for c in classes:
        out_rows.append({"industry": ind, "term": f"class{c}_mean_unadj",
                         "coef": round(float(g.loc[c, "mean"]), 3)})

pd.DataFrame(out_rows).to_csv(PROC / "distal_outcome_summary.csv", index=False, encoding="utf-8")
print("\n✓ 保存: distal_outcome_summary.csv")
print("※ 完全失業率は県レベル(2020断面)フォールバック。単一断面のため調整済みの関連まで（因果は主張しない）。")
