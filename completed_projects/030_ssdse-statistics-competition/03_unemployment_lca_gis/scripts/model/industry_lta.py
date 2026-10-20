#!/usr/bin/env python3
"""
Phase 4: LTA（潜在推移分析）— クラス遷移の動学記述

方針: プールGMMで全ウェーブ共通のクラス定義（＝測定不変）を用い、
      連続ウェーブ間のクラス遷移確率 P(class_{t+1} | class_t) を
      都道府県・ウェーブ対にわたり集計（一次・定常遷移）。
      要件定義書_改訂版 §3.4・第9章。

入力: lca_class_assignments.csv（prefecture, industry, year, class）
出力: lta_transition_<industry>.csv、figures/lta_<industry>.png

実行: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/model/industry_lta.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent.parent
PROC = ROOT / "data" / "processed"
FIG = PROC / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def transition_matrix(d: pd.DataFrame, k: int) -> np.ndarray:
    """連続ウェーブ対（同一都道府県）でのクラス遷移回数→行方向で正規化。"""
    waves = sorted(d["year"].unique())
    counts = np.zeros((k, k))
    for pref, g in d.groupby("prefecture"):
        g = g.set_index("year")
        for t0, t1 in zip(waves[:-1], waves[1:]):
            if t0 in g.index and t1 in g.index:
                c0, c1 = int(g.loc[t0, "class"]), int(g.loc[t1, "class"])
                counts[c0 - 1, c1 - 1] += 1
    row = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(row > 0, counts / row, 0.0)
    return P, counts


def plot_heatmap(P: np.ndarray, title: str, path: Path):
    k = P.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([f"→{i+1}" for i in range(k)])
    ax.set_yticklabels([f"クラス{i+1}" for i in range(k)])
    ax.set_xlabel("翌ウェーブのクラス"); ax.set_ylabel("現ウェーブのクラス")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center",
                    color="white" if P[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title(title, fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, label="遷移確率")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    src = PROC / "lca_class_assignments.csv"
    if not src.exists():
        raise SystemExit("先に industry_lca.py を実行してください（lca_class_assignments.csv 不在）")
    df = pd.read_csv(src, encoding="utf-8")
    out = []
    for ind, d in df.groupby("industry"):
        waves = sorted(d["year"].unique())
        if len(waves) < 2:
            print(f"[skip] {ind}: ウェーブ{len(waves)}（遷移に2点以上必要）")
            continue
        k = int(d["class"].max())
        P, counts = transition_matrix(d, k)
        print(f"\n=== {ind}: ウェーブ {waves} / {k}クラス ===")
        print(np.round(P, 2))
        diag = np.diag(P).mean()
        print(f"  対角平均（滞留確率）= {diag:.2f}  ※高いほどクラスが時間的に安定")
        plot_heatmap(P, f"{ind}：クラス遷移確率（一次・定常, {waves[0]}→{waves[-1]}）",
                     FIG / f"lta_{ind}.png")
        for i in range(k):
            for j in range(k):
                out.append({"industry": ind, "from_class": i + 1, "to_class": j + 1,
                            "prob": round(P[i, j], 4), "n": int(counts[i, j])})
    if out:
        pd.DataFrame(out).to_csv(PROC / "lta_transition.csv", index=False, encoding="utf-8")
        print("\n✓ 保存: lta_transition.csv, figures/lta_<industry>.png")


if __name__ == "__main__":
    main()
