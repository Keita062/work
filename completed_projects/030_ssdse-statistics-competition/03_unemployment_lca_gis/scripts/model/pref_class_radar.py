#!/usr/bin/env python3
"""
Phase 3c: 都道府県ごとのレーダーチャート（論文と同じ構成概念5軸のプロファイル）

各都道府県は単一クラスに固定されず、5つの潜在クラスへの所属確率（所属割合）で表される。
本図は「論文のクラス別レーダー」を各都道府県に適用したもの＝
  軸＝5構成概念（賃金/安定/需給/時間/構成）, 値＝その県の主成分プロファイル（年平均）。
色＝支配（最尤）クラス。各県が自分の型（クラス重心）とどれだけ近いか・独自かを一望できる。
所属割合の数値は class_membership_by_pref.csv（p1..p5）に、加重失業率も併記。

入力: panel_pca_scores.csv, class_membership_by_pref.csv
出力:
  figures/pref_profile_radar_<industry>.png … 47県の小レーダー・グリッド（構成概念プロファイル）
  figures/pref_membership_radar_<industry>.png … 参考: クラス所属割合(5クラス軸)の小レーダー

実行: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/model/pref_class_radar.py
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

TARGETS = ["製造業", "卸売業，小売業", "サービス業他"]
K = 5
PC_FULL = ["PC1_賃金処遇", "PC2_雇用安定性", "PC3_労働需給", "PC4_労働時間負荷", "PC5_労働力構成"]
PC_SHORT = ["賃金", "安定", "需給", "時間", "構成"]
PCOLS = [f"p{c+1}" for c in range(K)]
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

ANG = np.linspace(0, 2 * np.pi, len(PC_FULL), endpoint=False)
ANG_C = np.append(ANG, ANG[0])


def grid_radar(rows, draw, title, path, nmini):
    ncol = 6
    nrow = int(np.ceil(nmini / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.5 * nrow),
                             subplot_kw={"polar": True})
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < nmini:
            draw(ax, rows.iloc[i])
        else:
            ax.axis("off")
    handles = [plt.Line2D([0], [0], color=CLASS_COLORS[c], lw=6, label=f"クラス{c+1}")
               for c in range(K)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, fontsize=13, y=1.004)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    pca = pd.read_csv(PROC / "panel_pca_scores.csv", encoding="utf-8")
    mem = pd.read_csv(PROC / "class_membership_by_pref.csv", encoding="utf-8")

    for ind in TARGETS:
        p = pca[pca["industry"] == ind]
        prof = p.groupby("prefecture")[PC_FULL].mean()            # 県平均プロファイル
        m = mem[mem["industry"] == ind].set_index("prefecture")
        prof = prof.join(m[["支配クラス", "支配割合", "加重失業率"] + PCOLS])
        prof = prof.sort_values("加重失業率", ascending=False).reset_index()
        rmin = float(np.floor(p[PC_FULL].to_numpy().min()))
        rmax = float(np.ceil(p[PC_FULL].to_numpy().max()))

        # ---- (1) 構成概念プロファイル・レーダー（論文と同じ5軸）----
        def draw_profile(ax, row):
            vals = row[PC_FULL].to_numpy(dtype=float)
            vals_c = np.append(vals, vals[0])
            col = CLASS_COLORS[int(row["支配クラス"]) - 1]
            ax.plot(ANG_C, vals_c, color=col, linewidth=1.7)
            ax.fill(ANG_C, vals_c, color=col, alpha=0.30)
            ax.set_ylim(rmin, rmax)
            ax.set_xticks(ANG)
            ax.set_xticklabels(PC_SHORT, fontsize=6.5, color="#52514e")
            ax.set_yticklabels([])
            ax.grid(color="#e1e0d9", linewidth=0.5)
            ax.set_title(f"{row['prefecture']}\n加重{row['加重失業率']:.1f}% / 主C{int(row['支配クラス'])}",
                         fontsize=7.5, pad=4)

        grid_radar(prof, draw_profile,
                   f"{ind}（参考・補足）：都道府県別 構成概念プロファイル・レーダー"
                   f"（5軸＝賃金/安定/需給/時間/構成, 色＝支配クラス, 加重失業率の降順）",
                   FIG / f"pref_profile_radar_{ind}.png", len(prof))
        print(f"✓ 図: figures/pref_profile_radar_{ind}.png（{len(prof)}県）")

        # ---- (2) 参考: クラス所属割合レーダー（5クラス軸, 値=所属割合0-1）----
        ang5 = np.linspace(0, 2 * np.pi, K, endpoint=False)
        ang5c = np.append(ang5, ang5[0])

        def draw_member(ax, row):
            vals = row[PCOLS].to_numpy(dtype=float)
            vals_c = np.append(vals, vals[0])
            col = CLASS_COLORS[int(row["支配クラス"]) - 1]
            ax.plot(ang5c, vals_c, color=col, linewidth=1.7)
            ax.fill(ang5c, vals_c, color=col, alpha=0.30)
            ax.set_ylim(0, 1)
            ax.set_xticks(ang5)
            ax.set_xticklabels([f"C{c+1}" for c in range(K)], fontsize=6.5, color="#52514e")
            ax.set_yticklabels([])
            ax.grid(color="#e1e0d9", linewidth=0.5)
            ax.set_title(f"{row['prefecture']}\n加重失業{row['加重失業率']:.1f}% / 主C{int(row['支配クラス'])} {row['支配割合']:.2f}",
                         fontsize=7.5, pad=4)

        grid_radar(prof, draw_member,
                   f"{ind}：都道府県別 クラス所属確率レーダー（個別リスク評価）"
                   f"（5軸＝クラス1〜5, 値＝所属確率0-1／加重失業率の降順・色＝支配クラス）",
                   FIG / f"pref_membership_radar_{ind}.png", len(prof))
        print(f"✓ 図: figures/pref_membership_radar_{ind}.png（{len(prof)}県）")


if __name__ == "__main__":
    main()
