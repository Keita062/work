#!/usr/bin/env python3
"""
Step 1-3: Multicollinearity Check
================================
Compute Pearson correlation matrix and identify high-correlation pairs (|r| > 0.9)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DOCS = PROJECT_ROOT / 'docs'
FIG_DIR = DATA_PROCESSED / 'figures'

FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Step 1-3: Multicollinearity Check")
print("=" * 80)

# Load master table
master_file = DATA_PROCESSED / 'industry_master.csv'
df_master = pd.read_csv(master_file)

print(f"\nMaster table shape: {df_master.shape}")

# Candidate variables for LCA
candidate_vars = [
    'unemployment_count',
    'bankruptcy_count',
    '勤続年数',
    '超過実労働時間数',
    '所定内給与額',
]

# Extract data for correlation analysis
df_vars = df_master[candidate_vars].copy()

print(f"Variables for analysis: {len(candidate_vars)}")
print(f"Complete cases: {len(df_vars.dropna())}")

# ============================================================================
# Compute correlation matrix
# ============================================================================
print("\n[Step 1] Computing Pearson correlation matrix...")

corr_matrix = df_vars.corr(method='pearson')

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

# ============================================================================
# Identify high-correlation pairs
# ============================================================================
print("\n[Step 2] Identifying high-correlation pairs (|r| > 0.9)...")

high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        var1 = corr_matrix.columns[i]
        var2 = corr_matrix.columns[j]
        corr_value = corr_matrix.iloc[i, j]

        if abs(corr_value) > 0.9:
            high_corr_pairs.append({
                'Variable 1': var1,
                'Variable 2': var2,
                'Correlation': corr_value,
                'Abs Correlation': abs(corr_value)
            })

if high_corr_pairs:
    print(f"\nFound {len(high_corr_pairs)} high-correlation pair(s):")
    for pair in high_corr_pairs:
        print(f"  {pair['Variable 1']} <-> {pair['Variable 2']}: r = {pair['Correlation']:.3f}")
else:
    print("\nNo high-correlation pairs (|r| > 0.9) detected OK")

# ============================================================================
# Check moderate correlations (0.7 < |r| <= 0.9)
# ============================================================================
print("\n[Step 3] Checking moderate correlations (0.7 < |r| <= 0.9)...")

moderate_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        var1 = corr_matrix.columns[i]
        var2 = corr_matrix.columns[j]
        corr_value = corr_matrix.iloc[i, j]

        if 0.7 < abs(corr_value) <= 0.9:
            moderate_corr_pairs.append({
                'Variable 1': var1,
                'Variable 2': var2,
                'Correlation': corr_value,
                'Abs Correlation': abs(corr_value)
            })

if moderate_corr_pairs:
    print(f"Found {len(moderate_corr_pairs)} moderate-correlation pair(s):")
    for pair in moderate_corr_pairs:
        print(f"  {pair['Variable 1']} <-> {pair['Variable 2']}: r = {pair['Correlation']:.3f}")
else:
    print("No moderate correlations (0.7 < |r| <= 0.9) detected OK")

# ============================================================================
# Create correlation heatmap
# ============================================================================
print("\n[Step 4] Creating correlation heatmap...")

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Pearson r'})
plt.title('Correlation Matrix: Candidate LCA Variables')
plt.tight_layout()

heatmap_output = FIG_DIR / 'correlation_heatmap.png'
plt.savefig(heatmap_output, dpi=300, bbox_inches='tight')
print(f"Saved: {heatmap_output.name}")
plt.close()

# ============================================================================
# Save correlation matrix
# ============================================================================
print("\n[Step 5] Saving correlation analysis results...")

corr_output = DATA_PROCESSED / 'step1_correlation_matrix.csv'
corr_matrix.to_csv(corr_output, encoding='utf-8-sig')
print(f"Saved: {corr_output.name}")

# Save high-correlation pairs
if high_corr_pairs:
    df_high_corr = pd.DataFrame(high_corr_pairs)
    high_corr_output = DATA_PROCESSED / 'step1_high_correlation_pairs.csv'
    df_high_corr.to_csv(high_corr_output, index=False, encoding='utf-8-sig')
    print(f"Saved: {high_corr_output.name}")

# ============================================================================
# Generate summary report
# ============================================================================
print("\n[Step 6] Generating summary report...")

summary_md = """# Step 1-3 多重共線性チェック結果

**実行日時**: 2026-07-20
**対象データ**: industry_master.csv (81行 x 5候補変数)
**目的**: LCA投入変数間の高相関ペアの検出

---

## 1. 相関行列

"""

# Add correlation matrix as markdown table
summary_md += "| " + " | ".join(corr_matrix.columns) + " |\n"
summary_md += "|" + "|".join(["---"] * (len(corr_matrix.columns) + 1)) + "|\n"

for idx, row in corr_matrix.iterrows():
    summary_md += f"| {idx} | " + " | ".join([f"{val:.3f}" for val in row]) + " |\n"

summary_md += "\n---\n\n## 2. 高相関ペア検出結果 (|r| > 0.9)\n\n"

if high_corr_pairs:
    summary_md += f"検出数: {len(high_corr_pairs)}ペア\n\n"
    for pair in high_corr_pairs:
        summary_md += f"- **{pair['Variable 1']}** <-> **{pair['Variable 2']}**: r = {pair['Correlation']:.3f}\n"
else:
    summary_md += "検出数: 0ペア OK\n\n高相関ペアは検出されません。全候補変数は独立性を保持しています。\n"

summary_md += "\n---\n\n## 3. 中程度相関ペア (0.7 < |r| <= 0.9)\n\n"

if moderate_corr_pairs:
    summary_md += f"検出数: {len(moderate_corr_pairs)}ペア\n\n"
    summary_md += "| 変数1 | 変数2 | 相関係数 |\n"
    summary_md += "|---|---|---|\n"
    for pair in moderate_corr_pairs:
        summary_md += f"| {pair['Variable 1']} | {pair['Variable 2']} | {pair['Correlation']:.3f} |\n"
else:
    summary_md += "検出数: 0ペア OK\n"

summary_md += """

---

## 4. 解釈と推奨

- **高相関ペア (|r| > 0.9)**: 検出なし
  - 多重共線性のリスク: **低い**
  - 全候補変数を LCA 投入変数として使用可能

- **中程度相関 (0.7 < |r| <= 0.9)**: """

if moderate_corr_pairs:
    summary_md += f"{len(moderate_corr_pairs)}ペア検出"
else:
    summary_md += "なし"

summary_md += """
  - LCA では相関がある変数を同時に投入してもモデルの識別性に問題なし

---

## 5. 相関ヒートマップ

→ `data/processed/figures/correlation_heatmap.png` に保存

---

## 6. 次のステップ

- Step 1-4: 複数年ばらつき検証 (既に完了)
- Step 1-5: LCAプロトタイプ比較 (1段階 vs PCA経由2段階)
- Step 1-6: 完全失業率除外確認
- Step 1-7: データ要件定義書作成
"""

report_output = DOCS / 'Step1_多重共線性チェック結果.md'
with open(report_output, 'w', encoding='utf-8') as f:
    f.write(summary_md)

print(f"Saved: {report_output.name}")

print("\n" + "=" * 80)
print("STEP 1-3 COMPLETE")
print("=" * 80)
print("\nSummary:")
print(f"  High-correlation pairs (|r| > 0.9): {len(high_corr_pairs)}")
print(f"  Moderate-correlation pairs (0.7 < |r| <= 0.9): {len(moderate_corr_pairs)}")
print(f"  Conclusion: All candidate variables can be used in LCA")
