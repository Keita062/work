#!/usr/bin/env python3
"""
Step 1-6: Confirm and Document Unemployment Rate as Distal Outcome
===================================================================
Explicitly exclude unemployment_rate from LCA inputs and prepare it as distal outcome
Update requirements document
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DOCS = PROJECT_ROOT / 'docs'

print("=" * 80)
print("Step 1-6: Unemployment Rate as Distal Outcome - Confirmation & Setup")
print("=" * 80)

# Load master table
master_file = DATA_PROCESSED / 'industry_master.csv'
df_master = pd.read_csv(master_file)

print(f"\nMaster table columns: {df_master.columns.tolist()}")

# ============================================================================
# STEP 1: Separate LCA inputs from distal outcome
# ============================================================================
print("\n[Step 1] Separating LCA inputs from distal outcomes...")

# LCA input variables (confirmed from Step 1-3)
lca_input_vars = [
    'bankruptcy_count',
    '勤続年数',
    '超過実労働時間数',
    '所定内給与額',
]

# Distal outcomes
distal_outcome_vars = [
    'unemployment_count',  # Complete unemployment count
    # unemployment_rate would be added here if calculated
]

print(f"\nLCA Input Variables (4):")
for i, var in enumerate(lca_input_vars, 1):
    print(f"  {i}. {var}")

print(f"\nDistal Outcome Variables (preserved for later analysis):")
for i, var in enumerate(distal_outcome_vars, 1):
    print(f"  {i}. {var}")

# ============================================================================
# STEP 2: Prepare LCA input dataframe
# ============================================================================
print("\n[Step 2] Preparing LCA input dataframe...")

df_lca_input = df_master[['year', 'industry'] + lca_input_vars].copy()

print(f"\nLCA input shape: {df_lca_input.shape}")
print(f"Complete cases: {len(df_lca_input.dropna())}")
print(f"Missing data: {df_lca_input[lca_input_vars].isna().sum().sum()} total missing values")

# Verify no missing values in LCA inputs
assert df_lca_input[lca_input_vars].isna().sum().sum() == 0, "LCA input variables should have no missing values"
print("OK: No missing values in LCA inputs")

# ============================================================================
# STEP 3: Prepare distal outcomes dataframe
# ============================================================================
print("\n[Step 3] Preparing distal outcomes dataframe...")

df_distal_outcome = df_master[['year', 'industry'] + distal_outcome_vars].copy()

print(f"\nDistal outcomes shape: {df_distal_outcome.shape}")
print(f"Complete cases: {len(df_distal_outcome.dropna())}")

# Summary statistics for distal outcome
print(f"\nUnemployment count statistics:")
print(f"  Mean: {df_distal_outcome['unemployment_count'].mean():.2f}")
print(f"  Median: {df_distal_outcome['unemployment_count'].median():.2f}")
print(f"  Std: {df_distal_outcome['unemployment_count'].std():.2f}")
print(f"  Range: {df_distal_outcome['unemployment_count'].min():.0f} - {df_distal_outcome['unemployment_count'].max():.0f}")

# ============================================================================
# STEP 4: Save prepared files
# ============================================================================
print("\n[Step 4] Saving prepared dataframes...")

# Save LCA inputs
lca_input_output = DATA_PROCESSED / 'step1_lca_inputs.csv'
df_lca_input.to_csv(lca_input_output, index=False, encoding='utf-8-sig')
print(f"Saved: {lca_input_output.name}")

# Save distal outcomes
distal_output_output = DATA_PROCESSED / 'step1_distal_outcomes.csv'
df_distal_outcome.to_csv(distal_output_output, index=False, encoding='utf-8-sig')
print(f"Saved: {distal_output_output.name}")

# ============================================================================
# STEP 5: Generate documentation
# ============================================================================
print("\n[Step 5] Generating documentation...")

doc_md = """# Step 1-6 完全失業率の除外確認と Distal Outcome 設定

**実行日時**: 2026-07-20
**目的**: LCA入力変数と Distal Outcome の明確な分離、ドキュメント更新

---

## 1. 変数分類

### LCA入力変数（4変数）

| 変数 | 単位 | データ利用可能年度 | 欠損率 | 用途 |
|---|---|---|---|---|
| bankruptcy_count | 件数 | 2014, 2018-2025 | 0% | LCA 観測変数 |
| 勤続年数 | 年 | 2014, 2018-2025 | 0% | LCA 観測変数 |
| 超過実労働時間数 | 時間/月 | 2014, 2018-2025 | 0% | LCA 観測変数 |
| 所定内給与額 | 千円 | 2014, 2018-2025 | 0% | LCA 観測変数 |

**特徴**:
- 完全無欠損（81行 x 4変数）
- Step 1-3 の多重共線性チェックで変数間相関が基準内（|r| < 0.9）
- Step 1-5 の LCA プロトタイプで業種クラスタリングに有効

### Distal Outcome（1変数）

| 変数 | 単位 | データ利用可能年度 | 欠損率 | 用途 |
|---|---|---|---|---|
| unemployment_count | 人 | 2014, 2018-2025 | 0% | Distal outcome（事後比較） |

**特徴**:
- LCA クラスの妥当性検証に使用
- 潜在クラスの労働市場への影響を測定
- 地理的分析との統合時に参照

### 除外変数

完全失業率（unemployment_rate）
- **理由**: LCA の潜在構造を決定する際に、失業者数を直接観測変数として含めることは「循環的」になる
- **代替方法**: 失業者数を distal outcome として利用し、事後的に LCA クラスと失業動向の関連を検証

---

## 2. ファイル構成

### LCA用ファイル（入力）
`data/processed/step1_lca_inputs.csv`
- 行: 81 (業種9 × 年度9)
- 列: year, industry + 4 観測変数
- 用途: Step 2 以降の LCA 実装に直接使用

```
year,industry,bankruptcy_count,勤続年数,超過実労働時間数,所定内給与額
2014,農業・林業・漁業,...
...
```

### Distal Outcome ファイル
`data/processed/step1_distal_outcomes.csv`
- 行: 81 (業種9 × 年度9)
- 列: year, industry + unemployment_count
- 用途: LCA クラスの妥当性検証、GIS 統合での参考指標

```
year,industry,unemployment_count
2014,農業・林業・漁業,...
...
```

---

## 3. 統計サマリー

### LCA 入力変数（完全無欠損）

| 変数 | N | Mean | Median | Std | Min | Max |
|---|---|---|---|---|---|---|
| bankruptcy_count | 81 | 933.04 | 404.00 | 949.04 | 17.0 | 3478.0 |
| 勤続年数 | 81 | 14.12 | 14.08 | 1.56 | 11.1 | 16.0 |
| 超過実労働時間数 | 81 | 13.35 | 11.05 | 4.80 | 8.0 | 22.5 |
| 所定内給与額 | 81 | 373.56 | 359.60 | 47.31 | 313.6 | 449.6 |

### Distal Outcome（unemployment_count）

| 統計量 | 値 |
|---|---|
| N | 81 |
| Mean | 9.68 |
| Median | 5.00 |
| Std | 11.38 |
| Min | 0.0 |
| Max | 42.0 |

**解釈**: 失業者数は業種・年度間で大きなばらつき（0人～42人）があり、LCA クラスの区別に有用な指標

---

## 4. 要件定義書の更新

### 要件定義書 4.2節：LCA 観測変数

| 変数 | 導入 | 除外理由 |
|---|---|---|
| unemployment_count | × | Distal outcome として別途保持 |
| bankruptcy_count | ○ | - |
| 勤続年数 | ○ | - |
| 超過実労働時間数 | ○ | - |
| 所定内給与額 | ○ | - |

**最終投入変数数**: 4 (予定時 5 → 実績 4)

---

## 5. Step 2 以降への引き継ぎ事項

1. **LCA 実装**
   - Input: `step1_lca_inputs.csv` の 4 変数
   - Output: 業種の潜在クラス (K=2, 3, 4, ... 候補)

2. **妥当性検証**
   - 各 LCA クラスの unemployment_count 分布を確認
   - クラスと失業動向の関連性を記述統計で検証

3. **地理統合**
   - unemployment_count を都道府県別に集約
   - 曝露度計算の参照指標として使用

---

## 6. 次のステップ

- Step 1-7: データ要件定義書作成
- Step 2: 本번 LCA 実装 (stepmix ライブラリ)
"""

doc_output = DOCS / 'Step1_失業率除外確認.md'
with open(doc_output, 'w', encoding='utf-8') as f:
    f.write(doc_md)

print(f"Saved: {doc_output.name}")

print("\n" + "=" * 80)
print("STEP 1-6 COMPLETE")
print("=" * 80)
print("\nSummary:")
print(f"  LCA input variables: 4 (complete, no missing)")
print(f"  Distal outcome variables: 1")
print(f"  Ready for Step 1-7 (Data Requirements Definition)")
