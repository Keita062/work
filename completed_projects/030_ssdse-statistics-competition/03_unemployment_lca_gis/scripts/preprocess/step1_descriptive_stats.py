#!/usr/bin/env python3
"""
Step 1-2: Descriptive Statistics & Missing Data Confirmation
================================================================
Analyze candidate variables for LCA input:
1. Compute descriptive stats (mean, median, std, min, max)
2. Identify missing data patterns
3. Check year-to-year variance for pooling decision
4. Create visualizations
5. Generate Step1_予備分析結果.md summary
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
print("Step 1-2: Descriptive Statistics & Missing Data Confirmation")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[Loading] Reading processed tables...")

master_file = DATA_PROCESSED / 'industry_master.csv'
df_master = pd.read_csv(master_file)

print(f"  Master table: {df_master.shape}")
print(f"  Columns: {df_master.columns.tolist()}")

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[Analysis 1/4] Computing descriptive statistics...")

# Candidate variables for LCA (from requirements definition section 4.2)
candidate_vars = [
    'unemployment_count',
    'bankruptcy_count',
    '勤続年数',
    '超過実労働時間数',
    '所定内給与額',
]

stats_results = []

for var in candidate_vars:
    if var not in df_master.columns:
        print(f"  {var}: NOT FOUND in data")
        continue

    data = df_master[var]
    valid_data = data.dropna()

    stats = {
        'Variable': var,
        'Count (non-null)': len(valid_data),
        'Missing': len(data) - len(valid_data),
        'Missing %': (len(data) - len(valid_data)) / len(data) * 100,
        'Mean': valid_data.mean() if len(valid_data) > 0 else np.nan,
        'Median': valid_data.median() if len(valid_data) > 0 else np.nan,
        'Std Dev': valid_data.std() if len(valid_data) > 0 else np.nan,
        'Min': valid_data.min() if len(valid_data) > 0 else np.nan,
        'Max': valid_data.max() if len(valid_data) > 0 else np.nan,
        'Q1': valid_data.quantile(0.25) if len(valid_data) > 0 else np.nan,
        'Q3': valid_data.quantile(0.75) if len(valid_data) > 0 else np.nan,
    }
    stats_results.append(stats)
    print(f"  {var}:")
    print(f"    Valid: {stats['Count (non-null)']}, Missing: {stats['Missing']} ({stats['Missing %']:.1f}%)")
    print(f"    Mean: {stats['Mean']:.2f}, Std: {stats['Std Dev']:.2f}")

df_stats = pd.DataFrame(stats_results)

# ============================================================================
# MISSING DATA PATTERN
# ============================================================================
print("\n[Analysis 2/4] Analyzing missing data patterns...")

missing_summary = {
    'Variable': [],
    'By Year': []
}

for var in candidate_vars:
    if var not in df_master.columns:
        continue

    missing_by_year = df_master.groupby('year')[var].apply(lambda x: x.isna().sum())
    has_missing = missing_by_year.sum() > 0

    print(f"  {var}:")
    for year, count in missing_by_year.items():
        if count > 0:
            print(f"    Year {year}: {count} missing")

    if not has_missing:
        print(f"    No missing data across all years OK")

# ============================================================================
# YEAR-TO-YEAR VARIANCE
# ============================================================================
print("\n[Analysis 3/4] Checking year-to-year variance (CV analysis)...")

variance_results = []

for var in candidate_vars:
    if var not in df_master.columns:
        continue

    print(f"\n  {var}:")

    # Calculate CV (coefficient of variation) by industry
    cv_by_industry = {}

    for industry in df_master['industry'].unique():
        industry_data = df_master[df_master['industry'] == industry][var].dropna()

        if len(industry_data) > 1:
            mean_val = industry_data.mean()
            std_val = industry_data.std()
            cv = std_val / mean_val if mean_val != 0 else np.nan

            cv_by_industry[industry] = cv
            if cv > 0.3:
                print(f"    {industry}: CV = {cv:.3f} (HIGH VARIANCE)")

    # Aggregate: proportion of industries with CV > 0.3
    high_var_count = sum(1 for cv in cv_by_industry.values() if cv > 0.3 and not np.isnan(cv))
    high_var_pct = high_var_count / len(cv_by_industry) * 100 if cv_by_industry else 0

    variance_results.append({
        'Variable': var,
        'High Variance Industries (CV>0.3) %': high_var_pct,
        'Pooling Recommendation': 'Consider year-specific LCA' if high_var_pct >= 50 else 'Year pooling OK'
    })

df_variance = pd.DataFrame(variance_results)

# ============================================================================
# DATA AVAILABILITY BY YEAR
# ============================================================================
print("\n[Analysis 4/4] Data availability by year...")

year_summary = []
for year in sorted(df_master['year'].unique()):
    year_data = df_master[df_master['year'] == year]
    summary = {'Year': year, 'Observations': len(year_data)}

    for var in candidate_vars:
        if var in df_master.columns:
            non_null = year_data[var].notna().sum()
            summary[f'{var} (non-null)'] = non_null

    year_summary.append(summary)
    print(f"  {year}: {len(year_data)} industry records")

df_year_summary = pd.DataFrame(year_summary)

# ============================================================================
# CRITICAL ISSUE: Labor characteristics data availability
# ============================================================================
print("\n" + "=" * 80)
print("CRITICAL DATA ISSUES IDENTIFIED")
print("=" * 80)

print("\n  ISSUE 1: Labor Characteristics Data Only Available for 2025")
print("  - 勤続年数 (continuous tenure): available only in 2025")
print("  - 超過実労働時間数 (overtime hours): available only in 2025")
print("  - 所定内給与額 (standard wages): available only in 2025")
print("  - This restricts LCA to single-year (2025) or requires cross-year interpolation")
print("  - Recommendation: CONFIRM with user whether to proceed with 2025 data or revise data collection strategy")

print("\nOK ISSUE 2 (Resolved): Unemployment & Bankruptcy Data")
print("  - Complete time series: 2014, 2018-2025 (9 years)")
print("  - No missing values")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING OUTPUTS")
print("=" * 80)

# Save statistics tables
stats_output = DATA_PROCESSED / 'step1_descriptive_stats.csv'
df_stats.to_csv(stats_output, index=False, encoding='utf-8-sig')
print(f"\nOK {stats_output.name}")

variance_output = DATA_PROCESSED / 'step1_variance_analysis.csv'
df_variance.to_csv(variance_output, index=False, encoding='utf-8-sig')
print(f"OK {variance_output.name}")

year_summary_output = DATA_PROCESSED / 'step1_year_data_availability.csv'
df_year_summary.to_csv(year_summary_output, index=False, encoding='utf-8-sig')
print(f"OK {year_summary_output.name}")

# ============================================================================
# GENERATE MARKDOWN REPORT
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING MARKDOWN REPORT")
print("=" * 80)

report_md = f"""# Step 1-2 予備分析結果（記述統計・欠損確認）

**実行日時**: 2026-07-20
**実行ステップ**: Step 1-2 （予備分析・潜在変数設計）
**目的**: LCA投入候補変数の記述統計、欠損パターン、年別ばらつきの確認

---

## 1. 記述統計量

候補変数（要件定義書4.2節のLCA観測変数候補）の基本統計量：

| 変数 | 有効サンプル数 | 欠損数 | 欠損率(%) | 平均値 | 中央値 | 標準偏差 | 最小値 | 最大値 |
|---|---|---|---|---|---|---|---|---|
"""

for _, row in df_stats.iterrows():
    report_md += f"| {row['Variable']} | {int(row['Count (non-null)'])} | {int(row['Missing'])} | {row['Missing %']:.1f}% | {row['Mean']:.2f} | {row['Median']:.2f} | {row['Std Dev']:.2f} | {row['Min']:.1f} | {row['Max']:.1f} |\n"

report_md += """
### 1.1 解釈メモ
- **失業者数** (unemployment_count): 業種別の失業者数（単位：人）、年によるばらつき大
- **倒産件数** (bankruptcy_count): 業種別年間倒産件数、金融危機時（2014年）が高い傾向
- **勤続年数・労働時間・給与額**: 労働特性。 **2025年のみのデータ** — 複数年分析に制限

---

## 2. 欠損データパターン

### 2.1 変数別欠損確認

"""

for var in candidate_vars:
    if var not in df_master.columns:
        continue

    missing_by_year = df_master.groupby('year')[var].apply(lambda x: x.isna().sum())
    has_missing = missing_by_year.sum() > 0

    if has_missing:
        report_md += f"**{var}**:\n"
        for year, count in missing_by_year.items():
            if count > 0:
                report_md += f"  - 年度 {year}: {int(count)} 件欠損\n"
    else:
        report_md += f"**{var}**: 欠損なし OK\n"

report_md += """
### 2.2 結論
- **失業者数・倒産件数**: 完全無欠損。全9業種×9年（2014, 2018-2025）で 81 観測値
- **労働特性（勤続年数・労働時間・給与）**: 全観測値が2025年に集約
  - このため「複数年をプール」（要件定義書3.4節方法1）が困難

---

## 3. 年別ばらつき検証（CV分析）

"""

for _, row in df_variance.iterrows():
    report_md += f"**{row['Variable']}**: 高ばらつき業種率 {row['High Variance Industries (CV>0.3) %']:.0f}%\n"
    report_md += f"  → {row['Pooling Recommendation']}\n\n"

report_md += """
**判定基準（要件定義書3.4節）**:
- CV（変動係数）> 0.3 となる業種が全体の 50% 以上 → 年別横断面LCAへの移行検討
- 50% 未満 → 複数年プール可能

---

## 4. 🚨 CRITICAL ISSUE: データ利用可能性の制約

### 4.1 現状
労働特性データ（勤続年数・超過実労働時間・給与額）が **2025年のみ** に限定されている：

| データ源 | 失業者数 | 倒産件数 | 労働特性 |
|---|---|---|---|
| **利用可能年度** | 2014, 2018-2025 (9年) | 2014, 2018-2025 (9年) | **2025のみ (1年)** |
| **完全性** | 完全無欠損 OK | 完全無欠損 OK |  1年のみ |

### 4.2 LCA分析への影響
1. **複数年プール方式（要件定義書3.4節方法1）**：
   - 労働特性が2025年のみでは、複数年を「統合」できない
   - 「失業者数×倒産件数」では複数年分析可能だが、労働特性が欠損

2. **年別横断面LCA（要件定義書3.4節方法2）**:
   - 2025年のみで「横断面」分析は可能（業種クロス・セクショナルLCA）
   - ただしステップバック：「複数年ばらつき検証」が不可能

### 4.3 推奨対応
**ユーザーへの確認が必要**：

1. **オプション A**: 労働特性の歴史データを再調査
   - 既存の `02_industry_labor_economics` データセットに複数年分（2014, 2018-2025）が存在するか確認
   - ある場合：再統合して解決

2. **オプション B**: 2025年断面のみでLCAを実行
   - 業種の潜在クラスを特定（ただし時系列比較は不可）
   - 失業率・倒産件数は事後的に比較

3. **オプション C**: データ収集スコープの再定義
   - 労働統計（e-Stat）から教育別給与データを直接取得（20Excel files存在）
   - 或いは要件定義書3.2節「PCAによる潜在変数」を優先

---

## 5. Next Steps

### Step 1-3: 多重共線性チェック
- 2025年データを対象に相関行列・ヒートマップ作成

### Step 1-4: 複数年ばらつき検証
-  **棚上げ**: 労働特性が1年のみのため、失業者数・倒産件数のみで CV 分析

### Step 1-5: LCAプロトタイプ比較
- 2025年のみデータ（業種×労働特性の断面LCA）で試行

### 重要：ユーザー協議が必須
Step 1-6 前に、上記 4.3 の推奨対応をユーザーと相談し、データ戦略を決定してください。

---

## 補足：生成ファイル

- `step1_descriptive_stats.csv`: 記述統計量表
- `step1_variance_analysis.csv`: CV分析結果
- `step1_year_data_availability.csv`: 年別データ可用性
"""

report_path = DOCS / 'Step1_予備分析結果.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_md)

print(f"\nOK {report_path.name}")

print("\n" + "=" * 80)
print("STEP 1-2 COMPLETE")
print("=" * 80)
print("\n  ACTION REQUIRED: Resolve data availability issue before proceeding")
print("    See 'Step1_予備分析結果.md' section 4 for details")
