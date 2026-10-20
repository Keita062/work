#!/usr/bin/env python3
"""
Step 1-1: Build Industry Master Table
================================
Process raw data into consolidated table for LCA input:
1. Load unemployment, bankruptcy, and labor characteristics (already aggregated by year)
2. Extract labor characteristics for male+female, all education, all age (already done in source)
3. Consolidate into industry_master.csv with proper joins
4. Prepare unemployment_rate table with substitution flags where census data is used
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DOCS = PROJECT_ROOT / 'docs'

# Create output directories
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
(DATA_PROCESSED / 'figures').mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Step 1-1: Building Industry Master Table")
print("=" * 80)

# Year to nearest census year mapping for unemployment rate denominator
# unemployment data years -> nearest census year (2005, 2010, 2015, 2020)
YEAR_TO_CENSUS = {
    2014: (2015, 1),  # (census_year, substitution_flag)
    2018: (2015, 1),
    2019: (2015, 1),
    2020: (2020, 0),
    2021: (2020, 1),
    2022: (2020, 1),
    2023: (2020, 1),
    2024: (2020, 1),
    2025: (2020, 1),
}

# ============================================================================
# STEP 1: Load unemployment data
# ============================================================================
print("\n[Step 1/5] Loading unemployment data...")
unemployment_file = DATA_RAW / '02_reuse' / '失業者 - シート1_共通年_統一.csv'
df_unemployment = pd.read_csv(unemployment_file, encoding='utf-8')

print(f"  Shape: {df_unemployment.shape}")
print(f"  Columns: {df_unemployment.columns.tolist()}")
print(f"  Available years: {sorted(df_unemployment['年_共通'].unique())}")

# Prepare unemployment data: aggregate genders (already done in source)
# Use male+female total
df_unemployment = df_unemployment[['年_共通', '共通産業列', '男女計（失業者数）']].copy()
df_unemployment.columns = ['year', 'industry', 'unemployment_count']
df_unemployment = df_unemployment[df_unemployment['industry'] != '産業合計']  # Exclude total

print(f"  After cleanup: {df_unemployment.shape}")
print(f"  Industries: {sorted(df_unemployment['industry'].unique())}")

# ============================================================================
# STEP 2: Load labor characteristics data
# ============================================================================
print("\n[Step 2/5] Loading labor characteristics...")
labor_file = DATA_RAW / '02_reuse' / '勤続年数・超過実労働時間数・所定内給与額_時系列_共通年_統一.csv'
df_labor_raw = pd.read_csv(labor_file, encoding='utf-8')

print(f"  Raw shape: {df_labor_raw.shape}")

# Filter for aggregated row: "男　女　計\n学　歴　計" (male+female, all education, all age)
# This value contains newline character
target_division = '男　女　計\n学　歴　計'

df_labor = df_labor_raw[df_labor_raw['区　分'] == target_division].copy()
print(f"  After filtering for aggregated row: {df_labor.shape}")
print(f"  Industries (include total): {sorted(df_labor['共通産業列'].unique())}")

# Remove industry totals
df_labor = df_labor[~df_labor['共通産業列'].isin(['産業合計', '産業_民+公'])].copy()
print(f"  After removing totals: {df_labor.shape}")

# Prepare labor data: enterprise-size-specific indicators
# Extract metrics for each enterprise size (10人未満, 10-99, 100-999, 1000+)
enterprise_sizes = ['10人', '10～99人', '100～999人', '1,000人以上']
indicators = ['勤続年数', '超過実労働時間数', '所定内給与額']

labor_records = []
for _, row in df_labor.iterrows():
    for size in enterprise_sizes:
        record = {
            'year': row['年_共通'],
            'industry': row['共通産業列'],
            'enterprise_size': size,
        }
        # Build metric columns using the naming pattern from source
        for ind in indicators:
            col_name = f"{size}_{ind}_{'年' if ind == '勤続年数' else ('時間' if ind == '超過実労働時間数' else '千円')}"
            if col_name in df_labor.columns:
                value = row[col_name]
                # Convert numeric types appropriately
                if pd.notna(value):
                    record[ind] = float(value) if isinstance(value, (int, float, str)) else np.nan
                else:
                    record[ind] = np.nan
        labor_records.append(record)

df_labor_long = pd.DataFrame(labor_records)
print(f"  Labor data (long format): {df_labor_long.shape}")
print(f"  Available years: {sorted(df_labor_long['year'].unique())}")

# ============================================================================
# STEP 3: Load bankruptcy data
# ============================================================================
print("\n[Step 3/5] Loading bankruptcy data...")
bankruptcy_file = DATA_RAW / '02_reuse' / '産業別年間倒産件数表_共通年_統一.csv'
df_bankruptcy = pd.read_csv(bankruptcy_file, encoding='utf-8')

print(f"  Shape: {df_bankruptcy.shape}")

# Remove industry totals
df_bankruptcy = df_bankruptcy[df_bankruptcy['共通産業列'] != '産業合計'].copy()
df_bankruptcy = df_bankruptcy[['年_共通', '共通産業列', '倒産件数']].copy()
df_bankruptcy.columns = ['year', 'industry', 'bankruptcy_count']

print(f"  After cleanup: {df_bankruptcy.shape}")
print(f"  Available years: {sorted(df_bankruptcy['year'].unique())}")

# ============================================================================
# STEP 4: Prepare unemployment rate table
# ============================================================================
print("\n[Step 4/5] Preparing unemployment rate table...")
# This table will have structure for LCA input
# unemployment_rate = unemployment_count / labor_force * 100
# We'll calculate this with census data in the next section

df_unemp_rate = df_unemployment.copy()
df_unemp_rate['census_year'] = df_unemp_rate['year'].map(lambda x: YEAR_TO_CENSUS.get(x, (2020, 1))[0])
df_unemp_rate['substitution_flag'] = df_unemp_rate['year'].map(lambda x: YEAR_TO_CENSUS.get(x, (2020, 1))[1])

print(f"  Structure prepared: {df_unemp_rate.shape}")
print(f"  Census year mapping:\n{df_unemp_rate[['year', 'census_year', 'substitution_flag']].drop_duplicates().sort_values('year')}")

# Note: Census employment counts should be added from census data
# For now, these columns are placeholders
df_unemp_rate['employment_count'] = np.nan
df_unemp_rate['labor_force'] = np.nan
df_unemp_rate['unemployment_rate'] = np.nan

# ============================================================================
# STEP 5: CONSOLIDATE INTO MASTER TABLE
# ============================================================================
print("\n[Step 5/5] Consolidating into master table...")

# Start with unemployment as base (defines year×industry combinations)
master = df_unemployment[['year', 'industry']].drop_duplicates().copy()
print(f"  Base (year×industry): {master.shape}")

# Add unemployment counts
master = master.merge(df_unemployment, on=['year', 'industry'], how='left')
print(f"  After unemployment: {master.shape}")

# Add bankruptcy (left join to preserve all year-industry pairs)
master = master.merge(df_bankruptcy, on=['year', 'industry'], how='left')
print(f"  After bankruptcy: {master.shape}")

# Add labor characteristics
# Group by year, industry to get average across enterprise sizes
df_labor_agg = df_labor_long.groupby(['year', 'industry']).agg({
    '勤続年数': 'mean',
    '超過実労働時間数': 'mean',
    '所定内給与額': 'mean',
}).reset_index()

master = master.merge(df_labor_agg, on=['year', 'industry'], how='left')
print(f"  After labor characteristics (averaged): {master.shape}")

# Note: Enterprise-size-specific data is available in separate table
# Master table uses aggregated values (averaged across all enterprise sizes)
# See labor_characteristics_by_industry.csv for detailed breakdown

# Reorder columns for clarity
column_order = [
    'year', 'industry',
    'unemployment_count', 'bankruptcy_count',
    '勤続年数', '超過実労働時間数', '所定内給与額',
]
# Add other columns that exist
other_cols = [c for c in master.columns if c not in column_order]
master = master[column_order + other_cols]

print(f"\nFinal master table: {master.shape}")
print(f"Columns: {master.columns.tolist()}")
print(f"\nData summary:")
print(f"  Years: {sorted(master['year'].unique())}")
print(f"  Industries: {sorted(master['industry'].unique())}")
print(f"  Year×Industry combinations: {len(master)}")
print(f"\nFirst 5 rows:")
print(master.head())

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING OUTPUTS")
print("=" * 80)

# Save unemployment rate table
unemp_rate_output = DATA_PROCESSED / 'unemployment_rate_by_industry.csv'
df_unemp_rate_save = df_unemp_rate[['year', 'industry', 'unemployment_count', 'employment_count', 'labor_force', 'unemployment_rate', 'census_year', 'substitution_flag']].copy()
df_unemp_rate_save.to_csv(unemp_rate_output, index=False, encoding='utf-8-sig')
print(f"\n✓ {unemp_rate_output.name}")
print(f"  Shape: {df_unemp_rate_save.shape}")

# Save labor characteristics (long format by enterprise size)
labor_output = DATA_PROCESSED / 'labor_characteristics_by_industry.csv'
df_labor_long.to_csv(labor_output, index=False, encoding='utf-8-sig')
print(f"✓ {labor_output.name}")
print(f"  Shape: {df_labor_long.shape}")

# Save bankruptcy
bankruptcy_output = DATA_PROCESSED / 'bankruptcy_by_industry.csv'
df_bankruptcy.to_csv(bankruptcy_output, index=False, encoding='utf-8-sig')
print(f"✓ {bankruptcy_output.name}")
print(f"  Shape: {df_bankruptcy.shape}")

# Save master table
master_output = DATA_PROCESSED / 'industry_master.csv'
master.to_csv(master_output, index=False, encoding='utf-8-sig')
print(f"✓ {master_output.name}")
print(f"  Shape: {master.shape}")

print("\n" + "=" * 80)
print("STEP 1-1 COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print(f"  - {unemp_rate_output}")
print(f"  - {labor_output}")
print(f"  - {bankruptcy_output}")
print(f"  - {master_output}")
print("\nNext: Step 1-2 (Descriptive statistics & missing data confirmation)")
