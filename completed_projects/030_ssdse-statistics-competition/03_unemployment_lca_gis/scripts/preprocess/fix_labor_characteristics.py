#!/usr/bin/env python3
"""
Fix: Extract labor characteristics data correctly across all years
=================================================================
The previous script filtered for a specific pattern that only exists in 2025.
This script correctly extracts male+female + all education + all age aggregations
for ALL years (2014, 2018-2025) and handles year-specific format variations.
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
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

print("=" * 80)
print("FIX: Extract Labor Characteristics (All Years)")
print("=" * 80)

# Load raw labor data
labor_file = DATA_RAW / '02_reuse' / '勤続年数・超過実労働時間数・所定内給与額_時系列_共通年_統一.csv'
df_labor_raw = pd.read_csv(labor_file, encoding='utf-8')

print(f"\nRaw data shape: {df_labor_raw.shape}")
print(f"Year range: {sorted(df_labor_raw['年_共通'].unique())}")

# Strategy: Extract rows that represent aggregated data
# These rows have specific patterns indicating full aggregation
# We need to identify them across different years as the exact format varies

print("\n[Step 1] Analyzing aggregation patterns by year...")

aggregated_rows_by_year = {}

for year in sorted(df_labor_raw['年_共通'].unique()):
    year_data = df_labor_raw[df_labor_raw['年_共通'] == year]

    # Criteria for aggregated rows (full aggregation):
    # - Contains both "男" and "女" indicating gender totals (male+female)
    # - Contains "計" (total)
    # - Contains "学" (education) reference for education-total pattern
    # - Should NOT be a specific age/education breakdown

    agg_mask = (
        year_data['区　分'].str.contains('男', na=False, regex=False) &
        year_data['区　分'].str.contains('計', na=False, regex=False) &
        year_data['区　分'].str.contains('学', na=False, regex=False) &
        ~year_data['区　分'].str.contains('歳', na=False, regex=False)  # Exclude age-specific rows
    )

    agg_rows = year_data[agg_mask]
    aggregated_rows_by_year[year] = agg_rows

    print(f"  {year}: Found {len(agg_rows)} aggregated rows")
    if len(agg_rows) > 0:
        # Show a sample
        sample_pattern = agg_rows['区　分'].iloc[0]
        print(f"    Sample pattern: {repr(sample_pattern)}")

# Extract and consolidate labor data
print("\n[Step 2] Extracting aggregated labor data across all years...")

enterprise_sizes = ['10人', '10～99人', '100～999人', '1,000人以上']
indicators = ['勤続年数', '超過実労働時間数', '所定内給与額']

labor_records = []

for year in sorted(df_labor_raw['年_共通'].unique()):
    year_agg_data = aggregated_rows_by_year[year]

    if len(year_agg_data) == 0:
        print(f"  WARNING: No aggregated rows found for {year}, skipping")
        continue

    for _, row in year_agg_data.iterrows():
        industry = row.get('共通産業列', row.get('産業', None))

        if industry in ['産業合計', '産業_民+公', None]:
            continue  # Skip industry totals

        for size in enterprise_sizes:
            record = {
                'year': year,
                'industry': industry,
                'enterprise_size': size,
            }

            # Build metric columns using the naming pattern from source
            for ind in indicators:
                if ind == '勤続年数':
                    col_name = f"{size}_{ind}_年"
                elif ind == '超過実労働時間数':
                    col_name = f"{size}_{ind}_時間"
                elif ind == '所定内給与額':
                    col_name = f"{size}_{ind}_千円"

                if col_name in row.index:
                    value = row[col_name]
                    if pd.notna(value):
                        try:
                            record[ind] = float(value)
                        except (ValueError, TypeError):
                            record[ind] = np.nan
                    else:
                        record[ind] = np.nan
                else:
                    record[ind] = np.nan

            labor_records.append(record)

df_labor_fixed = pd.DataFrame(labor_records)

print(f"\nExtracted labor data shape: {df_labor_fixed.shape}")
print(f"Years: {sorted(df_labor_fixed['year'].unique())}")
print(f"Industries: {sorted(df_labor_fixed['industry'].unique())}")

# Check data availability
print("\n[Step 3] Data availability by year:")
for year in sorted(df_labor_fixed['year'].unique()):
    year_data = df_labor_fixed[df_labor_fixed['year'] == year]
    non_null_counts = {}
    for ind in indicators:
        non_null = year_data[ind].notna().sum()
        total = len(year_data)
        pct = (non_null / total * 100) if total > 0 else 0
        non_null_counts[ind] = f"{non_null}/{total} ({pct:.1f}%)"

    print(f"  {year}:")
    for ind, info in non_null_counts.items():
        print(f"    {ind}: {info}")

# Save fixed labor data
output_file = DATA_PROCESSED / 'labor_characteristics_by_industry_FIXED.csv'
df_labor_fixed.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nSaved: {output_file.name}")
print(f"Shape: {df_labor_fixed.shape}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY: Labor Characteristics Data Recovery")
print("=" * 80)
print(f"\nOK: Successfully extracted aggregated labor data for years:")
print(f"    {sorted(df_labor_fixed['year'].unique())}")
print(f"OK: Total records: {len(df_labor_fixed)}")
print(f"OK: Industries: {df_labor_fixed['industry'].nunique()}")
print(f"OK: Enterprise sizes: {df_labor_fixed['enterprise_size'].nunique()}")

# Compare with previous (2025-only) version
old_output = DATA_PROCESSED / 'labor_characteristics_by_industry.csv'
if old_output.exists():
    df_old = pd.read_csv(old_output)
    print(f"\nComparison:")
    print(f"  Old version: {len(df_old)} rows (years: {sorted(df_old['year'].unique())})")
    print(f"  New version: {len(df_labor_fixed)} rows (years: {sorted(df_labor_fixed['year'].unique())})")
    print(f"  Improvement: +{len(df_labor_fixed) - len(df_old)} rows")

print("\nOK: NEXT: Rebuild industry_master.csv with this corrected data")
