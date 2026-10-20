# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import re

# Set encoding and font
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

# File paths
csv_path = r'C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\USE\勤続年数・超過実労働時間数・所定内給与額_時系列.csv'
output_dir = r'C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\Output'

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Read the converted data
print("Reading CSV file...")
df = pd.read_csv(csv_path, encoding='utf-8-sig')

print(f"Data shape: {df.shape}")

# Convert numeric columns to float
numeric_cols = [col for col in df.columns if any(x in col for x in ['年数', '時間', '千円', '労働者数'])]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Numeric columns converted to float")

# Convert year format to Western calendar
def convert_year(year_str):
    try:
        year_str = str(year_str).strip()
        # Handle 令和 (Reiwa era)
        if '令' in year_str:
            # 令和X年 -> 2018 + X
            match = re.search(r'令和(\d+)年', year_str)
            if match:
                reiwa = int(match.group(1))
                if reiwa == 0 or '元' in year_str:  # 令和元年 = 2019
                    return 2019
                else:
                    return 2018 + reiwa
        # Handle 平成 (Heisei era)
        elif '平' in year_str:
            # 平成X年 -> 1988 + X
            match = re.search(r'平成(\d+)年', year_str)
            if match:
                heisei = int(match.group(1))
                return 1988 + heisei
    except Exception as e:
        pass
    return None

df['年度_西暦'] = df['年度'].apply(convert_year)

print("\nUnique years:", sorted(df['年度_西暦'].unique()))

# Filter for "男女計学歴計" - the first category row (male-female total, total education)
# These rows have specific markers in the 区分 column
target_condition = (df['区　分'].str.contains('男', na=False)) & \
                   (df['区　分'].str.contains('計', na=False)) & \
                   (~df['区　分'].str.contains('～', na=False))  # Exclude age ranges

target_df = df[target_condition].copy()
print(f"\nFound {len(target_df)} target rows")

# Group by industry and year to get summary data
summary_by_industry_year = target_df.groupby(['産業', '年度_西暦']).agg({
    '10人_所定内給与額_千円': 'first',
    '1,000人以上_所定内給与額_千円': 'first',
    '100～999人_所定内給与額_千円': 'first',
    '10～99人_所定内給与額_千円': 'first',
    '10人_勤続年数_年': 'first',
    '1,000人以上_勤続年数_年': 'first',
    '100～999人_勤続年数_年': 'first',
    '10～99人勤続年数_年': 'first',
    '10人_超過実労働時間数_時間': 'first',
    '1,000人以上_超過実労働時間数_時間': 'first',
    '100～999人_超過実労働時間数_時間': 'first',
    '10～99人_超過実労働時間数_時間': 'first',
}).reset_index()

print("\nSummary data by industry and year:")
print(summary_by_industry_year.head(20))

# Get industries (exclude aggregate categories)
industries = sorted(summary_by_industry_year['産業'].unique())
industries = [ind for ind in industries if ind not in ['産業計', '(民＋公)産業計']]
print(f"\nNumber of industries: {len(industries)}")
print("Industries:", industries)

company_sizes = [
    ('10人以下', '10人_所定内給与額_千円', '10人_勤続年数_年', '10人_超過実労働時間数_時間'),
    ('1,000人以上', '1,000人以上_所定内給与額_千円', '1,000人以上_勤続年数_年', '1,000人以上_超過実労働時間数_時間'),
    ('100～999人', '100～999人_所定内給与額_千円', '100～999人_勤続年数_年', '100～999人_超過実労働時間数_時間'),
    ('10～99人', '10～99人_所定内給与額_千円', '10～99人勤続年数_年', '10～99人_超過実労働時間数_時間'),
]

colors_size = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
years_sorted = sorted(summary_by_industry_year['年度_西暦'].unique())

print("\n" + "="*60)
print("Creating time series graphs by industry...")
print("="*60)

# Create graphs for each industry
for industry in industries:
    industry_data = summary_by_industry_year[summary_by_industry_year['産業'] == industry]

    # Skip if no data for this industry
    if len(industry_data) == 0:
        continue

    # Sanitize filename (remove problematic characters)
    safe_industry_name = industry.replace('/', '_').replace('\\', '_').replace('：', '_')

    # Initialize time series arrays for this industry
    time_series_salary = []
    time_series_tenure = []
    time_series_overtime = []

    for year in years_sorted:
        year_industry_data = industry_data[industry_data['年度_西暦'] == year]

        salary_by_size = []
        tenure_by_size = []
        overtime_by_size = []

        for size_name, sal_col, ten_col, ot_col in company_sizes:
            sal_val = year_industry_data[sal_col].iloc[0] if len(year_industry_data) > 0 and pd.notna(year_industry_data[sal_col].iloc[0]) else 0
            salary_by_size.append(float(sal_val) if sal_val != 0 else 0)

            ten_val = year_industry_data[ten_col].iloc[0] if len(year_industry_data) > 0 and pd.notna(year_industry_data[ten_col].iloc[0]) else 0
            tenure_by_size.append(float(ten_val) if ten_val != 0 else 0)

            ot_val = year_industry_data[ot_col].iloc[0] if len(year_industry_data) > 0 and pd.notna(year_industry_data[ot_col].iloc[0]) else 0
            overtime_by_size.append(float(ot_val) if ot_val != 0 else 0)

        time_series_salary.append(salary_by_size)
        time_series_tenure.append(tenure_by_size)
        time_series_overtime.append(overtime_by_size)

    time_series_salary = np.array(time_series_salary)
    time_series_tenure = np.array(time_series_tenure)
    time_series_overtime = np.array(time_series_overtime)

    # Create salary graph
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    for i, (size_name, sal_col, ten_col, ot_col) in enumerate(company_sizes):
        ax1.plot(years_sorted, time_series_salary[:, i], marker='o', linewidth=2.5,
                 label=size_name, color=colors_size[i], markersize=6)

    ax1.set_xlabel('年度 (西暦)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('所定内給与額 (千円)', fontsize=12, fontweight='bold')
    ax1.set_title(f'【{industry}】企業規模別 所定内給与額の時系列推移 (男女計・学歴計)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(years_sorted[::2] if len(years_sorted) > 4 else years_sorted)
    ax1.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{safe_industry_name}_01_所定内給与額_時系列.png'), dpi=300, bbox_inches='tight')
    print(f"[OK] {safe_industry_name}_01_所定内給与額_時系列.png")
    plt.close()

    # Create tenure graph
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    for i, (size_name, sal_col, ten_col, ot_col) in enumerate(company_sizes):
        ax2.plot(years_sorted, time_series_tenure[:, i], marker='s', linewidth=2.5,
                 label=size_name, color=colors_size[i], markersize=6)

    ax2.set_xlabel('年度 (西暦)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('勤続年数 (年)', fontsize=12, fontweight='bold')
    ax2.set_title(f'【{industry}】企業規模別 勤続年数の時系列推移 (男女計・学歴計)',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='best', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(years_sorted[::2] if len(years_sorted) > 4 else years_sorted)
    ax2.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{safe_industry_name}_02_勤続年数_時系列.png'), dpi=300, bbox_inches='tight')
    print(f"[OK] {safe_industry_name}_02_勤続年数_時系列.png")
    plt.close()

    # Create overtime graph
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    for i, (size_name, sal_col, ten_col, ot_col) in enumerate(company_sizes):
        ax3.plot(years_sorted, time_series_overtime[:, i], marker='^', linewidth=2.5,
                 label=size_name, color=colors_size[i], markersize=6)

    ax3.set_xlabel('年度 (西暦)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('超過実労働時間数 (時間)', fontsize=12, fontweight='bold')
    ax3.set_title(f'【{industry}】企業規模別 超過実労働時間数の時系列推移 (男女計・学歴計)',
                  fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='best', fontsize=11, framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xticks(years_sorted[::2] if len(years_sorted) > 4 else years_sorted)
    ax3.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{safe_industry_name}_03_超過実労働時間数_時系列.png'), dpi=300, bbox_inches='tight')
    print(f"[OK] {safe_industry_name}_03_超過実労働時間数_時系列.png")
    plt.close()

print("\n" + "="*60)
print("Analysis Complete")
print("="*60)
print(f"Output folder: {output_dir}")
print("="*60)
