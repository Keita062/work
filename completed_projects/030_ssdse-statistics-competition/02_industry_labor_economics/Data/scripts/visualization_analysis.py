import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set encoding and font
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

# File paths
csv_path = r'C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\USE\勤続年数・超過実労働時間数・所定内給与額・労働者数.csv'
output_dir = r'C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\Output'

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Read CSV
df = pd.read_csv(csv_path, encoding='utf-8')

# Extract the row with "男　女　計" and "学　歴　計"
target_row = df[(df['区　分'].str.contains('男.*女.*計', na=False, regex=True)) &
                (df['区　分'].str.contains('学.*歴.*計', na=False, regex=True))].iloc[0]

# Company sizes
company_sizes = ['10人以下', '1,000人以上', '100～999人', '10～99人']

# Extract the three metrics for each company size
metrics = {
    '勤続年数': [],
    '超過実労働時間数': [],
    '所定内給与額': []
}

# Extract data for each company size
for size in company_sizes:
    if size == '10人以下':
        metrics['勤続年数'].append(float(target_row['10人_勤続年数_年']))
        metrics['超過実労働時間数'].append(float(target_row['10人_超過実労働時間数_時間']))
        metrics['所定内給与額'].append(float(target_row['10人_所定内給与額_千円']))
    elif size == '1,000人以上':
        metrics['勤続年数'].append(float(target_row['1,000人以上_勤続年数_年']))
        metrics['超過実労働時間数'].append(float(target_row['1,000人以上_超過実労働時間数_時間']))
        metrics['所定内給与額'].append(float(target_row['1,000人以上_所定内給与額_千円']))
    elif size == '100～999人':
        metrics['勤続年数'].append(float(target_row['100～999人_勤続年数_年']))
        metrics['超過実労働時間数'].append(float(target_row['100～999人_超過実労働時間数_時間']))
        metrics['所定内給与額'].append(float(target_row['100～999人_所定内給与額_千円']))
    elif size == '10～99人':
        metrics['勤続年数'].append(float(target_row['10～99人勤続年数_年']))
        metrics['超過実労働時間数'].append(float(target_row['10～99人_超過実労働時間数_時間']))
        metrics['所定内給与額'].append(float(target_row['10～99人_所定内給与額_千円']))

# Create visualization 1: 勤続年数 (Tenure Years)
fig1, ax1 = plt.subplots(figsize=(12, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars1 = ax1.bar(company_sizes, metrics['勤続年数'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_ylabel('勤続年数 (年)', fontsize=12, fontweight='bold')
ax1.set_xlabel('企業規模', fontsize=12, fontweight='bold')
ax1.set_title('企業規模別 勤続年数（男女計・学歴計）', fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(metrics['勤続年数']) * 1.15)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_勤続年数.png'), dpi=300, bbox_inches='tight')
print("[OK] 01_勤続年数.png")
plt.close()

# Create visualization 2: 超過実労働時間数 (Overtime Hours)
fig2, ax2 = plt.subplots(figsize=(12, 6))
bars2 = ax2.bar(company_sizes, metrics['超過実労働時間数'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_ylabel('超過実労働時間数 (時間)', fontsize=12, fontweight='bold')
ax2.set_xlabel('企業規模', fontsize=12, fontweight='bold')
ax2.set_title('企業規模別 超過実労働時間数（男女計・学歴計）', fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(metrics['超過実労働時間数']) * 1.15)

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_超過実労働時間数.png'), dpi=300, bbox_inches='tight')
print("[OK] 02_超過実労働時間数.png")
plt.close()

# Create visualization 3: 所定内給与額 (Base Salary)
fig3, ax3 = plt.subplots(figsize=(12, 6))
bars3 = ax3.bar(company_sizes, metrics['所定内給与額'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax3.set_ylabel('所定内給与額 (千円)', fontsize=12, fontweight='bold')
ax3.set_xlabel('企業規模', fontsize=12, fontweight='bold')
ax3.set_title('企業規模別 所定内給与額（男女計・学歴計）', fontsize=14, fontweight='bold', pad=20)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_ylim(0, max(metrics['所定内給与額']) * 1.15)

# Add value labels on bars
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_所定内給与額.png'), dpi=300, bbox_inches='tight')
print("[OK] 03_所定内給与額.png")
plt.close()

# Create summary data file
summary_df = pd.DataFrame({
    '企業規模': company_sizes,
    '勤続年数 (年)': metrics['勤続年数'],
    '超過実労働時間数 (時間)': metrics['超過実労働時間数'],
    '所定内給与額 (千円)': metrics['所定内給与額']
})

summary_df.to_csv(os.path.join(output_dir, 'summary_data.csv'), index=False, encoding='utf-8-sig')
print("[OK] summary_data.csv")

print("\nAnalysis Complete")
print("="*60)
print(f"Output folder: {output_dir}")
print("="*60)
