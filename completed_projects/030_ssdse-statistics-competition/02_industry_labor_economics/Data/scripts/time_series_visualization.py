import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import glob
import re
from datetime import datetime

# Set encoding and font
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

# File paths
data_dir = r'C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\学歴、年齢階級別きまって支給する現金給与額、所定内給与額及び年間賞与その他特別給与額'
output_dir = r'C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\Output'

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Get all Excel files and sort them by year
excel_files = sorted(glob.glob(f'{data_dir}/*.xlsx')) + sorted(glob.glob(f'{data_dir}/*.xls'))

# Extract year from filename
def extract_year(filename):
    name = Path(filename).stem
    # Convert 平成/令和 to Western calendar year
    if '令和' in name:
        reiwa_year = int(re.search(r'令和(\d+)', name).group(1))
        return 2018 + reiwa_year
    elif '平成' in name:
        heisei_year = int(re.search(r'平成(\d+)', name).group(1))
        return 1988 + heisei_year
    return None

# Sort files by year
files_with_years = [(f, extract_year(f)) for f in excel_files if extract_year(f)]
files_with_years.sort(key=lambda x: x[1])

print("Processing Excel files...")
print("="*60)

# Dictionary to store time series data
time_series_data = {
    '勤続年数': [],
    '超過実労働時間数': [],
    '所定内給与額': []
}
years = []

# Process each file
for file_path, year in files_with_years:
    print(f"Processing: {Path(file_path).name} ({year})")

    try:
        # Read the first sheet (which typically contains the summary data)
        xls = pd.ExcelFile(file_path)

        # Try to find the right sheet - usually the first one contains main data
        for sheet_name in xls.sheet_names[:3]:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine='openpyxl')

            # Look for the target row containing "男女計" and "学歴計" or similar
            target_found = False

            for idx, row in df.iterrows():
                row_str = ' '.join(str(val) for val in row if pd.notna(val))

                # Look for rows that indicate "Male-Female Total" and "Total Education"
                if ('男女計' in row_str or '計' in row_str) and idx > 5:
                    # Try to extract numeric values from the row
                    numeric_vals = []
                    for val in row:
                        if pd.notna(val):
                            try:
                                numeric_vals.append(float(val))
                            except:
                                pass

                    # If we found numeric values, this might be our target row
                    if len(numeric_vals) >= 3:
                        print(f"  Found data row at index {idx}")
                        target_found = True
                        break

            if target_found:
                break

        years.append(year)
        print(f"  [OK] Year {year} processed")

    except Exception as e:
        print(f"  [ERROR] {e}")
        continue

print("\n" + "="*60)
print("Note: Excel files are complex. Manual data extraction recommended.")
print("="*60)
print("\nFiles found:")
for i, (f, y) in enumerate(files_with_years, 1):
    print(f"{i}. {Path(f).name} ({y})")

