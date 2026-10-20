"""
深掘り診断：R² = 1.0 が続く原因の追跡
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

data_dir = r"C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\covert_use_fileted_update_1"

df_kinzoku = pd.read_csv(
    os.path.join(data_dir, "勤続年数・超過実労働時間数・所定内給与額_時系列_共通年_統一.csv"),
    encoding='utf-8'
)
df_shitsugyosha = pd.read_csv(
    os.path.join(data_dir, "失業者 - シート1_共通年_統一.csv"),
    encoding='utf-8'
)

target_industry = 'サービス業他'
df_shitsugyosha_filtered = df_shitsugyosha[df_shitsugyosha['共通産業列'] == target_industry].copy()
df_service_kinzoku = df_kinzoku[df_kinzoku['共通産業列'] == target_industry]

def find_best_bunrui(df, year_range):
    df_year = df[(df['年_共通'] >= year_range[0]) & (df['年_共通'] <= year_range[1])]
    best_bunrui = None
    max_count = 0
    for bunrui in df_year['区　分'].unique():
        if '男' in bunrui and '学' in bunrui:
            count = len(df_year[df_year['区　分'] == bunrui])
            if count > max_count:
                max_count = count
                best_bunrui = bunrui
    return best_bunrui

selected_bunrui_list = []
bunrui_2014_2019 = find_best_bunrui(df_service_kinzoku, (2014, 2019))
if bunrui_2014_2019:
    selected_bunrui_list.append(bunrui_2014_2019)
bunrui_2020_2024 = find_best_bunrui(df_service_kinzoku, (2020, 2024))
if bunrui_2020_2024:
    selected_bunrui_list.append(bunrui_2020_2024)
bunrui_2025 = find_best_bunrui(df_service_kinzoku, (2025, 2025))
if bunrui_2025:
    selected_bunrui_list.append(bunrui_2025)

df_kinzoku_filtered = pd.concat([
    df_service_kinzoku[df_service_kinzoku['区　分'] == bunrui]
    for bunrui in selected_bunrui_list if bunrui is not None
], ignore_index=True).copy()

merge_keys = ['年_共通', '共通産業列']
df_merged = pd.merge(
    df_shitsugyosha_filtered,
    df_kinzoku_filtered,
    on=merge_keys,
    how='inner'
)

explvar_continuous = [col for col in df_merged.columns
                      if any(x in col for x in ['勤続年数', '超過実労働時間数', '所定内給与額'])]

for col in explvar_continuous:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

df_analysis = df_merged.dropna(subset=explvar_continuous + ['男女計（失業者数）'])

print("=" * 80)
print("深掘り診断：データ構造の詳細分析")
print("=" * 80)

# ========================================
# 1. データの基本統計量
# ========================================
print("\n【1. データの基本統計量】")
print(f"\nサンプル数: {len(df_analysis)}")
print(f"サンプルの一意の年度数: {df_analysis['年_共通'].nunique()}")
print(f"年度別サンプル数:")
for year in sorted(df_analysis['年_共通'].unique()):
    print(f"  {year}: {len(df_analysis[df_analysis['年_共通'] == year])}")

# ========================================
# 2. 被説明変数（失業者数）の統計
# ========================================
print(f"\n【2. 被説明変数（男女計（失業者数））の統計】")
y = df_analysis['男女計（失業者数）']
print(f"平均: {y.mean():.4f}")
print(f"標準偏差: {y.std():.4f}")
print(f"最小値: {y.min():.4f}")
print(f"最大値: {y.max():.4f}")
print(f"中央値: {y.median():.4f}")

# ========================================
# 3. 説明変数の統計
# ========================================
print(f"\n【3. 説明変数の基本統計量】")
X_continuous = df_analysis[explvar_continuous]
for col in explvar_continuous:
    X_continuous[col] = pd.to_numeric(X_continuous[col], errors='coerce')

print(f"\n説明変数: {len(explvar_continuous)}個")
print(f"各説明変数の統計:")
for col in explvar_continuous[:3]:
    print(f"\n  {col}:")
    print(f"    平均: {X_continuous[col].mean():.4f}")
    print(f"    標準偏差: {X_continuous[col].std():.4f}")
    print(f"    範囲: [{X_continuous[col].min():.4f}, {X_continuous[col].max():.4f}]")

# ========================================
# 4. データの変動の大きさ
# ========================================
print(f"\n【4. 変動の分析】")
y_by_year = df_analysis.groupby('年_共通')['男女計（失業者数）'].agg(['mean', 'std', 'min', 'max'])
print(f"\n年度別失業者数の統計:")
print(y_by_year)

# ========================================
# 5. 被説明変数が説明変数で決定されているか確認
# ========================================
print(f"\n【5. 説明変数と被説明変数の関係】")

# 相関係数を計算
correlations = []
for col in explvar_continuous:
    X_continuous[col] = pd.to_numeric(X_continuous[col], errors='coerce')
    corr = X_continuous[col].corr(y)
    correlations.append((col, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"\n被説明変数との相関が強い説明変数TOP 5:")
for var, corr in correlations[:5]:
    print(f"  {var}: {corr:.6f}")

# ========================================
# 6. OLS推定による詳細診断
# ========================================
print(f"\n【6. OLS推定の詳細診断】")

X = X_continuous.copy()
X = sm.add_constant(X)

for col in X.columns:
    if col != 'const':
        X[col] = pd.to_numeric(X[col], errors='coerce')

y = pd.to_numeric(y, errors='coerce')

mask = X.notna().all(axis=1) & y.notna()
X_clean = X[mask].astype(float)
y_clean = y[mask].astype(float)

print(f"回帰に使用されたサンプル数: {len(y_clean)}")
print(f"説明変数数（定数項含む）: {X_clean.shape[1]}")

model = sm.OLS(y_clean, X_clean)
results = model.fit()

print(f"\nR²: {results.rsquared:.10f}")

# 残差を詳細分析
predictions = results.predict(X_clean)
residuals = y_clean.values - predictions.values

print(f"\n残差の統計:")
print(f"  最大絶対値: {np.max(np.abs(residuals)):.10e}")
print(f"  平均: {np.mean(residuals):.10e}")
print(f"  標準偏差: {np.std(residuals):.10e}")
print(f"  Sum of Squared Residuals (SSR): {np.sum(residuals**2):.10e}")

# ========================================
# 7. RSS（残差平方和）と TSS（全体平方和）
# ========================================
print(f"\n【7. 分散分解の詳細】")
y_mean = y_clean.mean()
SS_total = np.sum((y_clean - y_mean)**2)
SS_residual = np.sum(residuals**2)
SS_regression = SS_total - SS_residual

print(f"TSS (全体平方和): {SS_total:.10e}")
print(f"RSS (残差平方和): {SS_residual:.10e}")
print(f"ESS (説明平方和): {SS_regression:.10e}")
print(f"比率 RSS/TSS: {SS_residual / SS_total:.10e}")
print(f"計算されたR²: 1 - RSS/TSS = {1 - SS_residual / SS_total:.10f}")

# ========================================
# 8. サンプルの多次元性の確認
# ========================================
print(f"\n【8. データの特異性の確認】")
print(f"\n各年度の企業規模別サンプル分布:")
df_analysis_detailed = df_analysis.copy()
for col in explvar_continuous:
    df_analysis_detailed[col] = pd.to_numeric(df_analysis_detailed[col], errors='coerce')

# 同じ年度で異なる企業規模別カテゴリがあるか確認
print(f"\nサンプルの詳細:")
print(df_analysis_detailed[['年_共通', '男女計（失業者数）'] + explvar_continuous[:3]].head(15).to_string())

print("\n" + "=" * 80)
