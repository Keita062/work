"""
診断スクリプト：R² = 1.0 の原因調査
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from numpy.linalg import matrix_rank

data_dir = r"C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\covert_use_fileted_update_1"
output_dir = r"C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\model_results"

# データ読み込み
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
print("診断レポート：R² = 1.0 の原因調査")
print("=" * 80)

print(f"\n【基本情報】")
print(f"サンプル数: {len(df_analysis)}")
print(f"説明変数（勤続年数等）: {len(explvar_continuous)}")

# ========================================
# 1. ダミー変数の確認
# ========================================
print(f"\n【1. ダミー変数の分析】")
year_dummies = pd.get_dummies(df_analysis['年_共通'], prefix='year', drop_first=False)
dummy_cols = [col for col in year_dummies.columns]
print(f"ダミー変数の数（drop_first=False）: {len(dummy_cols)}")
print(f"含まれる年度: {sorted(df_analysis['年_共通'].unique())}")
print(f"ダミー変数リスト: {dummy_cols}")

# ========================================
# 2. デザイン行列のランク確認
# ========================================
print(f"\n【2. デザイン行列のランク確認】")
X = pd.concat([df_analysis[explvar_continuous], year_dummies], axis=1)

# すべての列を数値型に変換
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.astype(float)
X = sm.add_constant(X)

print(f"デザイン行列の形状: {X.shape}")
print(f"（サンプル数 × 説明変数数（定数項含む））: {X.shape[0]} × {X.shape[1]}")

rank_X = matrix_rank(X.values)
print(f"デザイン行列のランク: {rank_X}")
print(f"期待されるランク（定数項 + 説明変数）: {X.shape[1]}")

if rank_X < X.shape[1]:
    print(f"⚠️ 警告：ランク欠損が検出されました！")
    print(f"   ランク欠損の大きさ: {X.shape[1] - rank_X}")
    print(f"   → 説明変数間に完全な線形従属性がある可能性")

# ========================================
# 3. 説明変数間の相関行列
# ========================================
print(f"\n【3. 説明変数間の相関分析】")
correlation_matrix = X.corr()

# 高い相関を検出
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.99:
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))

if high_corr_pairs:
    print(f"相関係数 > 0.99 のペア:")
    for var1, var2, corr in high_corr_pairs:
        print(f"  {var1} ↔ {var2}: {corr:.6f}")
else:
    print(f"相関係数 > 0.99 のペアは見つかりません")

# ========================================
# 4. OLS推定と診断統計量
# ========================================
print(f"\n【4. OLS推定結果の診断統計量】")
y = df_analysis['男女計（失業者数）'].copy()
y = pd.to_numeric(y, errors='coerce')

mask = X.notna().all(axis=1) & y.notna()
X_clean = X[mask].astype(float)
y_clean = y[mask].astype(float)

model = sm.OLS(y_clean, X_clean)
results = model.fit()

print(f"R² (決定係数): {results.rsquared:.10f}")
print(f"Adjusted R²: {results.rsquared_adj:.10f}")
print(f"Condition Number: {np.linalg.cond(X_clean.values):.2e}")

# 固有値を確認
eigenvalues = np.linalg.eigvals(X_clean.T @ X_clean)
eigenvalues_sorted = np.sort(eigenvalues)[::-1]
print(f"\n最大固有値: {eigenvalues_sorted[0]:.6e}")
print(f"最小固有値: {eigenvalues_sorted[-1]:.6e}")
print(f"固有値の比（条件数の指標）: {eigenvalues_sorted[0] / eigenvalues_sorted[-1]:.6e}")

# ========================================
# 5. 予測値と実績値の比較
# ========================================
print(f"\n【5. 予測値と実績値の比較】")
predictions = results.predict(X_clean)
residuals = y_clean.values - predictions.values

print(f"残差の最大値: {np.max(np.abs(residuals)):.10f}")
print(f"残差の平均: {np.mean(residuals):.10e}")
print(f"残差の標準偏差: {np.std(residuals):.10e}")

# ========================================
# 6. ダミー変数のトラップ診断
# ========================================
print(f"\n【6. ダミー変数トラップの診断】")
print(f"カテゴリ数（年度数）: {len(df_analysis['年_共通'].unique())}")
print(f"ダミー変数数: {len(dummy_cols)}")

if len(dummy_cols) == len(df_analysis['年_共通'].unique()):
    print("⚠️ 警告：ダミー変数トラップが発生しています！")
    print("   すべてのカテゴリに対してダミー変数が作成されています。")
    print("   通常は、(カテゴリ数 - 1) 個のダミー変数を使用します。")
    print("   定数項があるため、ランク欠損が発生します。")

# ========================================
# 7. サンプル数と自由度の確認
# ========================================
print(f"\n【7. 自由度の確認】")
print(f"サンプル数: {len(y_clean)}")
print(f"説明変数数（定数項含む）: {X_clean.shape[1]}")
print(f"自由度（残差）: {len(y_clean) - X_clean.shape[1]}")

if len(y_clean) - X_clean.shape[1] < 0:
    print("⚠️ 警告：サンプル数が説明変数数より少ないです！")
elif len(y_clean) - X_clean.shape[1] == 0:
    print("⚠️ 警告：完全に多元回帰（サンプル数 = 説明変数数）です！")

print("\n" + "=" * 80)
