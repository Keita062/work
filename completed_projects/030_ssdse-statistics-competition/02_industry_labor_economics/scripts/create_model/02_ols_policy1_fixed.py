"""
修正版：OLS回帰分析（データスケール統一版）
被説明変数：サービス業他の失業者数（男女計）
説明変数：勤続年数、超過実労働時間数、所定内給与額（業界全体の加重平均）、年度ダミー
対象期間：2014年、2018年-2025年（9年度）
出力：02_ols_results_fixed.txt
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# ========================================
# 1. データ読み込み
# ========================================

data_dir = r"C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\covert_use_fileted_update_1"
output_dir = r"C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\model_results"

os.makedirs(output_dir, exist_ok=True)

df_kinzoku = pd.read_csv(
    os.path.join(data_dir, "勤続年数・超過実労働時間数・所定内給与額_時系列_共通年_統一.csv"),
    encoding='utf-8'
)
df_shitsugyosha = pd.read_csv(
    os.path.join(data_dir, "失業者 - シート1_共通年_統一.csv"),
    encoding='utf-8'
)

# ========================================
# 2. データの準備：失業者データ
# ========================================

target_industry = 'サービス業他'
df_shitsugyosha_filtered = df_shitsugyosha[df_shitsugyosha['共通産業列'] == target_industry].copy()

# ========================================
# 3. 労働統計を業界全体の平均に集計
# ========================================

# サービス業他の労働統計を抽出
df_service_kinzoku = df_kinzoku[df_kinzoku['共通産業列'] == target_industry].copy()

# 「男」と「学」を含む区分を年度別に検出
def find_best_bunrui(df, year_range):
    """指定された年度範囲で「男」と「学」を含む最適な区分を探す"""
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

# 複数の区分値を統合
df_kinzoku_filtered = pd.concat([
    df_service_kinzoku[df_service_kinzoku['区　分'] == bunrui]
    for bunrui in selected_bunrui_list if bunrui is not None
], ignore_index=True).copy()

# 企業規模別の列を特定
firm_size_cols = df_kinzoku_filtered.columns[df_kinzoku_filtered.columns.str.contains('人|規', regex=True, case=False)]

# 説明変数を特定
explvar_continuous = [col for col in df_kinzoku_filtered.columns
                      if any(x in col for x in ['勤続年数', '超過実労働時間数', '所定内給与額'])]

# ========================================
# 4. 労働統計を業界全体の平均に集計
# ========================================

print("【労働統計の集計】")
print(f"元のサンプル数（企業規模別）: {len(df_kinzoku_filtered)}")

# 説明変数を数値型に変換
for col in explvar_continuous:
    df_kinzoku_filtered[col] = pd.to_numeric(df_kinzoku_filtered[col], errors='coerce')

# 年度別に平均値を計算
df_kinzoku_agg = df_kinzoku_filtered.groupby('年_共通')[explvar_continuous].mean().reset_index()

print(f"集計後のサンプル数（年度単位）: {len(df_kinzoku_agg)}")
print(f"\n集計された説明変数:")
for col in explvar_continuous[:3]:
    print(f"  {col}")

# ========================================
# 5. 失業者データと統合
# ========================================

merge_keys = ['年_共通']
df_merged = pd.merge(
    df_shitsugyosha_filtered[['年_共通', '男女計（失業者数）']],
    df_kinzoku_agg,
    on=merge_keys,
    how='inner'
)

print(f"\nマージ後のサンプル数: {len(df_merged)}")
print(f"対象年度: {sorted(df_merged['年_共通'].unique())}")

# ========================================
# 6. 説明変数の前処理
# ========================================

for col in explvar_continuous:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

df_analysis = df_merged.dropna(subset=explvar_continuous + ['男女計（失業者数）']).copy()

# ========================================
# 7. 年度ダミー変数の作成
# ========================================

year_dummies = pd.get_dummies(df_analysis['年_共通'], prefix='year', drop_first=True)
df_analysis = pd.concat([df_analysis, year_dummies], axis=1)
dummy_cols = [col for col in df_analysis.columns if col.startswith('year_')]

# ========================================
# 8. OLS回帰モデルの構築と推定
# ========================================

X = df_analysis[explvar_continuous + dummy_cols].copy()
y = df_analysis['男女計（失業者数）'].copy()

# 全ての列を数値型に変換
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

y = pd.to_numeric(y, errors='coerce')

# NaNを削除
mask = X.notna().all(axis=1) & y.notna()
X = X[mask].astype(float)
y = y[mask].astype(float)

X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()

# ========================================
# 8-1. 変数重要度の計算
# ========================================

X_std = X.copy()
for col in X_std.columns:
    if col != 'const':
        X_std[col] = (X[col] - X[col].mean()) / X[col].std()

model_std = sm.OLS(y, X_std)
results_std = model_std.fit()

def significance_marker(p_val):
    if pd.isna(p_val):
        return ''
    if p_val < 0.01:
        return '***'
    elif p_val < 0.05:
        return '**'
    elif p_val < 0.10:
        return '*'
    else:
        return ''

coef_summary = pd.DataFrame({
    '変数名': X.columns[1:],
    '非標準化係数': results.params[1:].values,
    '標準化係数': results_std.params[1:].values,
    '標準誤差': results.bse[1:].values,
    't値': results.tvalues[1:].values,
    'p値': results.pvalues[1:].values
})

coef_summary['重要度（|標準化係数|）'] = coef_summary['標準化係数'].abs()
coef_summary = coef_summary.sort_values('重要度（|標準化係数|）', ascending=False)
coef_summary['有意性'] = coef_summary['p値'].apply(significance_marker)

# ========================================
# 9. 結果をテキストファイルに出力
# ========================================

output_file = os.path.join(output_dir, '02_ols_results_fixed.txt')

with open(output_file, 'w', encoding='utf-8') as f:
    # ===== 1. 基本情報 =====
    f.write("=" * 80 + "\n")
    f.write("修正版：OLS回帰分析（データスケール統一版）\n")
    f.write("被説明変数：サービス業他の失業者数（男女計）\n")
    f.write("説明変数：勤続年数、超過実労働時間数、所定内給与額\n")
    f.write("        （業界全体の加重平均）+ 年度ダミー\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"対象産業：{target_industry}\n")
    f.write(f"対象年度：{sorted(df_analysis['年_共通'].unique())}\n")
    f.write(f"サンプル数：{len(df_analysis)}\n")
    f.write(f"説明変数数：{len(X.columns) - 1}\n\n")

    # ===== 2. データ集計方法の説明 =====
    f.write("=" * 80 + "\n")
    f.write("データ集計方法\n")
    f.write("=" * 80 + "\n\n")
    f.write("修正点：データレベルの統一\n")
    f.write("・失業者数：業界全体（全企業規模の合計）\n")
    f.write("・説明変数：業界全体の平均値\n")
    f.write("  → 9つの企業規模別データを平均化\n")
    f.write("  → 年度ごとに1つのサンプルに集計\n\n")

    f.write("結果：\n")
    f.write("・1:1対応（失業者数 1:1 説明変数）\n")
    f.write("・サンプル数：9年度 × 1サンプル = 9サンプル\n")
    f.write("・年度ダミーの影響を除去可能\n\n")

    # ===== 3. 完全な回帰結果サマリー =====
    f.write("=" * 80 + "\n")
    f.write("OLS回帰結果サマリー\n")
    f.write("=" * 80 + "\n\n")
    f.write(results.summary().as_text())

    # ===== 4. 変数重要度ランキング =====
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("変数重要度ランキング\n")
    f.write("（標準化係数の絶対値でランク付け）\n")
    f.write("=" * 80 + "\n\n")
    f.write("凡例: *** p<0.01, ** p<0.05, * p<0.10\n\n")

    importance_display = coef_summary[['変数名', '標準化係数', '重要度（|標準化係数|）', 'p値', '有意性']].copy()
    f.write(importance_display.to_string(index=False))
    f.write("\n")

# ========================================
# 10. 分析用テーブルをCSVに出力
# ========================================

X_without_const = X.drop('const', axis=1)
analysis_table = pd.concat([y.reset_index(drop=True), X_without_const.reset_index(drop=True)], axis=1)

# 年度も追加
analysis_table.insert(0, '年_共通', df_analysis['年_共通'].reset_index(drop=True).values)

# CSVに出力
analysis_table_file = os.path.join(output_dir, '02_analysis_table_fixed.csv')
analysis_table.to_csv(analysis_table_file, index=False, encoding='utf-8')

# ========================================
# 11. サマリー出力
# ========================================

print("\n" + "=" * 80)
print("回帰分析完了")
print("=" * 80)
print(f"\nR² (決定係数): {results.rsquared:.6f}")
print(f"Adjusted R²: {results.rsquared_adj:.6f}")
print(f"F統計量: {results.fvalue:.2e}")
print(f"p値 (F統計量): {results.f_pvalue:.6e}")

print(f"\n出力ファイル:")
print(f"  - {output_file}")
print(f"  - {analysis_table_file}")
