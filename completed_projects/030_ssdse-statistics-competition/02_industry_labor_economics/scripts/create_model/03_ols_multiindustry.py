"""
複数業界プーリング版：OLS回帰分析
被説明変数：失業者数（男女計）
説明変数：勤続年数、超過実労働時間数、所定内給与額（業界全体の平均）、年度ダミー、業界ダミー
対象期間：2014年、2018年-2025年（9年度）
対象業界：全10業界
出力：03_ols_results_multiindustry.txt
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
# 2. 対象業界を定義
# ========================================

# 失業者データから全業界を取得
all_industries = df_shitsugyosha['共通産業列'].unique()
print(f"利用可能な業界数: {len(all_industries)}")
print(f"業界リスト: {list(all_industries)}")

# ========================================
# 3. 複数業界のデータを統合
# ========================================

dfs_list = []

for industry in all_industries:
    # 失業者データをフィルタリング
    df_shitsugyosha_ind = df_shitsugyosha[df_shitsugyosha['共通産業列'] == industry][['年_共通', '共通産業列', '男女計（失業者数）']].copy()

    # 労働統計データをフィルタリング
    df_kinzoku_ind = df_kinzoku[df_kinzoku['共通産業列'] == industry].copy()

    if len(df_kinzoku_ind) == 0:
        continue

    # 「男」と「学」を含む区分を年度別に検出
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
    bunrui_2014_2019 = find_best_bunrui(df_kinzoku_ind, (2014, 2019))
    if bunrui_2014_2019:
        selected_bunrui_list.append(bunrui_2014_2019)
    bunrui_2020_2024 = find_best_bunrui(df_kinzoku_ind, (2020, 2024))
    if bunrui_2020_2024:
        selected_bunrui_list.append(bunrui_2020_2024)
    bunrui_2025 = find_best_bunrui(df_kinzoku_ind, (2025, 2025))
    if bunrui_2025:
        selected_bunrui_list.append(bunrui_2025)

    if not selected_bunrui_list:
        continue

    # 複数の区分値を統合
    df_kinzoku_ind_filtered = pd.concat([
        df_kinzoku_ind[df_kinzoku_ind['区　分'] == bunrui]
        for bunrui in selected_bunrui_list if bunrui is not None
    ], ignore_index=True).copy()

    # 説明変数を特定
    explvar_continuous = [col for col in df_kinzoku_ind_filtered.columns
                          if any(x in col for x in ['勤続年数', '超過実労働時間数', '所定内給与額'])]

    # 説明変数を数値型に変換
    for col in explvar_continuous:
        df_kinzoku_ind_filtered[col] = pd.to_numeric(df_kinzoku_ind_filtered[col], errors='coerce')

    # 年度別に平均値を計算
    df_kinzoku_ind_agg = df_kinzoku_ind_filtered.groupby('年_共通')[explvar_continuous].mean().reset_index()

    # マージ
    df_merged_ind = pd.merge(
        df_shitsugyosha_ind,
        df_kinzoku_ind_agg,
        on='年_共通',
        how='inner'
    )

    dfs_list.append(df_merged_ind)

# 全業界を統合
df_merged = pd.concat(dfs_list, ignore_index=True).copy()

print(f"\n統合後のサンプル数: {len(df_merged)}")
print(f"対象年度: {sorted(df_merged['年_共通'].unique())}")
print(f"対象業界（実際に使用）: {df_merged['共通産業列'].nunique()}業界")

# ========================================
# 4. 説明変数の前処理
# ========================================

# 説明変数を特定
explvar_continuous = [col for col in df_merged.columns
                      if any(x in col for x in ['勤続年数', '超過実労働時間数', '所定内給与額'])]

for col in explvar_continuous:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

df_analysis = df_merged.dropna(subset=explvar_continuous + ['男女計（失業者数）']).copy()

# ========================================
# 5. ダミー変数の作成（縮約形）
# ========================================

# 年度ダミー（最初の年度を基準に除外）
year_dummies = pd.get_dummies(df_analysis['年_共通'], prefix='year', drop_first=True)

# 業界ダミー（最初の業界を基準に除外）
industry_dummies = pd.get_dummies(df_analysis['共通産業列'], prefix='industry', drop_first=True)

df_analysis = pd.concat([df_analysis, year_dummies, industry_dummies], axis=1)

year_dummy_cols = [col for col in df_analysis.columns if col.startswith('year_')]
industry_dummy_cols = [col for col in df_analysis.columns if col.startswith('industry_')]

print(f"\n説明変数:")
print(f"  労働統計: {len(explvar_continuous)}個")
print(f"  年度ダミー: {len(year_dummy_cols)}個")
print(f"  業界ダミー: {len(industry_dummy_cols)}個")
print(f"  合計: {len(explvar_continuous) + len(year_dummy_cols) + len(industry_dummy_cols)}個（定数項含まず）")

# ========================================
# 6. OLS回帰モデルの構築と推定
# ========================================

X = df_analysis[explvar_continuous + year_dummy_cols + industry_dummy_cols].copy()
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
# 6-1. 変数重要度の計算
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
# 7. 結果をテキストファイルに出力
# ========================================

output_file = os.path.join(output_dir, '03_ols_results_multiindustry.txt')

with open(output_file, 'w', encoding='utf-8') as f:
    # ===== 1. 基本情報 =====
    f.write("=" * 80 + "\n")
    f.write("複数業界プーリング版：OLS回帰分析\n")
    f.write("被説明変数：失業者数（男女計）\n")
    f.write("説明変数：勤続年数、超過実労働時間数、所定内給与額（業界全体の平均）\n")
    f.write("        + 年度ダミー + 業界ダミー\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"対象年度：{sorted(df_analysis['年_共通'].unique())}\n")
    f.write(f"対象業界数：{df_analysis['共通産業列'].nunique()}業界\n")
    f.write(f"サンプル数：{len(df_analysis)}\n")
    f.write(f"説明変数数：{len(X.columns) - 1}\n\n")

    # ===== 2. データ集計方法の説明 =====
    f.write("=" * 80 + "\n")
    f.write("分析方針\n")
    f.write("=" * 80 + "\n\n")
    f.write("複数業界でパネルデータを構築：\n")
    f.write("・失業者数：業界全体（各年度・各業界）\n")
    f.write("・説明変数：業界全体の平均値（各年度・各業界）\n")
    f.write("・年度ダミー：経済トレンドを制御\n")
    f.write("・業界ダミー：業界固有効果を制御\n\n")

    f.write(f"結果：\n")
    f.write(f"・標準的なパネルデータOLS（被説明変数レベルで分析）\n")
    f.write(f"・サンプル数：{len(df_analysis)}（業界数×年度数）\n")
    f.write(f"・十分な自由度確保\n\n")

    # ===== 3. 完全な回帰結果サマリー =====
    f.write("=" * 80 + "\n")
    f.write("OLS回帰結果サマリー\n")
    f.write("=" * 80 + "\n\n")
    f.write(results.summary().as_text())

    # ===== 4. 変数重要度ランキング =====
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("変数重要度ランキング（TOP 20）\n")
    f.write("（標準化係数の絶対値でランク付け）\n")
    f.write("=" * 80 + "\n\n")
    f.write("凡例: *** p<0.01, ** p<0.05, * p<0.10\n\n")

    importance_display = coef_summary.head(20)[['変数名', '標準化係数', '重要度（|標準化係数|）', 'p値', '有意性']].copy()
    f.write(importance_display.to_string(index=False))
    f.write("\n")

# ========================================
# 8. 分析用テーブルをCSVに出力
# ========================================

X_without_const = X.drop('const', axis=1)
analysis_table = pd.concat([y.reset_index(drop=True), X_without_const.reset_index(drop=True)], axis=1)

# 年度と業界も追加
analysis_table.insert(0, '共通産業列', df_analysis['共通産業列'].reset_index(drop=True).values)
analysis_table.insert(0, '年_共通', df_analysis['年_共通'].reset_index(drop=True).values)

# CSVに出力
analysis_table_file = os.path.join(output_dir, '03_analysis_table_multiindustry.csv')
analysis_table.to_csv(analysis_table_file, index=False, encoding='utf-8')

# ========================================
# 9. サマリー出力
# ========================================

print("\n" + "=" * 80)
print("回帰分析完了")
print("=" * 80)
print(f"\nR² (決定係数): {results.rsquared:.6f}")
print(f"Adjusted R²: {results.rsquared_adj:.6f}")
print(f"F統計量: {results.fvalue:.2e}")
print(f"p値 (F統計量): {results.f_pvalue:.6e}")
print(f"Durbin-Watson統計量: {sm.stats.durbin_watson(results.resid):.6f}")

print(f"\n出力ファイル:")
print(f"  - {output_file}")
print(f"  - {analysis_table_file}")
