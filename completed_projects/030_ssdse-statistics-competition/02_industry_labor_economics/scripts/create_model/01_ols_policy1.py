"""
方針1：OLS回帰分析（サービス業他）
被説明変数：サービス業他の失業者数（男女計）
説明変数：勤続年数、超過実労働時間数、所定内給与額（事業者人数別）、年度ダミー
対象期間：2014年、2018年-2025年（9年度）
出力：01_ols_results.txt（一つのテキストファイル）
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
# 2. データの準備と統合
# ========================================

target_industry = 'サービス業他'

# 失業者データをフィルタリング
df_shitsugyosha_filtered = df_shitsugyosha[df_shitsugyosha['共通産業列'] == target_industry].copy()

# 勤続年数等データをフィルタリング（男女計学歴計のみ）
# 複数年度間で区分形式が異なるため、年度別に適切な区分を選択
df_service_kinzoku = df_kinzoku[df_kinzoku['共通産業列'] == target_industry]

# 「男」と「学」を含む区分値を年度別に取得
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

# 年度別に区分を選択
selected_bunrui_list = []

# 2014, 2018-2019用
bunrui_2014_2019 = find_best_bunrui(df_service_kinzoku, (2014, 2019))
if bunrui_2014_2019:
    selected_bunrui_list.append(bunrui_2014_2019)

# 2020-2024用
bunrui_2020_2024 = find_best_bunrui(df_service_kinzoku, (2020, 2024))
if bunrui_2020_2024:
    selected_bunrui_list.append(bunrui_2020_2024)

# 2025用
bunrui_2025 = find_best_bunrui(df_service_kinzoku, (2025, 2025))
if bunrui_2025:
    selected_bunrui_list.append(bunrui_2025)

# 複数の区分値を統合
df_kinzoku_filtered = pd.concat([
    df_service_kinzoku[df_service_kinzoku['区　分'] == bunrui]
    for bunrui in selected_bunrui_list if bunrui is not None
], ignore_index=True).copy()

# マージ
merge_keys = ['年_共通', '共通産業列']
df_merged = pd.merge(
    df_shitsugyosha_filtered,
    df_kinzoku_filtered,
    on=merge_keys,
    how='inner'
)

# ========================================
# 3. 説明変数の選択と前処理
# ========================================

# マージ後のカラムから勤続年数、超過労働時間、給与額に関する列を取得
explvar_continuous = [col for col in df_merged.columns
                      if any(x in col for x in ['勤続年数', '超過実労働時間数', '所定内給与額'])]

# 数値型に変換
for col in explvar_continuous:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

df_analysis = df_merged.dropna(subset=explvar_continuous + ['男女計（失業者数）'])

# ========================================
# 4. 年度ダミー変数の作成
# ========================================

year_dummies = pd.get_dummies(df_analysis['年_共通'], prefix='year', drop_first=True)
df_analysis = pd.concat([df_analysis, year_dummies], axis=1)
dummy_cols = [col for col in df_analysis.columns if col.startswith('year_')]

# ========================================
# 5. OLS回帰モデルの構築と推定
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
# 5-1. 変数重要度の計算
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
# 6. 結果をテキストファイルに出力
# ========================================

output_file = os.path.join(output_dir, '01_ols_results.txt')

with open(output_file, 'w', encoding='utf-8') as f:
    # ===== 1. 基本情報 =====
    f.write("=" * 80 + "\n")
    f.write("方針1：OLS回帰分析（サービス業他）\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"対象産業：{target_industry}\n")
    f.write(f"対象年度：{sorted(df_analysis['年_共通'].unique())}\n")
    f.write(f"サンプル数：{len(df_analysis)}\n")
    f.write(f"説明変数数：{len(X.columns) - 1}\n\n")

    # ===== 2. 完全な回帰結果サマリー =====
    f.write("=" * 80 + "\n")
    f.write("OLS回帰結果サマリー\n")
    f.write("=" * 80 + "\n\n")
    f.write(results.summary().as_text())

    # ===== 3. 変数重要度ランキング =====
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("変数重要度ランキング\n")
    f.write("（標準化係数の絶対値でランク付け）\n")
    f.write("=" * 80 + "\n\n")
    f.write("凡例: *** p<0.01, ** p<0.05, * p<0.10\n\n")

    importance_display = coef_summary[['変数名', '標準化係数', '重要度（|標準化係数|）', 'p値', '有意性']].copy()
    f.write(importance_display.to_string(index=False))
    f.write("\n")

# ========================================
# 7. 分析用テーブルをCSVに出力
# ========================================

# X（定数項なし）とyを結合したテーブルを作成
X_without_const = X.drop('const', axis=1)
analysis_table = pd.concat([y.reset_index(drop=True), X_without_const.reset_index(drop=True)], axis=1)

# 年度と産業列も追加
analysis_table.insert(0, '年_共通', df_analysis['年_共通'].reset_index(drop=True))
analysis_table.insert(1, '共通産業列', df_analysis['共通産業列'].reset_index(drop=True))

# CSVに出力
analysis_table_file = os.path.join(output_dir, '01_analysis_table.csv')
analysis_table.to_csv(analysis_table_file, index=False, encoding='utf-8')

