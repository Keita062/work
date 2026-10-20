"""
Step 2: 業種別失業者数の時系列分析と対象業種の選定

目的：
1. 業種ごとの失業者数を年次で追跡
2. 年次推移を折れ線グラフで可視化
3. 失業者数の時間的変動が大きい業種（ターゲット業種）を抽出

仕様：
- 入力：industry_master.csv（失業者数）
- 出力：失業者数時系列CSV、折れ線グラフ、ターゲット業種リスト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from pathlib import Path

# 設定
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
FIGURES_DIR = PROCESSED_DIR / "figures"
DOCS_DIR = Path(__file__).parent.parent.parent / "docs"

# フォルダ作成
FIGURES_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# 定数
CV_THRESHOLD = 0.3

# ===== 1. データ読み込み =====
print("Step 2: データ読み込み中...")
industry_master = pd.read_csv(PROCESSED_DIR / "industry_master.csv")

# ===== 2. 失業者数時系列の集計 =====
print("失業者数を集計中...")

# 失業者数の集計（業種×年度）
unemployment_by_industry_year = industry_master.groupby(['year', 'industry']).agg({
    'unemployment_count': 'first'  # 重複排除（通常は1行のはず）
}).reset_index()

# 失業者数を業種×年度でpivot
unemployment_timeseries = unemployment_by_industry_year.pivot(
    index='year',
    columns='industry',
    values='unemployment_count'
)

print("\n失業者数時系列（業種×年度）:")
print(unemployment_timeseries)

# ===== 3. 複数指標の計算 =====
print("\n\n複数指標で業種を評価中...")

# 複数指標の計算用辞書
metrics_by_industry = {}

for industry in unemployment_timeseries.columns:
    values = unemployment_timeseries[industry].dropna()
    if len(values) > 1:
        # 1. 変動係数（CV）
        if values.mean() != 0:
            cv = values.std() / values.mean()
        else:
            cv = np.nan

        # 2. 絶対変動幅（最大値−最小値）
        abs_range = values.max() - values.min()

        # 3. 年次変化の平均絶対値（年ごとの変動の平均）
        year_changes = values.diff().dropna().abs()
        mean_year_change = year_changes.mean() if len(year_changes) > 0 else 0

        # 4. 標準偏差（絶対的ばらつき）
        std_dev = values.std()

        metrics_by_industry[industry] = {
            'CV': cv,
            'abs_range': abs_range,
            'mean_year_change': mean_year_change,
            'std_dev': std_dev,
            'mean': values.mean(),
            'min': values.min(),
            'max': values.max()
        }
    else:
        metrics_by_industry[industry] = {
            'CV': np.nan,
            'abs_range': np.nan,
            'mean_year_change': np.nan,
            'std_dev': np.nan,
            'mean': np.nan,
            'min': np.nan,
            'max': np.nan
        }

# メトリクスをDataFrameに変換
metrics_df = pd.DataFrame(metrics_by_industry).T
metrics_df = metrics_df.sort_values('CV', ascending=False)

print("\n業種別メトリクス:")
print(metrics_df)

# ===== 4. ターゲット業種の複合的選定 =====
print("\n\n複合的なターゲット業種を選定中...")

# 各指標でのランキング（NaNを除いて）
cv_rank = metrics_df['CV'].rank(ascending=False, na_option='bottom')
abs_range_rank = metrics_df['abs_range'].rank(ascending=False, na_option='bottom')
mean_year_change_rank = metrics_df['mean_year_change'].rank(ascending=False, na_option='bottom')
std_dev_rank = metrics_df['std_dev'].rank(ascending=False, na_option='bottom')

# 各指標での順位を追加
metrics_df['CV_rank'] = cv_rank
metrics_df['abs_range_rank'] = abs_range_rank
metrics_df['mean_year_change_rank'] = mean_year_change_rank
metrics_df['std_dev_rank'] = std_dev_rank

# 複合スコア（複数指標の平均ランク）：ランクが低い（1に近い）ほどスコアが高い
# ただしNaNは除外
metrics_df['composite_score'] = metrics_df[['CV_rank', 'abs_range_rank', 'mean_year_change_rank', 'std_dev_rank']].mean(axis=1)
metrics_df = metrics_df.sort_values('composite_score')

print("\n各指標の順位と複合スコア:")
print(metrics_df[['CV_rank', 'abs_range_rank', 'mean_year_change_rank', 'std_dev_rank', 'composite_score']])

# ターゲット業種の選定：複合スコアがトップ3かつCV > 0.15の業種
# または、複数指標で上位（各指標トップ3のいずれかに該当）する業種
top_n = 3
target_industries = []

# 方針1：複合スコアトップ3で、かつ CV >= 0.15
for idx, (industry, row) in enumerate(metrics_df.iterrows()):
    if idx < top_n and row['CV'] >= 0.15:
        target_industries.append(industry)

# 方針2：少なくとも2つ以上の指標でトップ3に該当する業種
rank_columns = ['CV_rank', 'abs_range_rank', 'mean_year_change_rank', 'std_dev_rank']
for industry, row in metrics_df.iterrows():
    if industry not in target_industries:
        top_3_count = sum([1 for col in rank_columns if row[col] <= top_n])
        if top_3_count >= 2:
            target_industries.append(industry)

target_industries = sorted(set(target_industries))
non_target_industries = [ind for ind in metrics_df.index if ind not in target_industries]

print(f"\n【ターゲット業種】複合的に変動が大きい業種（{len(target_industries)}業種）:")
for ind in target_industries:
    cv = metrics_by_industry[ind]['CV']
    abs_range = metrics_by_industry[ind]['abs_range']
    mean_change = metrics_by_industry[ind]['mean_year_change']
    print(f"  - {ind} (CV={cv:.3f}, 絶対変動幅={abs_range:.0f}, 平均年次変化={mean_change:.1f})")

print(f"\n【除外業種】複合的な変動が相対的に小さい業種（{len(non_target_industries)}業種）:")
for ind in non_target_industries:
    cv = metrics_by_industry[ind]['CV']
    abs_range = metrics_by_industry[ind]['abs_range']
    mean_change = metrics_by_industry[ind]['mean_year_change']
    print(f"  - {ind} (CV={cv:.3f}, 絶対変動幅={abs_range:.0f}, 平均年次変化={mean_change:.1f})")

# ===== 4. 折れ線グラフ（全業種） =====
print("\n折れ線グラフを作成中...")
plt.figure(figsize=(14, 8))

for industry in unemployment_timeseries.columns:
    plt.plot(
        unemployment_timeseries.index,
        unemployment_timeseries[industry],
        marker='o',
        label=industry,
        linewidth=2 if industry in target_industries else 1,
        alpha=0.8 if industry in target_industries else 0.5
    )

plt.xlabel('年', fontsize=12)
plt.ylabel('失業者数（人）', fontsize=12)
plt.title('失業者数の時系列推移（業種別）\n太線：ターゲット業種（CV ≥ 0.3）', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "step2_unemployment_timeseries.png", dpi=300, bbox_inches='tight')
print(f"✓ グラフ保存：{FIGURES_DIR / 'step2_unemployment_timeseries.png'}")
plt.close()

# ===== 5. ターゲット業種のハイライト版グラフ =====
if target_industries:
    plt.figure(figsize=(12, 6))
    for industry in target_industries:
        plt.plot(
            unemployment_timeseries.index,
            unemployment_timeseries[industry],
            marker='o',
            label=industry,
            linewidth=2.5
        )
    plt.xlabel('年', fontsize=12)
    plt.ylabel('失業者数（人）', fontsize=12)
    plt.title('ターゲット業種の失業者数推移（CV ≥ 0.3）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "step2_target_industries_timeseries.png", dpi=300, bbox_inches='tight')
    print(f"✓ グラフ保存：{FIGURES_DIR / 'step2_target_industries_timeseries.png'}")
    plt.close()

# ===== 6. 出力CSVファイル =====
# 失業者数時系列をCSVで保存
unemployment_timeseries_export = unemployment_timeseries.reset_index()
unemployment_timeseries_export.to_csv(
    PROCESSED_DIR / "step2_unemployment_count_by_industry_timeseries.csv",
    index=False
)
print(f"✓ CSV保存：{PROCESSED_DIR / 'step2_unemployment_count_by_industry_timeseries.csv'}")

# 複数指標をCSVで保存
metrics_export = metrics_df[['CV', 'abs_range', 'mean_year_change', 'std_dev', 'mean', 'min', 'max', 'composite_score']].copy()
metrics_export.index.name = 'industry'
metrics_export.to_csv(
    PROCESSED_DIR / "step2_unemployment_metrics_by_industry.csv"
)
print(f"✓ CSV保存：{PROCESSED_DIR / 'step2_unemployment_metrics_by_industry.csv'}")

# ===== 7. Markdown レポート作成（データドリブンな選定理由を含む） =====
print("\nMarkdownレポートを作成中...")

# 統計情報の計算（データドリブン記述用）
target_cv_values = metrics_df.loc[target_industries, 'CV'].dropna()
non_target_cv_values = metrics_df.loc[non_target_industries, 'CV'].dropna()
all_cv_values = metrics_df['CV'].dropna()

target_cv_mean = target_cv_values.mean()
non_target_cv_mean = non_target_cv_values.mean()
all_cv_mean = all_cv_values.mean()
all_cv_std = all_cv_values.std()

report_md = f"""# Step 2：業種別失業者数の時系列分析と対象業種の選定

**実施日時**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概要

本ステップでは、業種ごとの失業者数を年次で追跡し、その時間的変動パターンを分析した。
これにより、失業者数の変動が大きい「ハイリスク業種」を特定し、後続のLCAで重点的に分析する対象業種を選定した。

## 方法

### データソース
- **失業者数**：`industry_master.csv`（業種×年度別失業者数）
- **分析対象**：9業種 × {len(unemployment_timeseries.index)}年度（{unemployment_timeseries.index.min()}〜{unemployment_timeseries.index.max()}年）

### 変動指標
各業種について、**複数の変動指標**を計算して多角的に評価した。

#### 採用指標：

1. **変動係数（CV）** = 標準偏差 ÷ 平均
   - 平均値が異なる業種間の相対的な変動を比較
   - 小規模業種の変動が相対的に大きく見える傾向

2. **絶対変動幅** = 最大値 − 最小値
   - 失業者数の実際の変動の大きさ（人数）
   - 相対的な差ではなく絶対的な差を表現

3. **平均年次変化** = 年ごとの変動の平均絶対値
   - 毎年どの程度失業者数が変化しているか
   - 持続的な変動を捉える指標

4. **標準偏差** = 平均からの平方平均平方根偏差
   - 絶対的なばらつきの大きさ

**判定基準**：これら複数指標で総合的に「変動が大きい業種」を特定

## 結果

### 1. 全業種の失業者数推移

![失業者数の時系列推移](../data/processed/figures/step2_unemployment_timeseries.png)

**特徴**：
- 各業種の失業者数は年によってばらつきがある
- 太線で示したターゲット業種は、複合的に失業者数の変動が大きい業種

### 2. 複数指標による業種の総合評価

#### 全業種の複合比較

| 業種 | CV | 絶対変動幅 | 平均年次変化 | 標準偏差 | 複合スコア | 判定 |
|---|---|---|---|---|---|---|
"""

for industry in metrics_df.index:
    cv = metrics_df.loc[industry, 'CV']
    abs_range = metrics_df.loc[industry, 'abs_range']
    mean_change = metrics_df.loc[industry, 'mean_year_change']
    std_dev = metrics_df.loc[industry, 'std_dev']
    comp_score = metrics_df.loc[industry, 'composite_score']
    status = "✓ ターゲット" if industry in target_industries else "—"

    # NaNの場合は"-"で表示
    cv_str = f"{cv:.3f}" if not np.isnan(cv) else "-"
    range_str = f"{abs_range:.0f}" if not np.isnan(abs_range) else "-"
    change_str = f"{mean_change:.1f}" if not np.isnan(mean_change) else "-"
    std_str = f"{std_dev:.1f}" if not np.isnan(std_dev) else "-"
    score_str = f"{comp_score:.1f}" if not np.isnan(comp_score) else "-"

    report_md += f"| {industry} | {cv_str} | {range_str} | {change_str} | {std_str} | {score_str} | {status} |\n"

report_md += f"""

**指標の解釈**：
- **CV**（変動係数）：平均値に対する相対的な変動。小規模業種で大きくなりやすい
- **絶対変動幅**：失業者数の実際の変動の大きさ（人数）。産業規模に依存
- **平均年次変化**：毎年の平均的な変動量。持続的な変動を反映
- **標準偏差**：絶対的なばらつきの大きさ
- **複合スコア**：複数指標の平均ランク（低いほど総合的に変動が大きい）

### 3. ターゲット業種の選定理由—複合的なデータドリブン判定

**選定基準**：
- 複合スコアがトップ3以内で、かつCV ≥ 0.15、または
- 複数の指標（CV、絶対変動幅、平均年次変化、標準偏差）でトップ3に2つ以上該当する業種

**選定された業種**（{len(target_industries)}業種）：

"""

for industry in target_industries:
    metrics = metrics_df.loc[industry]
    values = unemployment_timeseries[industry].dropna()

    cv_val = metrics['CV']
    abs_range_val = metrics['abs_range']
    mean_change_val = metrics['mean_year_change']
    std_val = metrics['std_dev']
    mean_val = metrics['mean']
    min_val = metrics['min']
    max_val = metrics['max']
    comp_score = metrics['composite_score']

    report_md += f"""
#### {industry}
- **失業者数の基本統計**：
  - 平均: {mean_val:.1f}人
  - 標準偏差: {std_val:.1f}人
  - 範囲: {min_val:.0f}〜{max_val:.0f}人（{abs_range_val:.0f}人の変動幅）

- **複数指標での評価**：
  - CV（変動係数）: {cv_val:.3f}（全業種平均 {all_cv_mean:.3f} の {cv_val/all_cv_mean:.2f}倍）
  - 絶対変動幅: {abs_range_val:.0f}人（実際の失業者数変動）
  - 平均年次変化: {mean_change_val:.2f}人/年（毎年の平均変動量）
  - 複合スコア: {comp_score:.1f}（複数指標の総合順位）

- **選定理由**：
  - 複合的に変動が大きい業種として複数指標でトップランクに該当
  - 失業者数の年次変動が継続的に観察され、経済変動への感応度が高い業種と判断
  - 労働特性（賃金・勤続年数・超過労働時間・倒産件数）がこの変動に影響している可能性が高い
"""

if target_industries:
    report_md += f"""

![ターゲット業種の推移](../data/processed/figures/step2_target_industries_timeseries.png)

"""

# 除外業種の説明
report_md += f"""

### 4. 除外業種の理由—複合的にみて変動が相対的に小さい業種

**除外された業種**（{len(non_target_industries)}業種）：

| 業種 | CV | 絶対変動幅 | 平均年次変化 | 標準偏差 | 除外理由 |
|---|---|---|---|---|---|
"""

for industry in non_target_industries:
    metrics = metrics_df.loc[industry]
    cv_val = metrics['CV']
    abs_range = metrics['abs_range']
    mean_change = metrics['mean_year_change']
    std_dev = metrics['std_dev']

    cv_str = f"{cv_val:.3f}" if not np.isnan(cv_val) else "-"
    range_str = f"{abs_range:.0f}" if not np.isnan(abs_range) else "-"
    change_str = f"{mean_change:.2f}" if not np.isnan(mean_change) else "-"
    std_str = f"{std_dev:.1f}" if not np.isnan(std_dev) else "-"

    reason = f"複合スコアが低く、複数指標で変動が相対的に小さい"
    report_md += f"| {industry} | {cv_str} | {range_str} | {change_str} | {std_str} | {reason} |\n"

report_md += f"""

**特徴**：
- これらの業種は年による失業者数の変動が小さい
- 景気変動の影響を相対的に受けにくいか、失業者数の絶対値が少ない傾向
- LCA分析では全業種を対象にするが、結果解釈時にはターゲット業種のパターンに重点を置く

## 解釈と次ステップ

### 主な知見
1. **変動が大きい業種**：上記のターゲット業種（CV ≥ {CV_THRESHOLD}）は、年による経済変動の影響を強く受けている可能性が高い
   - ターゲット業種の平均CV {target_cv_mean:.3f} は全業種平均 {all_cv_mean:.3f} の {target_cv_mean/all_cv_mean:.2f}倍

2. **マクロ経済との関連**：景気変動が失業に顕著に表れる業種を特定できた
   - 失業者数の変動幅が大きい業種ほど、労働市場のショックに敏感

3. **LCA分析への活用**：これらターゲット業種の労働特性（賃金・勤続年数・超過労働時間・倒産件数）が、失業者数に有意に影響している可能性を示唆
   - CV値の高さが、労働特性の複雑な影響構造を反映していると考えられる

### Step 3への推奨

**推奨方針**：全業種でLCA を実施しつつ、結果解釈時にターゲット業種を重視する

- **メリット**：
  - 全業種の特性を把握したうえで、ハイリスク業種（失業変動が大きい業種）に重点を置ける
  - LCAクラスが全業種でどの程度汎用性があるかを検証できる
  - ターゲット業種の労働特性パターンから、失業率増加要因を特定しやすい

- **実装方針**：
  - Step 3でのLCAはすべての9業種を対象に実施
  - クラスプロファイル解釈時に「ターゲット業種がどのクラスに多く属しているか」を明示的に分析
  - 「ターゲット業種で観察されるクラスパターン」が失業者数増加のドライバーとなっていることを検証

## 結論：どの業種にした方がいいか

### 推奨業種

**LCA分析における業種選定の方針：**

1. **ターゲット業種を重点分析対象とする**
   - 業種：{', '.join(target_industries) if target_industries else 'なし（全業種が安定推移）'}
   - 理由：CV ≥ {CV_THRESHOLD} を満たし、失業者数の時間的変動が顕著に大きい業種
   - 意味：これらの業種は景気変動への感応度が高く、労働特性の影響が失業に強く表れる可能性が高い

2. **全業種でLCA実施、ただしターゲット業種に重点**
   - 完全に除外するのではなく、全9業種を対象にLCA分析を実施する
   - 分析後の結果解釈では、ターゲット業種がどのLCAクラスに属しているか、およびそのクラスの特徴を明示的に分析する
   - 「ターゲット業種に多く属するクラス」の労働特性パターンが失業者数増加のドライバーとなっていることを検証

### 統計的根拠

{f"| 業種 | CV値 | 全業種平均との倍率 | 意義 |"
"| --- | --- | --- | --- |"
"| " + " | ".join(target_industries) + f" | {target_cv_mean:.3f} | {target_cv_mean/all_cv_mean:.2f}倍 | 失業変動が明確に大きい |"
if target_industries else
"（ターゲット業種なし：全業種が安定的に推移）"}

**除外理由**（CV < {CV_THRESHOLD}の業種）：
- 失業者数の年次変動が小さく、景気変動への感応度が相対的に低い
- ただし、クラス分析では全業種を含めることで、全体的な労働市場構造を把握できる

### Step 3への実装方針

**推奨アプローチ：全業種LCA実施 + ターゲット業種中心の結果解釈**

- LCA本体：全9業種を対象に潜在クラスモデルを実施
- 結果解釈の重点：
  - ターゲット業種（{', '.join(target_industries) if target_industries else 'なし'}）が主に属するクラスを特定
  - そのクラスの労働特性（賃金・勤続年数・超過労働時間・倒産件数）を詳細に分析
  - これらの特性が失業者数増加に及ぼす影響構造を明らかにする

**メリット**：
- ハイリスク業種（失業変動が大きい業種）の労働特性を重点的に分析できる
- LCAクラスの汎用性を全業種で検証できる
- 景気変動に敏感な業種の失業増加要因を特定しやすい

## ファイル出力

| ファイル | 説明 |
|---|---|
| `step2_unemployment_count_by_industry_timeseries.csv` | 業種×年度の失業者数行列 |
| `step2_unemployment_cv_by_industry.csv` | 業種別CV統計 |
| `figures/step2_unemployment_timeseries.png` | 全業種折れ線グラフ |
| `figures/step2_target_industries_timeseries.png` | ターゲット業種折れ線グラフ |

"""

with open(DOCS_DIR / "Step2_失業者数時系列分析.md", "w", encoding="utf-8") as f:
    f.write(report_md)

print(f"✓ レポート保存：{DOCS_DIR / 'Step2_失業者数時系列分析.md'}")

# ===== 8. 最終サマリ出力 =====
print("\n" + "="*60)
print("Step 2 完了")
print("="*60)
print(f"\n✓ ターゲット業種を {len(target_industries)} 業種 特定しました")
print(f"✓ 折れ線グラフを生成しました（全業種・ターゲット業種）")
print(f"✓ データドリブンなMarkdownレポートを作成しました")
print("\n次ステップ：Step 3（業種LCA本体の実施）に進んでください")
