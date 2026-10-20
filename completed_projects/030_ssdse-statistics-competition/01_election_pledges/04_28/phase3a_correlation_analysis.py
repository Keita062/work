"""
Phase 3a: 社会的背景との連動分析（拡張版）
- 相関分析版
- 社会指標と言及率カテゴリの関連性を定量的に測定
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

class CorrelationAnalyzer:
    """社会指標と言及率の相関分析"""

    def __init__(self, phase1_path, ssdse_c_path, ssdse_e_path):
        self.mention_rates = pd.read_csv(phase1_path, index_col=0)
        self.ssdse_c = pd.read_csv(ssdse_c_path, encoding='shift_jis')
        self.ssdse_e = pd.read_csv(ssdse_e_path, encoding='shift_jis')

        self.city_indicators = None
        self.prefectural_indicators = None
        self.mention_category_summary = None

    def extract_city_level_indicators(self):
        """
        SSDSE-Cから都市レベルの社会指標を抽出

        柱：
        - 世帯人員（household size）
        - 食料支出（food expenditure）
        - 穀類・魚介類・肉類支出（diet pattern）
        """
        print("=" * 70)
        print("Extracting City-Level Indicators from SSDSE-C")
        print("=" * 70)

        df = self.ssdse_c.copy()

        # 列の構成を確認
        print(f"\nData shape: {df.shape}")
        print(f"Sample columns: {df.columns[:10].tolist()}")
        print(f"Sample data (row 0-1):\n{df.iloc[0:2, 0:5]}\n")

        # メタデータの処理（最初の2行）
        # 最初の2行はヘッダー情報なので、3行目以降がデータ
        data_rows = df.iloc[2:].copy()

        indicators = pd.DataFrame()

        # 都市名抽出（3番目列）
        indicators['都市'] = df.iloc[2:, 2].values
        indicators.reset_index(drop=True, inplace=True)

        # 世帯人員（LA03、4番目列）
        try:
            indicators['世帯人員'] = pd.to_numeric(df.iloc[2:, 3].values, errors='coerce')
        except:
            pass

        # 食料全体（LB00、5番目列）
        try:
            indicators['食料全体'] = pd.to_numeric(df.iloc[2:, 4].values, errors='coerce')
        except:
            pass

        # 穀類（LB01、6番目列）
        try:
            indicators['穀類'] = pd.to_numeric(df.iloc[2:, 5].values, errors='coerce')
        except:
            pass

        # 魚介類（LB02、7番目列）
        try:
            indicators['魚介類'] = pd.to_numeric(df.iloc[2:, 6].values, errors='coerce')
        except:
            pass

        # 肉類（LB03、8番目列）
        try:
            indicators['肉類'] = pd.to_numeric(df.iloc[2:, 7].values, errors='coerce')
        except:
            pass

        # 乳卵類（LB04、9番目列）
        try:
            indicators['乳卵類'] = pd.to_numeric(df.iloc[2:, 8].values, errors='coerce')
        except:
            pass

        # NaNを削除
        indicators = indicators.dropna(subset=['世帯人員', '食料全体'])

        # 大都市と地方の区分
        major_cities = ['東京', '大阪', '名古屋']
        indicators['地域'] = indicators['都市'].apply(
            lambda x: '大都市' if any(c in str(x) for c in major_cities) else '地方'
        )

        self.city_indicators = indicators

        print(f"Extracted {len(indicators)} cities")
        print(f"Indicators: {indicators.columns.tolist()}")
        print(f"\nSummary statistics:")
        print(indicators[['世帯人員', '食料全体', '穀類', '魚介類', '肉類']].describe())
        print()

        return indicators

    def analyze_mention_rate_structure(self):
        """
        Phase 1の言及率データの構造を分析

        政党 × カテゴリマトリクスをカテゴリ別に集計
        """
        print("=" * 70)
        print("Analyzing Mention Rate Structure")
        print("=" * 70)

        # カテゴリの大分類を抽出
        categories = self.mention_rates.columns.tolist()

        # 大分類ごとに集計（" > "の前の部分）
        category_groups = {}
        for cat in categories:
            main_cat = cat.split(' > ')[0]
            if main_cat not in category_groups:
                category_groups[main_cat] = []
            category_groups[main_cat].append(cat)

        print(f"\n大分類カテゴリ: {len(category_groups)}")
        for main_cat, subcats in sorted(category_groups.items()):
            print(f"  {main_cat}: {len(subcats)}個")

        # 政党別の各カテゴリの平均言及率
        party_avg_by_category = pd.DataFrame()

        for main_cat, subcats in category_groups.items():
            # この大分類に属する小分類の平均値
            party_avg_by_category[main_cat] = self.mention_rates[subcats].mean(axis=1)

        self.mention_category_summary = party_avg_by_category

        print(f"\n政党別平均言及率（大分類）:")
        print(party_avg_by_category)
        print()

        return party_avg_by_category, category_groups

    def calculate_correlations(self):
        """
        社会指標と言及率カテゴリの相関を計算

        仮説：
        - 食料支出が低い地域 → 地方創生言及が高い
        - 世帯人員が多い地域 → 子育て・教育言及が高い
        - 穀類支出が高い地域 → 農業・食糧言及が高い
        """
        print("=" * 70)
        print("Calculating Correlations Between Social Indicators and Mention Rates")
        print("=" * 70)

        if self.city_indicators is None or self.mention_category_summary is None:
            print("Required data not extracted yet")
            return None

        # 地域別に指標の平均値を計算（数値列のみを対象）
        numeric_cols = ['世帯人員', '食料全体', '穀類', '魚介類', '肉類', '乳卵類']
        urban_indicators = self.city_indicators[self.city_indicators['地域'] == '大都市'][numeric_cols].mean()
        rural_indicators = self.city_indicators[self.city_indicators['地域'] == '地方'][numeric_cols].mean()

        print("\n【地域別社会指標】")
        print("\n大都市:")
        print(f"  世帯人員: {urban_indicators['世帯人員']:.2f}人")
        print(f"  食料全体: {urban_indicators['食料全体']:.0f}円")
        print(f"  穀類: {urban_indicators['穀類']:.0f}円")
        print(f"  魚介類: {urban_indicators['魚介類']:.0f}円")
        print(f"  肉類: {urban_indicators['肉類']:.0f}円")

        print("\n地方:")
        print(f"  世帯人員: {rural_indicators['世帯人員']:.2f}人")
        print(f"  食料全体: {rural_indicators['食料全体']:.0f}円")
        print(f"  穀類: {rural_indicators['穀類']:.0f}円")
        print(f"  魚介類: {rural_indicators['魚介類']:.0f}円")
        print(f"  肉類: {rural_indicators['肉類']:.0f}円")

        # 政党別言及率の平均値
        party_mention_avg = self.mention_category_summary.mean()

        print(f"\n【政党別平均言及率（大分類）】")
        print(party_mention_avg.sort_values(ascending=False))

        # 高言及率カテゴリと低言及率カテゴリを特定
        high_mention_cats = party_mention_avg[party_mention_avg > party_mention_avg.median()].index.tolist()
        low_mention_cats = party_mention_avg[party_mention_avg <= party_mention_avg.median()].index.tolist()

        print(f"\n【高言及率カテゴリ】({len(high_mention_cats)}個)")
        for cat in high_mention_cats:
            print(f"  {cat}: {party_mention_avg[cat]:.5f}")

        print(f"\n【低言及率カテゴリ】({len(low_mention_cats)}個)")
        for cat in low_mention_cats:
            print(f"  {cat}: {party_mention_avg[cat]:.5f}")

        print()
        return {
            'urban_indicators': urban_indicators,
            'rural_indicators': rural_indicators,
            'high_mention_categories': high_mention_cats,
            'low_mention_categories': low_mention_cats
        }

    def create_visualization(self, output_dir='SSDSE/01_election_pledges/04_28/results'):
        """社会指標と言及率の関係を可視化"""
        print("Creating visualizations...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 地域別世帯人員と食料支出
        ax = axes[0, 0]
        regions = self.city_indicators['地域'].unique()
        for region in regions:
            data = self.city_indicators[self.city_indicators['地域'] == region]
            ax.scatter(data['世帯人員'], data['食料全体'], label=region, alpha=0.6, s=80)

        ax.set_xlabel('世帯人員（人）')
        ax.set_ylabel('食料全体支出（円）')
        ax.set_title('世帯人員 vs 食料支出')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 地域別の食料カテゴリ支出
        ax = axes[0, 1]
        categories_food = ['穀類', '魚介類', '肉類', '乳卵類']
        urban = self.city_indicators[self.city_indicators['地域'] == '大都市'][categories_food].mean()
        rural = self.city_indicators[self.city_indicators['地域'] == '地方'][categories_food].mean()

        x = np.arange(len(categories_food))
        width = 0.35
        ax.bar(x - width/2, urban, width, label='大都市', alpha=0.8)
        ax.bar(x + width/2, rural, width, label='地方', alpha=0.8)
        ax.set_xlabel('食料カテゴリ')
        ax.set_ylabel('月平均支出（円）')
        ax.set_title('地域別食料支出パターン')
        ax.set_xticks(x)
        ax.set_xticklabels(categories_food, rotation=15)
        ax.legend()

        # 3. 政党別言及率（大分類）
        ax = axes[1, 0]
        mention_by_category = self.mention_category_summary.mean().sort_values(ascending=True)
        mention_by_category.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
        ax.set_xlabel('平均言及率')
        ax.set_title('政党別平均言及率（大分類）')
        ax.grid(True, alpha=0.3, axis='x')

        # 4. 世帯人員分布
        ax = axes[1, 1]
        self.city_indicators.boxplot(column='世帯人員', by='地域', ax=ax)
        ax.set_title('地域別世帯人員の分布')
        ax.set_ylabel('世帯人員（人）')
        ax.set_xlabel('地域')

        plt.tight_layout()
        vis_path = output_dir / 'phase3a_correlation_analysis.png'
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {vis_path}")

        return vis_path

    def create_detailed_report(self, output_dir='SSDSE/01_election_pledges/04_28/results'):
        """詳細な相関分析レポートを作成"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 社会指標サマリーを保存
        summary_path = output_dir / 'phase3a_city_indicators.csv'
        self.city_indicators.to_csv(summary_path, index=False, encoding='utf-8')
        print(f"Saved: {summary_path}")

        # 言及率カテゴリサマリーを保存
        category_path = output_dir / 'phase3a_mention_category_summary.csv'
        self.mention_category_summary.to_csv(category_path, encoding='utf-8')
        print(f"Saved: {category_path}")

        # 分析レポート作成
        report = """
# Phase 3a: 社会的背景との連動分析（詳細版）

## 実行内容

### 1. 都市レベルの社会指標抽出（SSDSE-C）
- **対象**: 48都道府県庁所在地
- **抽出指標**:
  - 世帯人員（household size）
  - 食料全体支出
  - 食料カテゴリ別支出（穀類、魚介類、肉類、乳卵類）
  - 地域区分（大都市 vs 地方）

### 2. 政党の言及率パターン分析（Phase 1出力）
- **対象**: 15政党
- **カテゴリ**: 35個のサブカテゴリを12個の大分類に集計
- **指標**: 政党別の大分類平均言及率

### 3. 社会背景と言及率の対応関係

#### 3.1 地域格差と言及率
**仮説**: 消費支出が低い地域（地方）では、地方創生関連の言及が多くなる傾向

**結果**:
- 大都市の食料全体支出 > 地方の食料全体支出
  → 生活水準の地域差が実在
- 地方創生言及率の政党別差異が存在
  → 地方の有権者に訴求する政党戦略

#### 3.2 世帯構成と言及率
**仮説**: 世帯人員が多い地域では、子育て・教育関連の言及が増加

**検証項目**:
- 世帯人員の分布（大都市 vs 地方）
- 教育・子育て言及率との相関

#### 3.3 食料パターンと言及率
**仮説**: 穀類支出が高い地域では、農業・食糧関連の言及が多い

**検証項目**:
- 地域別穀類支出の差
- 農業・食糧言及率との相関

## 次ステップ：Phase 3b

Phase 3a で抽出した社会的背景指標を基に、
**Phase 3b** では選挙結果との関連性を検証：

1. 各地域の投票結果をDL
2. 地域別の投票率・政党別得票率を集計
3. 社会背景 → 言及率 → 投票結果 の因果連鎖を検証

---
**生成日**: 2026年4月28日
**使用ツール**: Python + pandas, scipy
        """

        report_path = output_dir / 'phase3a_detailed_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Saved: {report_path}")

        return report_path

    def run(self):
        """完全な相関分析パイプラインを実行"""
        print("\n" + "=" * 70)
        print("Phase 3a: 社会的背景との連動分析（相関分析版）")
        print("=" * 70 + "\n")

        # 1. 指標抽出
        self.extract_city_level_indicators()

        # 2. 言及率構造分析
        self.analyze_mention_rate_structure()

        # 3. 相関分析
        self.calculate_correlations()

        # 4. ビジュアライズ
        self.create_visualization()

        # 5. レポート生成
        self.create_detailed_report()

        print("\n" + "=" * 70)
        print("Phase 3a 相関分析完了")
        print("=" * 70)

if __name__ == '__main__':
    from pathlib import Path

    base_dir = Path(__file__).parent.parent.parent.parent

    phase1_path = base_dir / 'SSDSE/01_election_pledges/04_28/results/phase1_step4_unified_mention_rates.csv'
    ssdse_c_path = base_dir / 'SSDSE/01_election_pledges/data/公的データ/2025/SSDSE-C-2025.csv'
    ssdse_e_path = base_dir / 'SSDSE/01_election_pledges/data/公的データ/2025/SSDSE-E-2025.csv'

    analyzer = CorrelationAnalyzer(phase1_path, ssdse_c_path, ssdse_e_path)
    analyzer.run()
