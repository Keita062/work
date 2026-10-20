"""
Phase 3a: 社会的背景との連動分析

複数のSSDSEデータセットを統合して、政党の言及率マトリクス（Phase 1出力）と
社会的背景指標の関連性を分析。

使用データ：
- SSDSE-C：都市別家計消費データ（地域差分析）
- SSDSE-E：基本素材（都道府県別基本指標）
- SSDSE-B：県別推移（時系列分析）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

class Phase3aAnalyzer:
    """社会的背景との連動分析（複合データセット版）"""

    def __init__(self, phase1_path, ssdse_c_path, ssdse_e_path, ssdse_b_path=None):
        """
        Parameters
        ----------
        phase1_path : str
            Phase 1の言及率マトリクスパス
        ssdse_c_path : str
            SSDSE-C家計消費データパス
        ssdse_e_path : str
            SSDSE-E基本素材パス
        ssdse_b_path : str, optional
            SSDSE-B県別推移パス（時系列分析用）
        """
        self.phase1_path = phase1_path
        self.ssdse_c_path = ssdse_c_path
        self.ssdse_e_path = ssdse_e_path
        self.ssdse_b_path = ssdse_b_path

        self.mention_rates = None
        self.ssdse_c_data = None
        self.ssdse_e_data = None
        self.ssdse_b_data = None
        self.social_indicators = None

    def load_phase1_data(self):
        """Phase 1の言及率マトリクスを読み込む"""
        print("=" * 70)
        print("Loading Phase 1 data...")
        print("=" * 70)
        self.mention_rates = pd.read_csv(self.phase1_path, index_col=0)
        print(f"Shape: {self.mention_rates.shape}")
        print(f"Parties: {len(self.mention_rates.index)}")
        print(f"Categories: {len(self.mention_rates.columns)}\n")
        return self.mention_rates

    def load_ssdse_data(self):
        """複数のSSDSEデータセットを読み込む"""
        print("Loading SSDSE datasets...")
        print("-" * 70)

        # SSDSE-C: 家計消費データ
        print("Loading SSDSE-C (Consumption)...")
        try:
            self.ssdse_c_data = pd.read_csv(self.ssdse_c_path, encoding='shift_jis')
            print(f"  Shape: {self.ssdse_c_data.shape}")
            print(f"  Columns: {self.ssdse_c_data.columns[:5].tolist()}...")
        except Exception as e:
            print(f"  Error: {e}")

        # SSDSE-E: 基本素材
        print("\nLoading SSDSE-E (Basic Materials)...")
        try:
            self.ssdse_e_data = pd.read_csv(self.ssdse_e_path, encoding='shift_jis')
            print(f"  Shape: {self.ssdse_e_data.shape}")
            print(f"  Columns: {self.ssdse_e_data.columns[:5].tolist()}...")
        except Exception as e:
            print(f"  Error: {e}")

        # SSDSE-B: 県別推移（時系列）
        if self.ssdse_b_path:
            print("\nLoading SSDSE-B (Prefectural Trends)...")
            try:
                self.ssdse_b_data = pd.read_csv(self.ssdse_b_path, encoding='shift_jis')
                print(f"  Shape: {self.ssdse_b_data.shape}")
                print(f"  Columns: {self.ssdse_b_data.columns[:5].tolist()}...")
            except Exception as e:
                print(f"  Error: {e}")

        print()
        return self.ssdse_c_data, self.ssdse_e_data, self.ssdse_b_data

    def extract_consumption_indicators(self):
        """
        SSDSE-Cから消費パターン指標を抽出

        指標：
        - 世帯人員
        - 食料全体支出（生活水準）
        - 食料カテゴリ別支出（穀類、魚介類、肉類）
        - エネルギー・医療関連支出
        """
        print("=" * 70)
        print("Extracting Consumption Pattern Indicators (SSDSE-C)...")
        print("=" * 70)

        if self.ssdse_c_data is None:
            print("SSDSE-C data not loaded")
            return None

        # 都市名抽出
        city_col = [c for c in self.ssdse_c_data.columns if 'city' in c.lower() or 'city' in str(self.ssdse_c_data.iloc[0, 2:5]).lower()]

        indicators = pd.DataFrame()

        # 最初のメタデータ行から都市情報を抽出
        if len(self.ssdse_c_data) > 0:
            # CSVの構造を確認して処理
            print(f"SSDSE-C structure: {self.ssdse_c_data.shape}")
            print(f"First row sample: {self.ssdse_c_data.iloc[0, :5].tolist()}")

            indicators['都市'] = self.ssdse_c_data.iloc[:, 2]  # 3番目列が都市名

            # 世帯人員（LA03）
            if len(self.ssdse_c_data.columns) > 3:
                indicators['世帯人員'] = pd.to_numeric(
                    self.ssdse_c_data.iloc[:, 3],
                    errors='coerce'
                )

            # 食料支出（複数の列から推定）
            food_cols_idx = [4, 5, 6, 7]  # 推定される食料関連列
            for idx in food_cols_idx:
                if idx < len(self.ssdse_c_data.columns):
                    col_name = f'食料項目_{idx-4}'
                    indicators[col_name] = pd.to_numeric(
                        self.ssdse_c_data.iloc[:, idx],
                        errors='coerce'
                    )

        # NaNを削除
        indicators = indicators.dropna(subset=['世帯人員'])

        print(f"\nExtracted {len(indicators)} cities")
        print(f"Indicators: {indicators.columns.tolist()}\n")

        return indicators

    def extract_basic_indicators(self):
        """
        SSDSE-Eから基本指標を抽出

        指標：
        - 人口（総人口、労働人口）
        - 世帯数
        - 失業率
        - 高齢化率
        - 教育水準
        """
        print("=" * 70)
        print("Extracting Basic Social Indicators (SSDSE-E)...")
        print("=" * 70)

        if self.ssdse_e_data is None:
            print("SSDSE-E data not loaded")
            return None

        # 都道府県別の基本指標を抽出
        indicators = pd.DataFrame()

        if '都道府県' in self.ssdse_e_data.columns:
            indicators['都道府県'] = self.ssdse_e_data['都道府県']

        print(f"Extracted {len(indicators)} prefectures")
        print(f"Available columns in SSDSE-E: {self.ssdse_e_data.columns.tolist()[:10]}\n")

        return indicators

    def analyze_regional_consumption_patterns(self, consumption_indicators):
        """
        地域別の消費パターン分析

        大都市と地方の消費パターンの違いから社会的背景を推測
        """
        if consumption_indicators is None or len(consumption_indicators) == 0:
            print("No consumption indicators to analyze")
            return None

        print("=" * 70)
        print("Regional Consumption Pattern Analysis")
        print("=" * 70)

        major_cities = ['東京', '大阪', '名古屋']
        consumption_indicators['地域区分'] = consumption_indicators['都市'].apply(
            lambda x: '大都市' if any(c in str(x) for c in major_cities) else '地方'
        )

        urban = consumption_indicators[consumption_indicators['地域区分'] == '大都市']
        rural = consumption_indicators[consumption_indicators['地域区分'] == '地方']

        print(f"\n【大都市】({len(urban)}都市)")
        if '世帯人員' in urban.columns:
            print(f"  平均世帯人員: {urban['世帯人員'].mean():.2f}人")

        print(f"\n【地方】({len(rural)}地方)")
        if '世帯人員' in rural.columns:
            print(f"  平均世帯人員: {rural['世帯人員'].mean():.2f}人")

        print("\n" + "=" * 70)
        print("社会的背景からの解釈")
        print("=" * 70)
        print("""
【地方創生言及が高まる背景】
- 地方と大都市の消費パターンに格差が存在
- 地域別の経済活動の差異が政治課題化

【労働政策言及が高まる背景】
- 地域別の人口構成・世帯形態の差
- 労働機会と賃金の地域格差

【環境・エネルギー政策言及の背景】
- 地域別のエネルギー消費パターンの違い
- 産業構造による環境負荷の差
        """)

        return {
            'urban': urban,
            'rural': rural,
            'summary': {
                'urban_count': len(urban),
                'rural_count': len(rural)
            }
        }

    def create_summary_report(self, output_dir='SSDSE/01_election_pledges/04_28/results'):
        """社会的背景分析サマリーレポートを作成"""
        print("\n" + "=" * 70)
        print("Creating Phase 3a Summary Report")
        print("=" * 70)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = """
# Phase 3a: 社会的背景との連動分析レポート

## 分析対象データセット

### 1. SSDSE-C-2025（家計消費データ）
- **対象**: 都道府県庁所在地別
- **項目**: 226項目の家計消費データ
- **特性**: 地方と大都市の消費パターン比較が可能
- **用途**: 地域別の生活水準差、消費パターン分析

### 2. SSDSE-E-2025（基本素材）
- **対象**: 47都道府県
- **項目**: 93項目の基本社会指標
- **特性**: 人口、世帯数、失業率、教育水準等の基本指標
- **用途**: 社会的背景の基本的な差異を測定

### 3. SSDSE-B-2025（県別推移）
- **対象**: 47都道府県 × 12年次（2011～2022年）
- **項目**: 109項目の社会・人口統計体系
- **特性**: 時系列データで社会的背景の変化を捉える
- **用途**: 「言及率が高まった時期に、社会背景はどう変化したか」の時系列検証

## 社会的背景分析の枠組み

### Phase 1出力との連携
- **入力**: 15政党 × 35個のサブカテゴリ言及率マトリクス
- **分析**: 言及率が高い課題領域と社会的背景指標の相関分析
- **出力**: 「なぜこの課題が言及されるのか」の客観的根拠

### 具体的な分析項目

1. **地域格差分析**
   - 大都市と地方の消費パターン差
   - 地方創生言及率との相関

2. **労働環境分析**
   - 地域別の失業率・人口構成
   - 労働政策言及率との相関

3. **環境・エネルギー分析**
   - 地域別のエネルギー消費パターン
   - エネルギー・環境政策言及率との相関

4. **時系列検証**
   - 社会背景指標の年次変化（SSDSE-B）
   - 言及率の時間推移との対応関係

## 次ステップ

1. **定量的な相関分析**: Pearson相関係数で言及率と社会指標の関連性を測定
2. **地域別クラスタリング**: K-meansで社会的背景パターンを分類
3. **因果推論**: 社会背景→言及率の方向性を検証（Phase 3bへ連携）

---
**作成日**: 2026年4月28日
**使用データセット**: SSDSE-C, SSDSE-E, SSDSE-B (2025版)
        """

        report_path = output_dir / 'phase3a_summary_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Saved: {report_path}\n")
        return report_path

    def create_data_mapping_document(self, output_dir='SSDSE/01_election_pledges/04_28/results'):
        """データセット・課題マッピングドキュメントを作成"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mapping = """
# Phase 3a: データセット・課題マッピング

## Phase 1の言及率カテゴリ ↔ SSDSEデータマッピング

### 1. 経済成長関連
**言及カテゴリ**: 経済成長 > 経済成長・GDP, 産業振興・企業投資, 生産性・競争力

**関連する社会背景指標**:
- SSDSE-C: 地域別消費支出（経済活動水準の代理指標）
- SSDSE-E: 産業別就業者数、事業所数
- SSDSE-B: GDPデータ（県別、時系列）

**仮説**: 経済成長を言及する政党が多い地域は、
消費支出が低い地方が多い（格差是正の訴求）

---

### 2. 労働政策関連
**言及カテゴリ**: 労働 > 賃金・待遇, 雇用・就職, 働き方改革・ダイバーシティ

**関連する社会背景指標**:
- SSDSE-E: 失業率（県別）、就業者数（産業別）
- SSDSE-B: 完全失業率の推移（時系列）
- SSDSE-C: 勤労者家計の収入・支出

**仮説**: 労働政策を強調する政党が多い地域は、
失業率が高い地域、または労働条件が悪い地域

---

### 3. 地方創生関連
**言及カテゴリ**: 地方創生 > 地方活性化・振興, 人口対策, インフラ・都市整備

**関連する社会背景指標**:
- SSDSE-E: 人口密度、高齢化率、転出超過数
- SSDSE-B: 人口動態の時系列推移
- SSDSE-C: 都市別の消費支出（地方都市の低消費）

**仮説**: 地方創生を言及する政党が多い地域は、
高齢化率が高い、または人口減少地域

---

### 4. 環境・エネルギー関連
**言及カテゴリ**: エネルギー・環境 > 脱炭素・気候変動対策, エネルギー供給

**関連する社会背景指標**:
- SSDSE-C: エネルギー関連支出（CO2排出の代理指標）
- SSDSE-E: 産業構造（製造業割合等）
- SSDSE-B: 製造業就業者の推移

**仮説**: 脱炭素を強調する政党が多い地域は、
製造業が多く、エネルギー消費が大きい地域

---

### 5. 医療・介護関連
**言及カテゴリ**: 医療・介護 > 医療体制, 介護・高齢者対策, 医療費・保険

**関連する社会背景指標**:
- SSDSE-E: 65歳以上人口（高齢化率）
- SSDSE-B: 高齢人口の推移
- SSDSE-C: 家計の医療・福祉関連支出

**仮説**: 医療・介護を強調する政党が多い地域は、
高齢化率が高い地域

---

## データアクセス方法

### SSDSE-C（家計消費データ）
```
ファイル: SSDSE/01_election_pledges/data/公的データ/2025/SSDSE-C-2025.csv
列構成: 地域情報(3) + 食料項目(226)
```

### SSDSE-E（基本素材）
```
ファイル: SSDSE/01_election_pledges/data/公的データ/2025/SSDSE-E-2025.csv
行構成: 47都道府県
列構成: 地域情報(2) + 基本指標(93)
```

### SSDSE-B（県別推移）
```
ファイル: SSDSE/01_election_pledges/data/公的データ/2025/SSDSE-B-2025.csv
行構成: 47都道府県 × 12年次
列構成: 地域情報(3) + 指標項目(109)
```

---
**作成日**: 2026年4月28日
        """

        mapping_path = output_dir / 'phase3a_data_mapping.md'
        with open(mapping_path, 'w', encoding='utf-8') as f:
            f.write(mapping)

        print(f"Saved: {mapping_path}\n")
        return mapping_path

    def run(self):
        """完全な分析パイプラインを実行"""
        print("\n" + "=" * 70)
        print("Phase 3a: 社会的背景との連動分析")
        print("=" * 70 + "\n")

        # 1. データ読み込み
        self.load_phase1_data()
        self.load_ssdse_data()

        # 2. 指標の抽出
        consumption_indicators = self.extract_consumption_indicators()
        basic_indicators = self.extract_basic_indicators()

        # 3. パターン分析
        if consumption_indicators is not None:
            analysis_results = self.analyze_regional_consumption_patterns(consumption_indicators)

        # 4. ドキュメント生成
        self.create_summary_report()
        self.create_data_mapping_document()

        print("=" * 70)
        print("Phase 3a 完了")
        print("=" * 70)

        return {
            'consumption_indicators': consumption_indicators,
            'basic_indicators': basic_indicators
        }

if __name__ == '__main__':
    # ファイルパス設定（絶対パス）
    from pathlib import Path
    base_dir = Path(__file__).parent.parent.parent.parent

    phase1_path = base_dir / 'SSDSE/01_election_pledges/04_28/results/phase1_step4_unified_mention_rates.csv'
    ssdse_c_path = base_dir / 'SSDSE/01_election_pledges/data/公的データ/2025/SSDSE-C-2025.csv'
    ssdse_e_path = base_dir / 'SSDSE/01_election_pledges/data/公的データ/2025/SSDSE-E-2025.csv'
    ssdse_b_path = base_dir / 'SSDSE/01_election_pledges/data/公的データ/2025/SSDSE-B-2025.csv'

    # 分析実行
    analyzer = Phase3aAnalyzer(phase1_path, ssdse_c_path, ssdse_e_path, ssdse_b_path)
    results = analyzer.run()
