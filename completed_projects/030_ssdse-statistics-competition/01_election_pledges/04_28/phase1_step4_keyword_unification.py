#!/usr/bin/env python3
"""
Phase 1 Step 4: 複合キーワードの統一

目的：「脱炭素」「脱炭素社会」「カーボンニュートラル」など、同じ概念が複数の表現で言及される
      場合に、それらを親キーワードで統一し、重複カウントを排除する。

入力：修辞除去後のテキスト（SSDSE/04_28/data）
出力：統一後の言及率マトリクス（細分化されたサブカテゴリ × 政党）
"""

import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Set, List, Tuple
import csv
import pandas as pd
import numpy as np

# janomeライブラリを試す
try:
    from janome.tokenizer import Tokenizer
    tokenizer = Tokenizer()
    USE_JANOME = True
except ImportError:
    print("[WARN] janomeが見つかりません。簡易形態素解析を使用します。")
    USE_JANOME = False


class KeywordUnifier:
    """複合キーワード統一クラス"""

    def __init__(self, text_dir: Path, output_dir: Path):
        """
        Args:
            text_dir: 修辞除去済みテキストのディレクトリ
            output_dir: 出力ディレクトリ
        """
        self.text_dir = Path(text_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 統一されたキーワード階層（多段階）
        # 最上位：大カテゴリ → 中カテゴリ（サブカテゴリ） → キーワード
        self.unified_hierarchy = {}

        # 政党テキストデータ
        self.party_texts = {}

        # トークン化済みテキスト
        self.party_tokens = {}

        # 統一後の言及度数（サブカテゴリごと）
        self.unified_mention_counts = defaultdict(lambda: defaultdict(int))

        # 統一後の言及率
        self.unified_mention_rates = defaultdict(lambda: defaultdict(float))

    def _initialize_unified_keyword_hierarchy(self):
        """
        統一されたキーワード階層を初期化

        構造：
        {
            '大カテゴリ': {
                'サブカテゴリ': ['キーワード1', 'キーワード2', ...]
            }
        }

        例：
        {
            'エネルギー・環境': {
                '脱炭素': ['脱炭素', 'CO2', 'カーボンニュートラル', '温暖化', '気候変動'],
                'エネルギー供給': ['エネルギー', '電力', '原発', 'LNG', '火力', '再生可能']
            }
        }
        """
        hierarchy = {
            '農業・食糧': {
                '農業振興': ['農業', '農産物', '農家', '農民', '振興', '輸出'],
                '食料安全保障': ['食糧', '食料', '食料安全保障', '飼料', '担い手'],
                '水産業': ['水産', '漁業', '水産物', '漁場']
            },
            '憲法・政治': {
                '憲法改正': ['憲法', '改正', '統治'],
                '民主主義・政治制度': ['民主主義', '選挙', '投票', '政治', '議会', '制度'],
                '法律・法案': ['法律', '法案', '立法']
            },
            'デジタル・行政': {
                'DX・デジタル化': ['デジタル', 'DX', 'IT', 'ICT'],
                'マイナンバー・行政手続': ['マイナンバー', '行政', '手続', '情報'],
                'AI・通信': ['AI', '通信', 'インターネット', '高速通信']
            },
            '外交・安全保障': {
                '防衛力強化': ['防衛', '防衛力', '国防', '敵基地', '抑止力'],
                '安全保障・国際関係': ['安全保障', '安保', '外交', '国際', '連携'],
                'クライシス対応': ['ミサイル', '台湾', '中国', 'NATO']
            },
            '経済成長': {
                '産業振興・企業投資': ['産業', '製造業', '企業', '投資', '事業', '市場'],
                '経済成長・GDP': ['経済', '成長', 'GDP', '経済成長', '経済政策'],
                '生産性・競争力': ['生産性', '競争力', '活性化', '拡大']
            },
            '教育・子育て': {
                '子育て支援': ['子ども', '子育て', '保育', '給食'],
                '教育・学校': ['教育', '学校', '学生', '学習', '大学', 'ICT教育'],
                '教育費の支援': ['無償', '修学支援', '教職員']
            },
            'エネルギー・環境': {
                '脱炭素・気候変動対策': ['脱炭素', 'CO2', 'カーボンニュートラル', '温暖化', '気候変動', '対策'],
                'エネルギー供給': ['エネルギー', '電力', '電気', '原発', 'LNG', '火力', '再生可能', '太陽光'],
                'エネルギー関連技術': ['CCS', '水素']
            },
            '医療・介護': {
                '医療体制': ['医療', '医者', '病院', '診療', '看護', 'ワクチン'],
                '介護・高齢者対策': ['介護', '高齢者', '高齢化'],
                '医療費・保険': ['医療費', '保険料', '介護費用', '健康']
            },
            '労働': {
                '雇用・就職': ['雇用', '失業', '求人', '就職', '職業'],
                '賃金・待遇': ['賃金', '給与', '待遇', '派遣', '残業'],
                '働き方改革・ダイバーシティ': ['働き方改革', '兼業', 'フリーランス', 'ジェンダー', '労働者']
            },
            '地方創生': {
                '地方活性化・振興': ['地方', '地域', '過疎', '振興', '活性化'],
                '人口対策': ['人口減少', '人口', '格差'],
                'インフラ・都市整備': ['インフラ', '交通', '整備', '都市', 'リモート']
            },
            '社会保障': {
                '年金・保険': ['年金', '社会保険', '保険', '給付'],
                '福祉・生活保護': ['福祉', '生活保護', 'セーフティネット'],
                '社会的支援': ['支援', '安心', '困窮', '貧困', '不安定']
            },
            '税制改革': {
                '所得税・法人税': ['所得税', '法人税'],
                '消費税・その他の税': ['消費税', '税', '税制'],
                '税制改革・税負担': ['税制改革', '減税', '増税', '負担', '資産課税', '不公正']
            }
        }

        self.unified_hierarchy = hierarchy

    def load_party_texts(self) -> Dict[str, str]:
        """修辞除去済みテキストを読み込む"""
        texts = {}

        if not self.text_dir.exists():
            print(f"[ERROR] テキストディレクトリが見つかりません: {self.text_dir}")
            return texts

        print(f"[INFO] テキスト読み込み中: {self.text_dir}")

        for file_path in sorted(self.text_dir.glob('*.txt')):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    party_name = file_path.stem
                    texts[party_name] = text
                    print(f"[OK] {party_name}: {len(text):,}文字")
            except Exception as e:
                print(f"[ERROR] {file_path.name}: {e}")

        return texts

    def tokenize(self, text: str) -> List[str]:
        """形態素解析で単語に分割"""
        tokens = []

        if USE_JANOME:
            # janomeを使用
            for token in tokenizer.tokenize(text):
                word = token.surface
                pos = token.part_of_speech.split(',')[0]

                if pos in {'名詞', '動詞', '形容詞', '副詞'} and len(word) >= 2:
                    tokens.append(word)
        else:
            # 簡易実装
            text = re.sub(r'[a-zA-Z0-9]', '', text)
            text = re.sub(r'[、。，．！？：；「」『』（）【】〈〉]', ' ', text)
            words = text.split()
            tokens = [w for w in words if len(w) >= 2]

        return tokens

    def count_unified_mentions(self, tokens: List[str], keywords: List[str]) -> int:
        """
        統一されたサブカテゴリのキーワード出現回数をカウント（前方一致）

        Args:
            tokens: トークンリスト
            keywords: キーワードリスト

        Returns:
            キーワードの出現回数
        """
        count = 0

        for token in tokens:
            # 各キーワードについて前方一致をチェック
            for keyword in keywords:
                if token.startswith(keyword):
                    count += 1
                    break  # 同じトークンで複数キーワードと一致しないようにする

        return count

    def calculate_unified_mention_rates(self):
        """統一後の言及率を計算"""
        print("\n[INFO] 統一後の言及率を計算中...")

        # テキストをトークン化
        for party_name, text in self.party_texts.items():
            self.party_tokens[party_name] = self.tokenize(text)

        for party_name, tokens in self.party_tokens.items():
            total_words = len(tokens)

            if total_words == 0:
                print(f"   [WARN] {party_name}: トークン数が0です")
                continue

            print(f"   処理中: {party_name} (全単語数: {total_words:,})...", end='', flush=True)

            # 各大カテゴリ → サブカテゴリ → キーワードを処理
            for category, subcategories in self.unified_hierarchy.items():
                for subcategory, keywords in subcategories.items():
                    mention_count = self.count_unified_mentions(tokens, keywords)
                    mention_rate = mention_count / total_words if total_words > 0 else 0

                    # サブカテゴリをキーにして格納
                    subcategory_key = f"{category} > {subcategory}"
                    self.unified_mention_counts[party_name][subcategory_key] = mention_count
                    self.unified_mention_rates[party_name][subcategory_key] = mention_rate

            print(" OK")

    def output_results(self):
        """結果をCSVファイルに出力"""
        print("\n[OUTPUT] 結果を出力中...")

        # ファイル1: 統一後の言及率マトリクス（サブカテゴリ単位）
        output_file_1 = self.output_dir / 'phase1_step4_unified_mention_rates.csv'

        parties = sorted(self.unified_mention_rates.keys())
        # サブカテゴリを大カテゴリごとにソート
        subcategories = sorted(self.unified_mention_rates[parties[0]].keys() if parties else [])

        with open(output_file_1, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['政党'] + subcategories
            writer.writerow(header)

            for party in parties:
                row = [party]
                for subcategory in subcategories:
                    row.append(self.unified_mention_rates[party][subcategory])
                writer.writerow(row)

        print(f"[OK] {output_file_1}")

        # ファイル2: 統一後の言及度数マトリクス（サブカテゴリ単位）
        output_file_2 = self.output_dir / 'phase1_step4_unified_mention_counts.csv'

        with open(output_file_2, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['政党'] + subcategories
            writer.writerow(header)

            for party in parties:
                row = [party]
                for subcategory in subcategories:
                    row.append(self.unified_mention_counts[party][subcategory])
                writer.writerow(row)

        print(f"[OK] {output_file_2}")

        # ファイル3: サマリーレポート
        output_file_3 = self.output_dir / 'phase1_step4_report.txt'

        with open(output_file_3, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Phase 1 Step 4: 複合キーワード統一 - レポート\n")
            f.write("=" * 80 + "\n\n")

            f.write("【統一されたカテゴリ構成】\n")
            for category, subcategories in self.unified_hierarchy.items():
                f.write(f"\n{category}:\n")
                for subcategory, keywords in subcategories.items():
                    f.write(f"  - {subcategory}: {', '.join(keywords[:5])}")
                    if len(keywords) > 5:
                        f.write(f", ... ({len(keywords)}個)")
                    f.write("\n")

            f.write("\n\n【統計情報】\n")
            f.write(f"政党数: {len(parties)}\n")
            f.write(f"サブカテゴリ数: {len(subcategories)}\n")

            # 政党ごとの統計
            f.write("\n【政党ごとの言及率統計】\n")
            for party in parties:
                rates = [self.unified_mention_rates[party][sc] for sc in subcategories]
                rates_nonzero = [r for r in rates if r > 0]

                f.write(f"\n{party}:\n")
                f.write(f"  全単語数: {sum(self.unified_mention_counts[party].values()):,}\n")
                f.write(f"  言及率 (平均): {np.mean(rates):.6f}\n")
                f.write(f"  言及率 (最大): {np.max(rates):.6f}\n")
                if rates_nonzero:
                    f.write(f"  言及率 (ゼロ除外平均): {np.mean(rates_nonzero):.6f}\n")
                f.write(f"  言及ありのサブカテゴリ数: {len(rates_nonzero)}\n")

        print(f"[OK] {output_file_3}")

    def run(self):
        """実行"""
        print("=" * 80)
        print("Phase 1 Step 4: 複合キーワード統一")
        print("=" * 80)

        # ステップ1: キーワード階層を初期化
        self._initialize_unified_keyword_hierarchy()

        # ステップ2: テキストを読み込む
        self.party_texts = self.load_party_texts()
        if not self.party_texts:
            print("[ERROR] テキストが読み込めません")
            return False

        # ステップ3: 統一後の言及率を計算
        self.calculate_unified_mention_rates()

        # ステップ4: 結果を出力
        self.output_results()

        print("\n" + "=" * 80)
        print("[SUCCESS] Phase 1 Step 4 の処理が完了しました")
        print("=" * 80)
        return True


if __name__ == '__main__':
    # ディレクトリ設定
    script_dir = Path(__file__).parent
    text_dir = script_dir / 'data'
    output_dir = script_dir / 'results'

    # 実行
    unifier = KeywordUnifier(text_dir, output_dir)
    unifier.run()
