#!/usr/bin/env python3
"""
Phase 1 Step 1: 全政党共通スローガンの除去

目的：全政党の80%以上で出現するキーワード（政治的修辞）を除去し、
      実質的な政策内容のみを抽出する

入力：衆議院・参議院の政党ごと公約テキスト
出力：修辞除去後のテキスト（data/フォルダに格納）
"""

import os
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Set, Tuple, List

# データセットパス
ELECTION_BASE = Path('/home/user/Data-Analysis_competition/SSDSE/election')
OUTPUT_BASE = Path('/home/user/Data-Analysis_competition/SSDSE/04/28/data')

# janomeライブラリを試す
try:
    from janome.tokenizer import Tokenizer
    tokenizer = Tokenizer()
    USE_JANOME = True
except ImportError:
    print("⚠️ janomeが見つかりません。簡易形態素解析を使用します。")
    USE_JANOME = False

class RhetoricRemover:
    """政治的修辞（スローガン）除去クラス"""

    def __init__(self, threshold: float = 0.8):
        """
        Args:
            threshold: キーワード除去の閾値（デフォルト0.8 = 80%）
        """
        self.threshold = threshold
        self.universal_terms = set()
        self.term_appearances = Counter()
        self.term_frequency = Counter()
        self.parties = {}
        self.pos_filter = {'名詞', '動詞', '形容詞', '副詞'}

    def load_party_texts(self) -> Dict[str, str]:
        """
        衆議院・参議院の全政党テキストを読み込む

        Returns:
            {政党名: テキスト内容}
        """
        all_texts = {}

        # 衆議院テキスト
        shu_path = ELECTION_BASE / '2025_shu'
        if shu_path.exists():
            for party_dir in shu_path.iterdir():
                if party_dir.is_dir():
                    party_name = party_dir.name
                    text = self._extract_text_from_party_dir(party_dir)
                    if text.strip():
                        all_texts[f'{party_name}（衆）'] = text

        # 参議院テキスト
        san_path = ELECTION_BASE / '2025_san'
        if san_path.exists():
            for party_dir in san_path.iterdir():
                if party_dir.is_dir():
                    party_name = party_dir.name
                    text = self._extract_text_from_party_dir(party_dir)
                    if text.strip():
                        all_texts[f'{party_name}（参）'] = text

        return all_texts

    def _extract_text_from_party_dir(self, party_dir: Path) -> str:
        """
        政党フォルダからテキストを抽出
        HTMLファイル、テキストファイル、PDFテキストに対応
        """
        texts = []

        for file_path in party_dir.rglob('*'):
            if file_path.suffix in ['.txt']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                except Exception as e:
                    print(f"⚠️ {file_path} の読み込みエラー: {e}")
            elif file_path.suffix == '.html':
                # HTMLから簡易的にテキスト抽出
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html = f.read()
                        # スクリプト・スタイル削除
                        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
                        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
                        # タグ削除
                        text = re.sub(r'<[^>]+>', '\n', html)
                        # エンティティ削除
                        text = re.sub(r'&[a-z]+;', ' ', text)
                        texts.append(text)
                except Exception as e:
                    print(f"⚠️ {file_path} の解析エラー: {e}")

        return '\n\n'.join(texts)

    def tokenize(self, text: str) -> List[str]:
        """
        形態素解析で単語に分割
        名詞・動詞・形容詞・副詞のみを抽出
        """
        tokens = []

        if USE_JANOME:
            # janomeを使用
            for token in tokenizer.tokenize(text):
                word = token.surface
                pos = token.part_of_speech.split(',')[0]

                if pos in self.pos_filter and len(word) >= 2:
                    tokens.append(word)
        else:
            # 簡易実装：スペースと句読点で分割
            # 日本語のみを抽出（ひらがな、カタカナ、漢字）
            text = re.sub(r'[a-zA-Z0-9]', '', text)  # 英数字削除
            text = re.sub(r'[、。，．！？：；「」『』（）【】〈〉]', ' ', text)  # 句読点→スペース
            words = text.split()

            # 2文字以上のみを抽出
            tokens = [w for w in words if len(w) >= 2]

        return tokens

    def detect_universal_terms(self, texts: Dict[str, str]) -> Set[str]:
        """
        全政党の80%以上で出現するキーワードを検出

        Args:
            texts: {政党名: テキスト}

        Returns:
            除去対象キーワードセット
        """
        total_parties = len(texts)
        appearance_threshold = total_parties * self.threshold

        print(f"\n📊 ユニバーサルキーワード検出処理")
        print(f"   対象政党数: {total_parties}")
        print(f"   出現閾値: {appearance_threshold:.1f} (80%以上)")

        # 各政党でのキーワード出現を追跡
        for party_name, text in texts.items():
            print(f"   処理中: {party_name}...", end='', flush=True)

            tokens = self.tokenize(text)
            unique_tokens = set(tokens)

            # この政党で出現した単語をカウント
            for token in unique_tokens:
                self.term_appearances[token] += 1
                self.term_frequency[token] += tokens.count(token)

            print(" OK")

        # 80%以上に出現するキーワード
        universal_terms = {
            term for term, count in self.term_appearances.items()
            if count >= appearance_threshold
        }

        return universal_terms

    def remove_rhetoric(self, text: str, universal_terms: Set[str]) -> str:
        """
        テキストから修辞キーワードを除去

        Args:
            text: 元のテキスト
            universal_terms: 除去対象キーワード

        Returns:
            修辞除去後のテキスト
        """
        tokens = self.tokenize(text)

        # 除去対象ではないトークンのみを保持
        filtered_tokens = [
            token for token in tokens
            if token not in universal_terms
        ]

        # トークンを復結（スペース区切り）
        return ' '.join(filtered_tokens)

    def process(self):
        """メイン処理"""
        print("=" * 70)
        print("Phase 1 Step 1: 全政党共通スローガン除去")
        print("=" * 70)

        # Step 1: テキスト読み込み
        print("\n📂 テキスト読み込み中...")
        texts = self.load_party_texts()
        print(f"   読み込み完了: {len(texts)}政党/議院")

        for party in sorted(texts.keys()):
            text_len = len(texts[party])
            print(f"   - {party}: {text_len:,}文字")

        # Step 2: ユニバーサルキーワード検出
        self.universal_terms = self.detect_universal_terms(texts)

        print(f"\n✂️ 除去対象キーワード: {len(self.universal_terms)}個")
        if self.universal_terms:
            # 上位20個を表示
            top_terms = sorted(
                self.universal_terms,
                key=lambda t: self.term_frequency[t],
                reverse=True
            )[:20]
            print(f"   上位20個: {', '.join(top_terms)}")

        # Step 3: 修辞除去
        print(f"\n🔄 修辞除去処理中...")
        cleaned_texts = {}

        for party_name, text in texts.items():
            print(f"   処理中: {party_name}...", end='', flush=True)
            cleaned = self.remove_rhetoric(text, self.universal_terms)
            cleaned_texts[party_name] = cleaned

            before_len = len(text)
            after_len = len(cleaned)
            reduction = (before_len - after_len) / before_len * 100
            print(f" OK ({reduction:.1f}%削減)")

        # Step 4: 出力
        print(f"\n💾 修辞除去後テキスト出力...")
        output_dir = Path('/home/user/Data-Analysis_competition/SSDSE/04/28/data')
        output_dir.mkdir(parents=True, exist_ok=True)

        for party_name, text in cleaned_texts.items():
            # ファイル名作成（括弧を削除）
            filename = party_name.replace('（', '_').replace('）', '') + '.txt'
            filepath = output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"   出力: {filename}")

        # Step 5: レポート生成
        self._generate_report(texts, cleaned_texts)

    def _generate_report(self, original: Dict[str, str], cleaned: Dict[str, str]):
        """処理結果レポート生成"""
        print("\n" + "=" * 70)
        print("処理結果レポート")
        print("=" * 70)

        print(f"\n【除去対象キーワード統計】")
        print(f"  総数: {len(self.universal_terms)}個")

        if self.universal_terms:
            print(f"\n  出現頻度 Top 30:")
            top_30 = sorted(
                self.universal_terms,
                key=lambda t: self.term_frequency[t],
                reverse=True
            )[:30]

            for i, term in enumerate(top_30, 1):
                freq = self.term_frequency[term]
                print(f"    {i:2d}. '{term}': {freq:,}回")

        print(f"\n【政党別処理結果】")
        print(f"{'政党':<20} {'修辞前(字)':<12} {'修辞後(字)':<12} {'削減率':<8} {'削減率(%)':<8}")
        print("-" * 70)

        total_before = 0
        total_after = 0

        for party_name in sorted(original.keys()):
            before = len(original[party_name])
            after = len(cleaned[party_name])
            reduction = before - after
            reduction_pct = (reduction / before * 100) if before > 0 else 0

            total_before += before
            total_after += after

            print(f"{party_name:<20} {before:<12,} {after:<12,} {reduction:<8,} {reduction_pct:<8.1f}")

        print("-" * 70)
        total_reduction = total_before - total_after
        total_pct = (total_reduction / total_before * 100) if total_before > 0 else 0
        print(f"{'合計':<20} {total_before:<12,} {total_after:<12,} {total_reduction:<8,} {total_pct:<8.1f}")

        print(f"\n✅ 処理完了")
        print(f"   出力先: {Path('/home/user/Data-Analysis_competition/SSDSE/04/28/data')}")


def main():
    remover = RhetoricRemover(threshold=0.8)
    remover.process()


if __name__ == '__main__':
    main()
