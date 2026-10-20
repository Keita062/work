"""
日本語トークナイザー（シンプル版 - 正規表現ベース）
"""
import re
from typing import List, Set

# ドメイン固有ストップワード（政治文書全般で出現頻度が高い汎用語）
POLITICAL_STOPWORDS = {
    '政策', '推進', '実現', '取り組み', '対策', '課題', '実施',
    '国民', '社会', '日本', '政府', '制度', '支援', '向上', '強化',
    '確保', '促進', '整備', '活用', '検討', 'こと', 'もの', 'ため',
    '進める', '図る', '行う', '進む', '進ん', '行っ', 'いく',
    '必要', '重要', '大切', 'これ', 'それ', 'あれ',
    'など', 'ほか', 'その他', 'さらに', 'また', 'および',
    'ため', 'とき', 'ところ', 'ご'
}

# 保持するPOS品詞
KEEP_POS = {
    '名詞-一般',
    '名詞-固有名詞',
    '名詞-サ変接続',
}

class JapaneseTokenizer:
    def __init__(self):
        """シンプルな正規表現ベースのトークナイザーを初期化"""
        pass

    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        テキストを単語に分割（正規表現ベース）

        Args:
            text: 入力テキスト
            remove_stopwords: ストップワードを除去するか

        Returns:
            トークンのリスト
        """
        if not text or len(text) == 0:
            return []

        # 日本語単語と英数字を抽出（ひらがな、カタカナ、漢字、英数字）
        # 連続する日本語文字をトークンとして抽出
        pattern = r'[一-鿿]+|[぀-ゟ]+|[゠-ヿ]+|[a-z]+|[A-Z]+'
        tokens = re.findall(pattern, text.lower())

        # ストップワード除去
        if remove_stopwords:
            tokens = [t for t in tokens if t not in POLITICAL_STOPWORDS and len(t) > 1]
        else:
            tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def tokenize_with_pos(self, text: str) -> List[tuple]:
        """
        品詞タグ付きでトークナイズ（簡略版）

        Args:
            text: 入力テキスト

        Returns:
            (トークン, 品詞) のタプルリスト
        """
        tokens = self.tokenize(text, remove_stopwords=False)
        # 簡略版では品詞情報を付与しない
        return [(t, 'noun') for t in tokens]

    @staticmethod
    def get_statistics(tokens: List[str]) -> dict:
        """トークン統計を計算"""
        if not tokens:
            return {
                'token_count': 0,
                'unique_count': 0,
                'avg_token_length': 0,
                'type_token_ratio': 0
            }

        unique = set(tokens)
        return {
            'token_count': len(tokens),
            'unique_count': len(unique),
            'avg_token_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
            'type_token_ratio': len(unique) / len(tokens) if tokens else 0
        }

    @staticmethod
    def add_stopword(word: str):
        """ストップワードを追加"""
        POLITICAL_STOPWORDS.add(word)

    @staticmethod
    def remove_stopword(word: str):
        """ストップワードを削除"""
        POLITICAL_STOPWORDS.discard(word)

    @staticmethod
    def get_stopwords() -> Set[str]:
        """ストップワード集合を取得"""
        return POLITICAL_STOPWORDS.copy()


def tokenize_documents(texts: List[str]) -> List[List[str]]:
    """
    複数テキストを一括トークナイズ

    Args:
        texts: テキストリスト

    Returns:
        トークンリストのリスト
    """
    tokenizer = JapaneseTokenizer()
    return [tokenizer.tokenize(text) for text in texts]

def token_frequency(tokens: List[str], top_n: int = 30) -> dict:
    """トークン頻度を計算（デバッグ用）"""
    from collections import Counter
    counter = Counter(tokens)
    return dict(counter.most_common(top_n))
