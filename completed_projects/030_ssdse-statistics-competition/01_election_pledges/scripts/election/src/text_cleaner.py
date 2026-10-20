"""
ボイラープレート除去とテキスト前処理
"""
import re
from typing import Tuple

PARTY_CANONICAL = {
    '国民民主': '国民民主党',
    'れいわ': '令和新選組',
    'みら': 'チームみらい',
    'チみ': 'チームみらい',
}

def identify_source_type(text: str) -> str:
    """
    テキストのソースタイプを推定
    - 'nikkei': 日経記者が作成した要約
    - 'party_site': 党公式サイト
    - 'pdf': PDF抽出
    """
    if '日本経済新聞' in text or 'NIKKEI' in text or '日経' in text:
        return 'nikkei'
    if '---PAGE BREAK---' in text:
        return 'pdf'
    return 'party_site'

def clean_nikkei(text: str) -> str:
    """Nikkei系テキスト（公明党, 立憲民主党など）のボイラープレート除去"""
    # 政策本文の開始マーカーを探す
    match = re.search(r'(【|の主要公約は次の通り|の公約は次の通り)', text)
    if match:
        text = text[match.start():]

    # フッター（日本経済新聞社について）を削除
    footer_match = re.search(r'日本経済新聞社について|Nikkei Inc\.|No reproduction', text)
    if footer_match:
        text = text[:footer_match.start()]

    return text.strip()

def clean_party_site(text: str) -> str:
    """党公式サイト系（自民党, チームみらいなど）のボイラープレート除去"""
    lines = text.splitlines()
    cleaned = []

    nav_items = {
        '別ウィンドウリンク', 'X', 'Facebook', 'Line', 'youtube', 'Instagram', 'TikTok',
        'メニュー閉じる', 'シェアする', '閉じる', 'メニュー開く',
        'SNS', 'Share', 'HOME', 'ホーム'
    }

    for line in lines:
        line_stripped = line.strip()

        # 空行はスキップ
        if not line_stripped:
            continue

        # ナビゲーションアイテムをスキップ
        if line_stripped in nav_items or '別ウィンドウリンク' in line_stripped:
            continue

        # 1文字または非常に短い行（日本語ナビ）をスキップ
        if len(line_stripped) < 15 and re.match(r'^[ぁ-ん一-鿿]+$', line_stripped):
            continue

        cleaned.append(line_stripped)

    return '\n'.join(cleaned).strip()

def clean_pdf(text: str) -> str:
    """PDF抽出テキストのページブレーク除去"""
    text = re.sub(r'---PAGE BREAK---\n?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def clean_manifesto_text(text: str, party: str = '', chamber: str = '') -> str:
    """
    マニフェストテキストを整形

    Args:
        text: 生テキスト
        party: 政党名
        chamber: '衆議院' or '参議院'

    Returns:
        整形済みテキスト
    """
    if not text:
        return ''

    source_type = identify_source_type(text)

    if source_type == 'nikkei':
        text = clean_nikkei(text)
    elif source_type == 'pdf':
        text = clean_pdf(text)
    else:
        text = clean_party_site(text)

    # 複数の空行を2行に統一
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def split_into_paragraphs(text: str, min_length: int = 30) -> list:
    """
    テキストをパラグラフに分割

    Args:
        text: 入力テキスト
        min_length: 最小文字数

    Returns:
        パラグラフのリスト
    """
    paragraphs = text.split('\n\n')
    # 30文字以上で、適切なテキストのみを保持
    filtered = [
        p.strip() for p in paragraphs
        if len(p.strip()) >= min_length and not p.strip().startswith('http')
    ]
    return filtered

def normalize_party_name(party_name: str) -> str:
    """政党名を正規化"""
    party_name = party_name.strip()
    for short, canonical in PARTY_CANONICAL.items():
        if party_name == short or party_name.startswith(short):
            return canonical
    return party_name

def quality_report(df_paragraphs):
    """データ品質レポートを生成"""
    print("\n" + "="*60)
    print("データ品質レポート")
    print("="*60)

    print(f"\n総パラグラフ数: {len(df_paragraphs)}")
    print(f"政党数: {df_paragraphs['party'].nunique()}")
    print(f"議院別: 衆議院 {(df_paragraphs['chamber']=='衆議院').sum()}, 参議院 {(df_paragraphs['chamber']=='参議院').sum()}")

    print("\n政党別パラグラフ数:")
    party_counts = df_paragraphs.groupby('party').size().sort_values(ascending=False)
    for party, count in party_counts.items():
        print(f"  {party}: {count}")

    print("\n政党別トークン数（概算）:")
    df_paragraphs['token_count'] = df_paragraphs['text'].str.split().apply(len)
    token_by_party = df_paragraphs.groupby('party')['token_count'].sum().sort_values(ascending=False)
    for party, tokens in token_by_party.items():
        print(f"  {party}: {tokens:,}")

    # 警告
    if (df_paragraphs['token_count'] == 0).any():
        print("\n⚠️  警告: 空のパラグラフが存在します")

    empty_parties = df_paragraphs.groupby('party').apply(lambda x: (x['token_count'] == 0).all())
    if empty_parties.any():
        empty_list = empty_parties[empty_parties].index.tolist()
        print(f"⚠️  警告: 完全に空の政党: {empty_list}")
