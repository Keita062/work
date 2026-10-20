"""
Phase 1: データ準備・整形
HTMLの解析・テキスト抽出、テキスト正規化、選挙結果データの構造化
"""
import os
import re
from pathlib import Path
from html.parser import HTMLParser
import pandas as pd
from bs4 import BeautifulSoup
import unicodedata

class BaseConfig:
    """基本設定"""
    BASE_DIR = Path("c:/GitHub/Data-Analysis_competition")
    ELECTION_DIR = BASE_DIR / "SSDSE" / "election"
    OUTPUT_DIR = BASE_DIR / "SSDSE" / "05_11" / "data"

    HOUSE_SOURCE = ELECTION_DIR / "2025_shu"
    SENATE_SOURCE = ELECTION_DIR / "2025_san"
    ELECTION_RESULTS_DIR = ELECTION_DIR / "結果"

    HOUSE_OUTPUT = OUTPUT_DIR / "2025_house_policies"
    SENATE_OUTPUT = OUTPUT_DIR / "2025_senate_policies"

def normalize_text(text):
    """テキスト正規化"""
    # HTML エンティティの解析
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&amp;', '&', text)

    # Unicode正規化（NFC形式）
    text = unicodedata.normalize('NFC', text)

    # 改行の正規化
    text = re.sub(r'\r\n|\r|\n', '\n', text)

    # 余分な空白の除去
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    text = '\n'.join(lines)

    # 連続する空白を1つにまとめる
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()

def extract_text_from_html(html_path):
    """HTMLファイルからテキストを抽出"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except UnicodeDecodeError:
        with open(html_path, 'r', encoding='shift_jis') as f:
            html_content = f.read()
    except UnicodeDecodeError:
        with open(html_path, 'r', encoding='cp932') as f:
            html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # scriptタグとstyleタグを除去
    for script in soup(['script', 'style', 'nav', 'footer']):
        script.decompose()

    # テキストを抽出
    text = soup.get_text()

    # テキスト正規化
    text = normalize_text(text)

    return text

def process_house_policies():
    """衆議院公約データを処理"""
    print("衆議院公約データを処理中...")
    BaseConfig.HOUSE_OUTPUT.mkdir(parents=True, exist_ok=True)

    for party_dir in BaseConfig.HOUSE_SOURCE.iterdir():
        if not party_dir.is_dir() or party_dir.name == 'work':
            continue

        party_name = party_dir.name
        html_files = list(party_dir.glob('*.html'))

        if not html_files:
            print(f"  警告: {party_name} にHTMLファイルが見つかりません")
            continue

        # 複数のHTMLファイルがある場合は結合
        combined_text = []
        for html_file in sorted(html_files):
            text = extract_text_from_html(html_file)
            combined_text.append(text)

        # テキストファイルに出力
        output_file = BaseConfig.HOUSE_OUTPUT / f"{party_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(combined_text))

        print(f"  OK {party_name}: {output_file.name}")

def process_senate_policies():
    """参議院公約データを処理"""
    print("参議院公約データを処理中...")
    BaseConfig.SENATE_OUTPUT.mkdir(parents=True, exist_ok=True)

    for party_dir in BaseConfig.SENATE_SOURCE.iterdir():
        if not party_dir.is_dir():
            continue

        party_name = party_dir.name
        html_files = list(party_dir.glob('*.html'))

        if not html_files:
            print(f"  警告: {party_name} にHTMLファイルが見つかりません")
            continue

        # 複数のHTMLファイルがある場合は結合
        combined_text = []
        for html_file in sorted(html_files):
            text = extract_text_from_html(html_file)
            combined_text.append(text)

        # テキストファイルに出力
        output_file = BaseConfig.SENATE_OUTPUT / f"{party_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(combined_text))

        print(f"  OK {party_name}: {output_file.name}")

def parse_election_results():
    """選挙結果データをパース"""
    results = []

    # 衆議院: 当選者数と立候補者数をパース
    house_elected_path = BaseConfig.ELECTION_RESULTS_DIR / "衆議院 当選者数.txt"
    house_candidates_path = BaseConfig.ELECTION_RESULTS_DIR / "衆議院 立候補者数.txt"

    house_elected = {}
    with open(house_elected_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4 and parts[0] not in ['党派']:
                party = parts[0]
                try:
                    elected = int(parts[3])  # 「当選」欄
                    house_elected[party] = elected
                except ValueError:
                    pass

    house_candidates = {}
    with open(house_candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5 and parts[0] not in ['党派']:
                party = parts[0]
                try:
                    candidates = int(parts[4])  # 「合計」欄
                    house_candidates[party] = candidates
                except ValueError:
                    pass

    # 衆議院結果を保存
    for party, elected in house_elected.items():
        candidates = house_candidates.get(party, 0)
        election_rate = elected / candidates if candidates > 0 else 0
        results.append({
            'election_type': 'house',
            'party': party,
            'candidates': candidates,
            'elected': elected,
            'election_rate': round(election_rate, 4)
        })

    # 参議院: 当選者数と立候補者数をパース
    senate_elected_path = BaseConfig.ELECTION_RESULTS_DIR / "参議院選挙 当選者数.txt"
    senate_candidates_path = BaseConfig.ELECTION_RESULTS_DIR / "参議院選挙 立候補者数.txt"

    senate_elected = {}
    with open(senate_elected_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4 and parts[0] not in ['党派']:
                party = parts[0]
                try:
                    elected = int(parts[3])  # 「今回当選」欄
                    senate_elected[party] = elected
                except (ValueError, IndexError):
                    pass

    senate_candidates = {}
    with open(senate_candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5 and parts[0] not in ['党派']:
                party = parts[0]
                try:
                    candidates = int(parts[4])  # 合計欄
                    senate_candidates[party] = candidates
                except ValueError:
                    pass

    # 参議院結果を保存
    for party, elected in senate_elected.items():
        candidates = senate_candidates.get(party, 0)
        election_rate = elected / candidates if candidates > 0 else 0
        results.append({
            'election_type': 'senate',
            'party': party,
            'candidates': candidates,
            'elected': elected,
            'election_rate': round(election_rate, 4)
        })

    return results

def save_election_results(results):
    """選挙結果をCSVで保存"""
    print("選挙結果データを構造化中...")

    df = pd.DataFrame(results)

    output_path = BaseConfig.OUTPUT_DIR / "election_results_normalized.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"  OK election_results_normalized.csv: {len(df)}件")
    print(f"    - 衆議院: {len(df[df['election_type'] == 'house'])} 政党")
    print(f"    - 参議院: {len(df[df['election_type'] == 'senate'])} 政党")

def main():
    """メイン処理"""
    print("\n===== Phase 1: データ準備・整形 =====\n")

    # 出力ディレクトリを作成
    BaseConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 衆議院公約データを処理
    process_house_policies()
    print()

    # 参議院公約データを処理
    process_senate_policies()
    print()

    # 選挙結果データを処理
    results = parse_election_results()
    save_election_results(results)
    print()

    print("===== Phase 1 完了 =====\n")

if __name__ == "__main__":
    main()