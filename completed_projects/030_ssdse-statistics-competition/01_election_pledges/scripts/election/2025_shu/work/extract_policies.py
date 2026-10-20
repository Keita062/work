import os
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
import pdfplumber
import subprocess
import fitz  # PyMuPDF
import sys
from PIL import Image
from io import BytesIO

# エンコーディング設定
os.environ['PYTHONIOENCODING'] = 'utf-8'

# パスの設定
base_path = Path("C:/GitHub/Data-Analysis_competition/SSDSE/01_election_pledges/election/2025_shu")

# データを格納するリスト
data = []

# Tesseractパスの設定（Windows用）
tesseract_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

tesseract_found = False
tesseract_cmd = None
for path in tesseract_paths:
    if os.path.exists(path):
        tesseract_cmd = path
        tesseract_found = True
        print(f"Tesseract found: {path}\n")
        # Tesseract言語データディレクトリをPATHに追加
        tessdata_dir = os.path.dirname(path).replace('\\', '/')
        os.environ['TESSDATA_PREFIX'] = tessdata_dir
        break

if not tesseract_found:
    print("WARNING: Tesseract-OCR not found. PDFs will be processed without OCR.\n")

def extract_text_from_html(file_path):
    """HTMLファイルからテキストを抽出"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            # スクリプトとスタイルタグを削除
            for script in soup(["script", "style"]):
                script.decompose()
            # テキストを取得
            text = soup.get_text()
            # 余分な空白を削除
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
    except Exception as e:
        print(f"HTMLファイル解析エラー {str(file_path)}: {str(e)}")
        return ""

def extract_text_from_pdf(file_path):
    """PDFファイルからテキストを抽出（テキストレイヤー+OCR）"""
    text = ""
    try:
        # まずテキストレイヤーを抽出
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except:
                    pass

        # テキストが十分でない場合、OCRを実行
        if len(text.strip()) < 100 and tesseract_found:
            print(f"  OCR処理中: {Path(file_path).name}")
            try:
                # PyMuPDFでPDFをロードしてページをイメージに変換
                pdf_document = fitz.open(file_path)
                page_count = pdf_document.page_count
                print(f"    ページ数: {page_count}")

                for page_num in range(page_count):
                    try:
                        page = pdf_document[page_num]
                        # ページをイメージにレンダリング（300 DPI）
                        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                        # PIL Imageに変換
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                        # 一時イメージファイルを保存
                        temp_img_path = f"/tmp/temp_page_{page_num}.png"
                        img.save(temp_img_path)

                        # Tesseractコマンドを直接実行
                        try:
                            result = subprocess.run(
                                [tesseract_cmd, temp_img_path, "stdout", "-l", "jpn"],
                                capture_output=True,
                                text=True,
                                timeout=60
                            )
                            page_text = result.stdout
                            if page_text.strip():
                                print(f"    ページ {page_num + 1}: {len(page_text)} 文字")
                                text += page_text + "\n"
                            else:
                                print(f"    ページ {page_num + 1}: テキストなし")
                        except subprocess.TimeoutExpired:
                            print(f"    ページ {page_num + 1}: OCR タイムアウト")
                        finally:
                            # 一時ファイルを削除
                            if os.path.exists(temp_img_path):
                                os.remove(temp_img_path)
                    except Exception as ocr_error:
                        print(f"    ページ {page_num + 1} エラー: {str(ocr_error)}")
                        pass
                pdf_document.close()
            except Exception as convert_error:
                print(f"  PDF処理エラー: {str(convert_error)}")

        return text
    except Exception as e:
        print(f"PDFファイル解析エラー {str(file_path)}: {str(e)}")
        return ""

# 各政党フォルダを走査
for party_folder in sorted(base_path.iterdir()):
    if not party_folder.is_dir() or party_folder.name == 'work':
        continue

    party_name = party_folder.name
    print(f"処理中: {party_name}")

    # フォルダ内のすべてのファイルを走査
    for file_path in sorted(party_folder.rglob('*')):
        # _filesフォルダのファイルは除外
        if '_files' in str(file_path):
            continue

        if file_path.is_file():
            policy_text = ""
            file_ext = file_path.suffix.lower()

            # HTMLファイルの処理
            if file_ext == '.html':
                policy_text = extract_text_from_html(file_path)
                if policy_text.strip():
                    data.append({
                        '政党名': party_name,
                        '政策': policy_text
                    })
                    print(f"  OK: {file_path.name}")

            # PDFファイルの処理
            elif file_ext == '.pdf':
                policy_text = extract_text_from_pdf(file_path)
                if policy_text.strip():
                    data.append({
                        '政党名': party_name,
                        '政策': policy_text
                    })
                    print(f"  OK: {file_path.name}")
                else:
                    print(f"  SKIP (empty): {file_path.name}")

# DataFrameの作成
df = pd.DataFrame(data)

# 結果を表示
print("\n" + "="*50)
print("DataFrameの概要")
print("="*50)
print(f"行数: {len(df)}")
print(f"カラム: {df.columns.tolist()}")
print("\nサンプル:")
print(df.head())

# DataFrameをCSVで保存
output_csv = base_path / "work" / "all_policies.csv"
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\nOK: CSV保存完了: {output_csv}")

# DataFrameをPandasピクルで保存（元のままのデータ構造を保持）
output_pickle = base_path / "work" / "all_policies.pkl"
df.to_pickle(output_pickle)
print(f"OK: Pickle保存完了: {output_pickle}")

print("\n処理完了！")
