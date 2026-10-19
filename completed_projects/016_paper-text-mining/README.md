# 論文PDFのテキスト分析

元フォルダ: `2025_1 / 2025_2`

手元の論文PDF群からテキストを抽出し、特徴量化して分析した一連の作業。

- PyPDF2 / pdfplumber でテキスト抽出、NLTK でトークン化・ストップワード除去
- TF-IDF、LDA、ワードクラウド、要約パイプライン（transformers）
- 抽出結果を `Data_note.csv` に集約し、本文文字数・結論文字数・著者数を標準化して相関検定
  （pearson / spearman）まで実施
