# 課題内容
あらすじ文から元となった作品を予測していただきます。
与えられるのは、LLM（大規模言語モデル）によって生成された「架空作品のあらすじ」。 

# データの概要
- ```base_stories.tsv```
    - 本コンペティションで利用している元ネタの作品名とあらすじが記載されています。作品名に対応するラベルが、その作品のラベルとなります。
        - columns
            - 'id', 
            - 'category', 
            - 'title', 
            - 'story'

- ```fiction_stories_practice.tsv```
    - 本コンペティションの練習用データです。このデータの予測結果は提出する必要はございません。本課題を理解するためのデータとしてご利用ください。
        - columns
            - 'id_a'
            - 'id_b'
            - 'title_a'
            - 'title_b'
            - 'story'

- ```fiction_stories_test.tsv```
    - 架空の作品340作品のidと本文が入った評価用データです。こちらの内容をもとに作品を予測していただきます。
        - columns
            - 'id', 
            - 'story'

- ```sample_submit.csv```
    - 投稿ファイルのサンプルです。aとbに予測した映画のid(```base_stories.csv```のラベル)を入れてください。

# 分析方針
1.前処理
2.スコアリング
    1. emmbeding
    2. ベクトル計算
3.予測
4.アウトプット

# 分析コード

```python
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from sudachipy import dictionary, tokenizer

# 1. データの読み込み
base_df = pd.read_csv('base_stories.tsv', sep='\t')
test_df = pd.read_csv('fiction_stories_test.tsv', sep='\t')

# 2. 日本語の分かち書き（BM25用）
tokenizer_obj = dictionary.Dictionary(dict="full").create()
mode = tokenizer.Tokenizer.SplitMode.C

def tokenize_japanese(text):
    if not isinstance(text, str): return []
    # Sudachiで単語に分割して、名詞・動詞・形容詞などの基本形を抽出
    return [m.normalized_form() for m in tokenizer_obj.tokenize(text, mode)]

print("Tokenizing for BM25...")
base_corpus = [tokenize_japanese(s) for s in base_df['story'].fillna("")]
test_queries = [tokenize_japanese(s) for s in test_df['story'].fillna("")]
bm25 = BM25Okapi(base_corpus)

# 3. ベクトル埋め込み（E5モデル）
print("Encoding for Vector Search...")
model = SentenceTransformer('intfloat/multilingual-e5-small')
base_embeddings = model.encode("passage: " + base_df['story'].fillna(""), convert_to_tensor=True)
test_embeddings = model.encode("query: " + test_df['story'].fillna(""), convert_to_tensor=True)

# 4. ハイブリッド・スコアリング
results = []
# 重みの設定（ベクトル：キーワード = 0.7 : 0.3 くらいが一般的）
alpha = 0.7

print("Calculating hybrid scores...")
for i in range(len(test_df)):
    # ベクトルスコア（コサイン類似度）: 0.0 ~ 1.0 に正規化
    vec_scores = util.cos_sim(test_embeddings[i], base_embeddings).cpu().numpy()[0]
    vec_scores = (vec_scores - vec_scores.min()) / (vec_scores.max() - vec_scores.min() + 1e-9)
    
    # BM25スコア: 0.0 ~ 1.0 に正規化
    bm25_scores = np.array(bm25.get_scores(test_queries[i]))
    if bm25_scores.max() > 0:
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
    
    # 重み付き合計スコア
    hybrid_scores = (alpha * vec_scores) + ((1 - alpha) * bm25_scores)
    
    # 上位2件のインデックス取得
    top_indices = np.argsort(hybrid_scores)[::-1][:2]
    
    results.append({
        'id': test_df.iloc[i]['id'],
        'a': base_df.iloc[top_indices[0]]['id'],
        'b': base_df.iloc[top_indices[1]]['id']
    })

# 5. 提出用CSV作成
submit_df = pd.DataFrame(results)
submit_df.to_csv('submission_hybrid.csv', index=False)
print("完了！ submission_hybrid.csv が出力されました。")
```
