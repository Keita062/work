#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final NLP analysis with ASCII output"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

# Path setup
base_path = r'C:\GitHub\Data-Analysis_competition\SSDSE'
election_path = os.path.join(base_path, 'election')
data_path = os.path.join(base_path, 'data')
src_path = os.path.join(election_path, 'src')
results_path = os.path.join(src_path, 'results')

# Load modules
import importlib.util
tc_spec = importlib.util.spec_from_file_location("text_cleaner", os.path.join(src_path, 'text_cleaner.py'))
text_cleaner_mod = importlib.util.module_from_spec(tc_spec)
tc_spec.loader.exec_module(text_cleaner_mod)

tok_spec = importlib.util.spec_from_file_location("tokenizer", os.path.join(src_path, 'tokenizer.py'))
tokenizer_mod = importlib.util.module_from_spec(tok_spec)
tok_spec.loader.exec_module(tokenizer_mod)

clean_manifesto_text = text_cleaner_mod.clean_manifesto_text
split_into_paragraphs = text_cleaner_mod.split_into_paragraphs
normalize_party_name = text_cleaner_mod.normalize_party_name
quality_report = text_cleaner_mod.quality_report
JapaneseTokenizer = tokenizer_mod.JapaneseTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px

matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("ELECTION MANIFESTO NLP ANALYSIS - PHASES 1-5")
print("=" * 80)

# Election results
seats_shu = {
    '自民党': 316, '中道': 49, '日本維新の党': 36, '国民民主党': 28,
    '参政党': 15, 'チームみらい': 11, '共産党': 4, '令和新選組': 1,
    '日本保守党': 0, '社民党': 0,
}

seats_san = {
    '自民党': 39, '立憲民主党': 22, '国民民主党': 17, '参政党': 14,
    '公明党': 8, '日本維新の党': 7, '令和新選組': 3, '共産党': 3,
    '日本保守党': 2, '社民党': 1, 'チームみらい': 1,
}

print("\n[PHASE 1] Loading and preprocessing manifesto data...")

def load_and_clean_data():
    data_records = []
    shu_files = glob.glob(os.path.join(data_path, '衆議院', '*.txt'))
    for file_path in shu_files:
        party_name = Path(file_path).stem
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                raw_text = f.read()
        party_name = normalize_party_name(party_name)
        cleaned_text = clean_manifesto_text(raw_text, party_name, '衆議院')
        data_records.append({
            'party': party_name, 'chamber': '衆議院', 'raw_text': raw_text,
            'cleaned_text': cleaned_text, 'file_path': file_path
        })

    san_files = glob.glob(os.path.join(data_path, '参議院', '*.txt'))
    for file_path in san_files:
        party_name = Path(file_path).stem
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                raw_text = f.read()
        party_name = normalize_party_name(party_name)
        cleaned_text = clean_manifesto_text(raw_text, party_name, '参議院')
        data_records.append({
            'party': party_name, 'chamber': '参議院', 'raw_text': raw_text,
            'cleaned_text': cleaned_text, 'file_path': file_path
        })
    return pd.DataFrame(data_records)

df_documents = load_and_clean_data()
print(f"  Loaded {len(df_documents)} documents")

paragraph_records = []
for idx, row in df_documents.iterrows():
    paragraphs = split_into_paragraphs(row['cleaned_text'], min_length=30)
    for p_idx, para in enumerate(paragraphs):
        paragraph_records.append({
            'party': row['party'], 'chamber': row['chamber'],
            'para_idx': p_idx, 'text': para
        })

df_paragraphs = pd.DataFrame(paragraph_records)
print(f"  Split into {len(df_paragraphs)} paragraphs")

print("\n[PHASE 2] Japanese tokenization...")
tokenizer = JapaneseTokenizer()
df_paragraphs['tokens'] = df_paragraphs['text'].apply(lambda x: tokenizer.tokenize(x))
df_paragraphs['token_count'] = df_paragraphs['tokens'].apply(len)
df_paragraphs['tokenized_text'] = df_paragraphs['tokens'].apply(lambda x: ' '.join(x))
print(f"  Tokenized {len(df_paragraphs)} paragraphs")

print("\n[PHASE 3] TF-IDF + NMF topic modeling...")
party_documents = df_documents.copy()
party_documents['tokenized_text'] = party_documents['cleaned_text'].apply(
    lambda x: ' '.join(tokenizer.tokenize(x))
)
party_documents['token_count'] = party_documents['tokenized_text'].apply(lambda x: len(x.split()))

tfidf = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf_party = tfidf.fit_transform(party_documents['tokenized_text'])
print(f"  TF-IDF matrix: {X_tfidf_party.shape}")

n_topics = 12
nmf = NMF(n_components=n_topics, init='nndsvda', random_state=42, max_iter=500)
W = nmf.fit_transform(X_tfidf_party)
H = nmf.components_
print(f"  NMF fitted with {n_topics} topics")

vocab = tfidf.get_feature_names_out()

topic_labels = {
    0: 'Economy-Growth', 1: 'Social Security', 2: 'Education-Childcare',
    3: 'Energy-Environment', 4: 'Diplomacy-Security', 5: 'Healthcare',
    6: 'Tax Reform', 7: 'Digital-Admin', 8: 'Regional Development',
    9: 'Labor', 10: 'Agriculture-Food', 11: 'Constitution-Politics',
}

party_seats = {}
for party in party_documents['party'].unique():
    seats = seats_shu.get(party, 0) + seats_san.get(party, 0)
    party_seats[party] = seats

seats_vector = np.array([party_seats.get(party, 0) for party in party_documents['party']])
total_seats = seats_vector.sum()

salience_weighted = (W * seats_vector.reshape(-1, 1)).sum(axis=0) / total_seats
salience_count = (W >= 0.1).sum(axis=0)

topic_scores = pd.DataFrame({
    'topic_id': range(n_topics),
    'topic_label': [topic_labels[i] for i in range(n_topics)],
    'salience_weighted': salience_weighted,
    'party_count': salience_count,
})
topic_scores = topic_scores.sort_values('salience_weighted', ascending=False)

party_topic_matrix = pd.DataFrame(
    W, index=party_documents['party'].values,
    columns=[topic_labels[i] for i in range(n_topics)]
)

topic_scores.to_csv(os.path.join(results_path, 'ranked_topics.csv'), index=False, encoding='utf-8')
party_topic_matrix.to_csv(os.path.join(results_path, 'party_topic_matrix.csv'), encoding='utf-8')
print(f"  Saved ranked_topics.csv and party_topic_matrix.csv")

print("\n[PHASE 4] Sentence embeddings and clustering...")
print("  Loading sentence-transformer model...")
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

print("  Embedding paragraphs...")
embeddings = embedding_model.encode(
    df_paragraphs['text'].tolist(), show_progress_bar=True, batch_size=32
)
print(f"  Embeddings shape: {embeddings.shape}")

np.save(os.path.join(results_path, 'paragraph_embeddings.npy'), embeddings)

print("  Applying PCA...")
# n_components は サンプル数と特徴数の最小値以下にする
n_pca_components = min(50, embeddings.shape[0] - 1)
pca = PCA(n_components=n_pca_components, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

print("  K-Means clustering...")
# n_clusters は サンプル数以下にする
n_clusters_actual = min(n_topics, embeddings.shape[0])
kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings_pca)
df_paragraphs['cluster'] = cluster_labels

silhouette = silhouette_score(embeddings_pca, cluster_labels)
print(f"  Silhouette Score: {silhouette:.3f}")

party_embeddings_dict = {}
for party in df_paragraphs['party'].unique():
    party_mask = df_paragraphs['party'] == party
    party_emb = embeddings[party_mask].mean(axis=0)
    party_embeddings_dict[party] = party_emb

party_list = sorted(party_embeddings_dict.keys())
party_embeddings_array = np.array([party_embeddings_dict[p] for p in party_list])

df_paragraphs_export = df_paragraphs[['party', 'chamber', 'text', 'token_count', 'cluster']].copy()
df_paragraphs_export['cluster_label'] = df_paragraphs_export['cluster'].map(
    {i: topic_labels[i] for i in range(n_topics)}
)
df_paragraphs_export.to_csv(os.path.join(results_path, 'paragraph_clusters.csv'), index=False, encoding='utf-8')

print("\n[PHASE 5] UMAP and visualization...")
print("  Applying UMAP...")
reducer = umap.UMAP(n_neighbors=10, min_dist=0.15, metric='cosine', random_state=42)
umap_para = reducer.fit_transform(embeddings)
umap_parties = reducer.transform(party_embeddings_array)
print("  UMAP complete")

df_paragraphs['umap_x'] = umap_para[:, 0]
df_paragraphs['umap_y'] = umap_para[:, 1]

ideology_map = {
    '共産党': 'left', '立憲民主党': 'left', '社民党': 'left', '令和新選組': 'left',
    '公明党': 'center', '国民民主党': 'center', 'チームみらい': 'center',
    '自民党': 'right', '日本維新の党': 'right', '参政党': 'right',
    '日本保守党': 'right', '中道': 'center',
}
ideology_colors = {'left': '#E74C3C', 'center': '#9B59B6', 'right': '#3498DB'}

print("  Creating Plot A (Party Policy Map)...")
party_info = []
for party, umap_coords in zip(party_list, umap_parties):
    total_seats = party_seats.get(party, 0)
    ideology = ideology_map.get(party, 'center')
    party_info.append({
        'party': party, 'umap_x': umap_coords[0], 'umap_y': umap_coords[1],
        'seats': max(total_seats, 1), 'ideology': ideology,
    })

df_parties = pd.DataFrame(party_info)
fig_a = px.scatter(
    df_parties, x='umap_x', y='umap_y', size='seats', color='ideology',
    hover_name='party', color_discrete_map=ideology_colors,
    size_max=60, title='Party Policy Map (Bubble size = Seats)',
    labels={'umap_x': 'Policy Axis 1', 'umap_y': 'Policy Axis 2'}, height=700,
)
fig_a.add_scatter(x=df_parties['umap_x'], y=df_parties['umap_y'], text=df_parties['party'],
                 mode='text', textposition='top center', textfont=dict(size=9),
                 showlegend=False, hoverinfo='skip')
fig_a.update_layout(font=dict(family='Arial', size=12), template='plotly_white')
fig_a.write_html(os.path.join(results_path, 'plot_A_party_map.html'))

print("  Creating Plot B (Topic Landscape)...")
df_plot_b = df_paragraphs.copy()
df_plot_b['cluster_label'] = df_plot_b['cluster'].map({i: topic_labels[i] for i in range(n_topics)})
fig_b = px.scatter(
    df_plot_b, x='umap_x', y='umap_y', color='cluster_label',
    hover_data={'umap_x': False, 'umap_y': False, 'party': True, 'chamber': True},
    opacity=0.5, title='Policy Topic Landscape (Paragraph Level)',
    labels={'umap_x': 'Policy Axis 1', 'umap_y': 'Policy Axis 2', 'cluster_label': 'Topic'}, height=700,
)
fig_b.update_layout(font=dict(family='Arial', size=10), template='plotly_white', hovermode='closest')
fig_b.write_html(os.path.join(results_path, 'plot_B_topic_landscape.html'))

print("  Creating Plot C (Topic-Party Heatmap)...")
party_order = [
    '共産党', '立憲民主党', '社民党', '令和新選組',
    '公明党', '国民民主党', 'チームみらい',
    '自民党', '中道', '日本維新の党', '参政党', '日本保守党',
]
party_topic_ordered = party_topic_matrix.loc[[p for p in party_order if p in party_topic_matrix.index]]

plt.figure(figsize=(14, 8))
sns.heatmap(party_topic_ordered, cmap='RdYlBu_r', annot=True, fmt='.2f',
            cbar_kws={'label': 'NMF Weight'}, linewidths=0.5)
plt.title('Topic-Party Matrix (NMF weights)', fontsize=14, fontweight='bold')
plt.xlabel('Topic', fontsize=12)
plt.ylabel('Party', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'plot_C_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print("  Creating Plot D (Cross-Party Importance Ranking)...")
plot_d_data = topic_scores.sort_values('salience_weighted', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(plot_d_data['topic_label'], plot_d_data['salience_weighted'], color='#3498DB')
for idx, (topic, score, party_count) in enumerate(zip(
    plot_d_data['topic_label'], plot_d_data['salience_weighted'], plot_d_data['party_count']
)):
    plt.text(score + 0.005, idx, f'{party_count}/19', va='center', fontsize=9)
plt.xlabel('Seat-weighted Score', fontsize=12)
plt.title('Cross-Party Importance Ranking', fontsize=14, fontweight='bold')
plt.xlim(0, plot_d_data['salience_weighted'].max() * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(results_path, 'plot_D_ranking.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - SUMMARY")
print("=" * 80)

print("\nTop 5 Cross-Party Important Topics:")
for idx, row in topic_scores.head(5).iterrows():
    print(f"  {row['topic_label']:25} Score: {row['salience_weighted']:.3f}  Parties: {int(row['party_count'])}/19")

print("\nStatistics:")
print(f"  Parties analyzed: {df_paragraphs['party'].nunique()}")
print(f"  Total paragraphs: {len(df_paragraphs)}")
print(f"  Topics extracted: {n_topics}")
print(f"  K-Means Silhouette Score: {silhouette:.3f}")

print("\nOutput files:")
output_files = [
    'ranked_topics.csv',
    'party_topic_matrix.csv',
    'paragraph_clusters.csv',
    'paragraph_embeddings.npy',
    'plot_A_party_map.html',
    'plot_B_topic_landscape.html',
    'plot_C_heatmap.png',
    'plot_D_ranking.png',
]

for fname in output_files:
    fpath = os.path.join(results_path, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath) / 1024
        print(f"  [OK] {fname:40} ({size:8.1f} KB)")
    else:
        print(f"  [NG] {fname:40}")

print("\n" + "=" * 80)
print("Output saved to: " + results_path)
print("=" * 80)
