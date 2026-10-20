"""
Research Question Analysis: 最も取り組むべき課題の特定
Combines manifesto content with election results to identify key policy issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pickle
import warnings
warnings.filterwarnings('ignore')

# Party mapping and ideology classification
PARTY_IDEOLOGY = {
    '自民': ('自民党', 'right'),
    '中道': ('中道', 'center'),
    '参政': ('参政党', 'right'),
    '国民': ('国民民主党', 'center'),
    'チーム': ('チームみらい', 'center'),
    'れい': ('令和新選組', 'left'),
    '社民': ('社民党', 'left'),
    '保守': ('日本保守党', 'right'),
    '立憲': ('立憲民主党', 'left'),
    '公明': ('公明党', 'center'),
    '共産': ('共産党', 'left'),
    '令和': ('令和新選組', 'left'),
    '維新': ('日本維新の党', 'right'),
    'みんな': ('みんなで作る党', 'center'),
}

def parse_party_name(name):
    """Parse party name from TSV and return canonical name and ideology."""
    for key, (canonical, ideology) in PARTY_IDEOLOGY.items():
        if key in name:
            return canonical, ideology
    # Fallback
    return name, 'unknown'


def load_manifestos():
    """Load manifesto texts from SSDSE/data/."""
    manifestos = []

    for chamber in ['衆議院', '参議院']:
        data_dir = Path('SSDSE/01_election_pledges') / 'data' / chamber
        if not data_dir.exists():
            print(f"Warning: {data_dir} not found")
            continue

        for txt_file in sorted(data_dir.glob('*.txt')):
            party_name = txt_file.stem
            text = txt_file.read_text(encoding='utf-8')

            if not text.strip():
                print(f"Skipping empty: {chamber}/{party_name}")
                continue

            manifestos.append({
                'party_file': party_name,
                'chamber': chamber,
                'text': text,
                'text_length': len(text)
            })

    df = pd.DataFrame(manifestos)

    # Parse party names to canonical form
    df['party'], df['ideology'] = zip(*df['party_file'].apply(
        lambda x: parse_party_name(x)
    ))

    print(f"\n[Data] Loaded {len(df)} manifesto documents")
    print(df[['party', 'chamber', 'text_length', 'ideology']].to_string())

    return df


def load_seat_data():
    """Load election results and create seat mapping."""
    seat_data = {}

    # Load 衆議院 results
    shu_file = Path('SSDSE/01_election_pledges/election/結果/衆議院 当選者数.txt')
    if shu_file.exists():
        df_shu = pd.read_csv(shu_file, sep='\t')
        party_col = df_shu.columns[0]
        seats_col = df_shu.columns[3]  # 当選列

        for _, row in df_shu.iterrows():
            party_tsv = row[party_col]
            seats = row[seats_col]
            party_canonical, _ = parse_party_name(party_tsv)
            seat_data[(party_canonical, '衆議院')] = int(seats)

    # Load 参議院 results
    san_file = Path('SSDSE/01_election_pledges/election/結果/参議院選挙 当選者数.txt')
    if san_file.exists():
        df_san = pd.read_csv(san_file, sep='\t')
        party_col = df_san.columns[0]
        seats_col = df_san.columns[3]  # 今回当選列

        for _, row in df_san.iterrows():
            party_tsv = row[party_col]
            seats = row[seats_col]
            party_canonical, _ = parse_party_name(party_tsv)
            seat_data[(party_canonical, '参議院')] = int(seats)

    print(f"\n[Seats] Loaded seat data for {len(seat_data)} party-chamber combinations:")
    for (party, chamber), seats in sorted(seat_data.items()):
        print(f"  {party:15} ({chamber}): {seats:3} seats")

    return seat_data


def calculate_seat_shares(seat_data):
    """Calculate seat share for each party-chamber combination."""
    # Calculate totals per chamber
    chamber_totals = {}
    for (party, chamber), seats in seat_data.items():
        chamber_totals[chamber] = chamber_totals.get(chamber, 0) + seats

    print(f"\n[Totals] Total seats per chamber:")
    for chamber, total in chamber_totals.items():
        print(f"  {chamber}: {total} seats")

    # Calculate shares
    seat_shares = {}
    for (party, chamber), seats in seat_data.items():
        share = seats / chamber_totals.get(chamber, 1)
        seat_shares[(party, chamber)] = share

    return seat_shares, chamber_totals


def create_document_index(df_manifestos, seat_data):
    """Create a mapping from document index to (party, chamber, ideology)."""
    doc_metadata = []
    for idx, row in df_manifestos.iterrows():
        party = row['party']
        chamber = row['chamber']
        seats = seat_data.get((party, chamber), 0)
        doc_metadata.append({
            'doc_id': idx,
            'party': party,
            'chamber': chamber,
            'ideology': row['ideology'],
            'seats': seats
        })

    return pd.DataFrame(doc_metadata)


def fit_topic_model(df_manifestos, n_topics=12):
    """Fit NMF topic model on manifesto texts."""
    print(f"\n[NMF] Fitting NMF with {n_topics} topics...")

    texts = df_manifestos['text'].tolist()

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=3000,
        lowercase=False  # Preserve Japanese text
    )
    tfidf = vectorizer.fit_transform(texts)
    print(f"  TF-IDF matrix shape: {tfidf.shape}")

    # NMF
    nmf = NMF(
        n_components=n_topics,
        init='nndsvda',
        random_state=42,
        max_iter=500
    )
    W = nmf.fit_transform(tfidf)  # documents x topics
    H = nmf.components_  # topics x vocabulary

    print(f"  NMF complete: W shape {W.shape}, H shape {H.shape}")

    # Get top terms for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics_terms = []
    for topic_idx, topic in enumerate(H):
        top_indices = topic.argsort()[-15:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        topics_terms.append(top_terms)
        print(f"  Topic {topic_idx:2}: {', '.join(top_terms[:5])}...")

    return W, H, vectorizer, topics_terms


def compute_rq_scores(W, df_docs, seat_shares, topics_terms):
    """Compute RQ-1 through RQ-4 scores."""

    n_topics = W.shape[1]
    topics_indexed = list(range(n_topics))

    # RQ-1: Mandate Score (seat-weighted topic importance)
    mandate_scores = {}
    for t in topics_indexed:
        score = 0.0
        for doc_id, row in df_docs.iterrows():
            party = row['party']
            chamber = row['chamber']
            seats_share = seat_shares.get((party, chamber), 0)
            score += seats_share * W[doc_id, t]
        mandate_scores[t] = score

    # RQ-2: Consensus Score (cross-party ideological breadth)
    consensus_scores = {}
    for t in topics_indexed:
        ideologies = set()
        coverage_sum = 0.0
        threshold = 0.1

        for doc_id, row in df_docs.iterrows():
            if W[doc_id, t] >= threshold:
                ideologies.add(row['ideology'])
                coverage_sum += 1.0

        breadth = len([i for i in ideologies if i != 'unknown']) / 3.0  # max 3 ideologies
        coverage = coverage_sum / len(df_docs)
        consensus_scores[t] = breadth * coverage

    # RQ-3: Temporal Consistency (衆院 vs 参院 agreement)
    temporal_scores = {}
    for t in topics_indexed:
        # Get vectors for each chamber
        shu_mask = df_docs['chamber'] == '衆議院'
        san_mask = df_docs['chamber'] == '参議院'

        if shu_mask.sum() > 0 and san_mask.sum() > 0:
            shu_vec = W[shu_mask, t].mean()
            san_vec = W[san_mask, t].mean()

            # Normalize by max to get similarity
            max_val = max(shu_vec, san_vec) + 1e-10
            temporal_scores[t] = min(shu_vec, san_vec) / max_val
        else:
            temporal_scores[t] = 0.5

    # RQ-4: Composite Score (3 dimensions integrated)
    # Normalize each dimension to [0, 1]
    def normalize(scores):
        min_s = min(scores.values())
        max_s = max(scores.values())
        if max_s - min_s < 1e-10:
            return {k: 0.5 for k in scores.keys()}
        return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}

    mandate_norm = normalize(mandate_scores)
    consensus_norm = normalize(consensus_scores)
    temporal_norm = normalize(temporal_scores)

    # Three weighting schemes
    composite_equal = {}
    composite_democracy = {}
    composite_stability = {}

    for t in topics_indexed:
        composite_equal[t] = (mandate_norm[t] + consensus_norm[t] + temporal_norm[t]) / 3
        composite_democracy[t] = 0.5 * mandate_norm[t] + 0.3 * consensus_norm[t] + 0.2 * temporal_norm[t]
        composite_stability[t] = 0.25 * mandate_norm[t] + 0.25 * consensus_norm[t] + 0.5 * temporal_norm[t]

    return {
        'mandate': mandate_scores,
        'consensus': consensus_scores,
        'temporal': temporal_scores,
        'composite_equal': composite_equal,
        'composite_democracy': composite_democracy,
        'composite_stability': composite_stability,
    }


def label_topics(topics_terms):
    """Assign human-readable labels to topics based on top terms."""
    labels = {}
    topic_keywords = {
        'economy': ['経済', '成長', '産業', 'GDP', '企業'],
        'security': ['安全保障', '防衛', '国防', '外交', '国際'],
        'health': ['医療', '健康', '病院', '介護', '社会保障'],
        'energy': ['エネルギー', '原発', '環境', '気候', '再生可能'],
        'child': ['子育て', '教育', '保育', '少子化'],
        'labor': ['労働', '雇用', '給与', '賃金'],
        'agri': ['農業', '食料', '農村', '漁業'],
        'digital': ['デジタル', 'IT', '政府', '行政'],
        'tax': ['税', '財政', '予算', '社会保障費'],
        'region': ['地域', '地方', '創生', '過疎'],
        'const': ['憲法', '政治', '選挙', '民主'],
    }

    for t, terms in enumerate(topics_terms):
        terms_lower = [term for term in terms]

        best_label = f'Topic-{t}'
        best_score = 0

        for label, keywords in topic_keywords.items():
            score = sum(1 for term in terms_lower for keyword in keywords if keyword in term)
            if score > best_score:
                best_score = score
                best_label = label

        labels[t] = best_label

    return labels


def save_results(scores, topics_terms, topic_labels, df_docs):
    """Save analysis results to CSV files."""
    out_dir = Path('SSDSE/01_election_pledges/election/src/results')
    out_dir.mkdir(exist_ok=True)

    n_topics = len(topics_terms)

    # Create results dataframe
    results = []
    for t in range(n_topics):
        results.append({
            'topic_id': t,
            'label': topic_labels[t],
            'mandate_score': scores['mandate'][t],
            'consensus_score': scores['consensus'][t],
            'temporal_score': scores['temporal'][t],
            'composite_equal': scores['composite_equal'][t],
            'composite_democracy': scores['composite_democracy'][t],
            'composite_stability': scores['composite_stability'][t],
            'top_terms': ', '.join(topics_terms[t][:7]),
        })

    df_results = pd.DataFrame(results)

    # Sort by different criteria
    df_rq1 = df_results.sort_values('mandate_score', ascending=False)
    df_rq2 = df_results.sort_values('consensus_score', ascending=False)
    df_rq3 = df_results.sort_values('temporal_score', ascending=False)
    df_rq4_equal = df_results.sort_values('composite_equal', ascending=False)
    df_rq4_democracy = df_results.sort_values('composite_democracy', ascending=False)
    df_rq4_stability = df_results.sort_values('composite_stability', ascending=False)

    # Save all results
    df_rq1.to_csv(out_dir / 'rq1_mandate.csv', index=False, encoding='utf-8')
    df_rq2.to_csv(out_dir / 'rq2_consensus.csv', index=False, encoding='utf-8')
    df_rq3.to_csv(out_dir / 'rq3_temporal.csv', index=False, encoding='utf-8')
    df_rq4_equal.to_csv(out_dir / 'rq4_composite_equal.csv', index=False, encoding='utf-8')
    df_rq4_democracy.to_csv(out_dir / 'rq4_composite_democracy.csv', index=False, encoding='utf-8')
    df_rq4_stability.to_csv(out_dir / 'rq4_composite_stability.csv', index=False, encoding='utf-8')

    print(f"\n[Output] Results saved to {out_dir}/")
    print("\nTop 5 topics by RQ-1 (Mandate Score):")
    print(df_rq1[['label', 'mandate_score']].head().to_string(index=False))

    print("\nTop 5 topics by RQ-2 (Consensus Score):")
    print(df_rq2[['label', 'consensus_score']].head().to_string(index=False))

    print("\nTop 5 topics by RQ-3 (Temporal Consistency):")
    print(df_rq3[['label', 'temporal_score']].head().to_string(index=False))

    print("\nTop 5 topics by RQ-4 (Composite - Equal):")
    print(df_rq4_equal[['label', 'composite_equal']].head().to_string(index=False))

    print("\nTop 5 topics by RQ-4 (Composite - Democracy-first):")
    print(df_rq4_democracy[['label', 'composite_democracy']].head().to_string(index=False))

    print("\nTop 5 topics by RQ-4 (Composite - Stability-first):")
    print(df_rq4_stability[['label', 'composite_stability']].head().to_string(index=False))

    # Find robust topic (stable across all weight schemes)
    all_ranks = []
    for scheme in ['composite_equal', 'composite_democracy', 'composite_stability']:
        ranks = df_results.sort_values(scheme, ascending=False).reset_index(drop=True)['label'].values
        all_ranks.append(list(ranks))

    print("\n[Robust Topics] Topics stable across all weighting schemes:")
    for i in range(min(5, len(all_ranks[0]))):
        topic = all_ranks[0][i]
        rank_equal = list(all_ranks[0]).index(topic) + 1
        rank_democracy = list(all_ranks[1]).index(topic) + 1 if topic in all_ranks[1] else '—'
        rank_stability = list(all_ranks[2]).index(topic) + 1 if topic in all_ranks[2] else '—'
        rank_variance = (rank_equal + (rank_democracy if isinstance(rank_democracy, int) else 0) +
                        (rank_stability if isinstance(rank_stability, int) else 0)) / 3
        print(f"  {topic:20} - Ranks: {rank_equal} / {rank_democracy} / {rank_stability} (avg variance: {rank_variance:.1f})")

    return df_results


def main():
    print("=" * 80)
    print("最も取り組むべき課題の分析 (Research Question Analysis)")
    print("=" * 80)

    # Load data
    df_manifestos = load_manifestos()
    seat_data = load_seat_data()
    seat_shares, chamber_totals = calculate_seat_shares(seat_data)
    df_docs = create_document_index(df_manifestos, seat_data)

    # Topic modeling
    W, H, vectorizer, topics_terms = fit_topic_model(df_manifestos, n_topics=12)
    topic_labels = label_topics(topics_terms)

    # Compute RQ scores
    scores = compute_rq_scores(W, df_docs, seat_shares, topics_terms)

    # Save results
    df_results = save_results(scores, topics_terms, topic_labels, df_docs)

    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)


if __name__ == '__main__':
    main()
