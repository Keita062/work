"""
Data loader for Japanese election manifesto analysis.
Reads manifesto text files and election results.
"""

import os
import pandas as pd
from pathlib import Path

# Party name mapping from files to canonical names
PARTY_MAP = {
    # 衆議院
    '自民党': '自民党',
    '中道': '中道',
    '参政党': '参政党',
    '国民民主': '国民民主党',
    'チームみらい': 'チームみらい',
    'れいわ': '令和新選組',
    '社民党': '社民党',
    '日本保守党': '日本保守党',
    # 参議院
    '立憲民主党': '立憲民主党',
    '公明党': '公明党',
    '共産党': '共産党',
    '令和新選組': '令和新選組',
    '国民民主党': '国民民主党',
    '日本維新の党': '日本維新の党',
    'みんなで作る党': 'みんなで作る党',
}

def load_manifesto_texts(ssdse_root='..'):
    """Load manifesto text files from SSDSE/01_election_pledges/data/ directory."""
    data_dir = Path(ssdse_root) / 'data'

    manifestos = []

    for chamber in ['衆議院', '参議院']:
        chamber_dir = data_dir / chamber
        if not chamber_dir.exists():
            print(f"Warning: {chamber_dir} not found (looking at {chamber_dir.absolute()})")
            continue

        for txt_file in sorted(chamber_dir.glob('*.txt')):
            # Extract party name from filename
            party_name = txt_file.stem
            canonical_name = PARTY_MAP.get(party_name, party_name)

            # Read text
            text = txt_file.read_text(encoding='utf-8')

            # Skip if empty
            if not text.strip():
                print(f"Warning: {chamber}/{party_name} is empty")
                continue

            manifestos.append({
                'party': canonical_name,
                'party_original': party_name,
                'chamber': chamber,
                'text': text,
                'text_length': len(text),
                'char_count': len(text.replace('\n', '').replace('\r', '').replace(' ', ''))
            })

    if len(manifestos) == 0:
        print("ERROR: No manifesto documents loaded!")
        return pd.DataFrame()

    df = pd.DataFrame(manifestos)
    print(f"Loaded {len(df)} manifesto documents")
    print(f"Chambers: {df['chamber'].unique()}")
    print(f"\nParty coverage:")
    print(df.groupby(['chamber', 'party'])[['char_count']].agg(['count', 'sum']))

    return df


def load_election_results(ssdse_root='..'):
    """Load election results from TSV files."""
    results_dir = Path(ssdse_root) / 'election' / '結果'

    election_data = {}

    # Load 衆議院 (House of Representatives) results
    shu_file = results_dir / '衆議院 当選者数.txt'
    if shu_file.exists():
        df_shu = pd.read_csv(shu_file, sep='\t')
        df_shu['chamber'] = '衆議院'
        election_data['shu'] = df_shu
        print(f"Loaded 衆議院 results: {df_shu.shape}")
        print(df_shu)

    # Load 参議院 (House of Councillors) results
    san_file = results_dir / '参議院選挙 当選者数.txt'
    if san_file.exists():
        df_san = pd.read_csv(san_file, sep='\t')
        df_san['chamber'] = '参議院'
        election_data['san'] = df_san
        print(f"\nLoaded 参議院 results: {df_san.shape}")
        print(df_san)

    return election_data


def create_seat_map(election_data):
    """Create a mapping of (party, chamber) -> seats."""
    seat_map = {}

    for chamber_key, df in election_data.items():
        # Column names vary slightly; try to find the seats column
        chamber_name = '衆議院' if chamber_key == 'shu' else '参議院'

        # Try different possible column names for seats
        seats_cols = ['当選', '今回当選', 'seats']
        seats_col = None
        for col in seats_cols:
            if col in df.columns:
                seats_col = col
                break

        if seats_col is None:
            print(f"Warning: Could not find seats column in {chamber_name}")
            print(f"Available columns: {df.columns.tolist()}")
            continue

        party_col = df.columns[0]  # First column is usually party

        for _, row in df.iterrows():
            party_name = row[party_col]
            seats = row[seats_col]

            # Normalize party name
            canonical_name = PARTY_MAP.get(party_name, party_name)

            seat_map[(canonical_name, chamber_name)] = int(seats)

    print(f"\nCreated seat map with {len(seat_map)} entries:")
    for (party, chamber), seats in sorted(seat_map.items()):
        print(f"  {party} ({chamber}): {seats} seats")

    return seat_map


def calculate_seat_shares(seat_map):
    """Calculate seat share for each party-chamber combination."""
    # Group by chamber and sum seats
    chamber_totals = {}
    for (party, chamber), seats in seat_map.items():
        chamber_totals[chamber] = chamber_totals.get(chamber, 0) + seats

    print(f"\nTotal seats per chamber:")
    for chamber, total in chamber_totals.items():
        print(f"  {chamber}: {total}")

    seat_shares = {}
    for (party, chamber), seats in seat_map.items():
        seat_shares[(party, chamber)] = seats / chamber_totals.get(chamber, 1)

    return seat_shares, chamber_totals


if __name__ == '__main__':
    # Test loading
    df_manifestos = load_manifesto_texts()
    print("\n" + "="*80 + "\n")

    election_data = load_election_results()
    print("\n" + "="*80 + "\n")

    seat_map = create_seat_map(election_data)
    print("\n" + "="*80 + "\n")

    seat_shares, chamber_totals = calculate_seat_shares(seat_map)

    # Save for later use
    df_manifestos.to_pickle('SSDSE/01_election_pledges/election/src/results/df_manifestos.pkl')
    pd.Series(seat_map).to_pickle('SSDSE/01_election_pledges/election/src/results/seat_map.pkl')
    pd.Series(seat_shares).to_pickle('SSDSE/01_election_pledges/election/src/results/seat_shares.pkl')

    print("\nData saved to src/results/")
