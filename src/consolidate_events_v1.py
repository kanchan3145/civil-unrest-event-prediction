
"""Stage 12: conservative India-only incident consolidation (V4 and V5).

Run: python src/consolidate_events_v1.py
Dependencies: pandas, scikit-learn (already used in the ML project).
Does not overwrite existing files. Does not claim human verification.
"""
from pathlib import Path
from collections import defaultdict
from datetime import timedelta
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / 'data' / 'extracted'
OUTPUT_DIR = ROOT / 'data' / 'consolidated'
VERSIONS = ('v4', 'v5')
SIMILARITY_THRESHOLD = 0.78
MAX_DAY_GAP = 2

# These are conservative geographic cues, not a complete Indian gazetteer.
INDIAN_PLACES = {
    'delhi', 'new delhi', 'singhu', 'tikri', 'ghazipur', 'red fort',
    'punjab', 'haryana', 'uttar pradesh', 'up', 'rajasthan', 'jaipur',
    'maharashtra', 'mumbai', 'karnal', 'lakhimpur kheri', 'kerala',
    'wayanad', 'chandigarh', 'noida', 'ghaziabad', 'muzaffarnagar',
    'bengaluru', 'bangalore', 'karnataka', 'west bengal', 'kolkata',
    'madhya pradesh', 'bhopal', 'gujarat', 'ahmedabad', 'bihar',
    'patna', 'tamil nadu', 'chennai', 'telangana', 'hyderabad',
    'andhra pradesh', 'odisha', 'assam', 'jammu', 'kashmir',
    'india', 'indian'
}
FOREIGN_PLACES = {
    'united states', 'usa', 'america', 'canada', 'toronto',
    'united kingdom', 'uk', 'london', 'pakistan', 'china',
    'australia', 'sydney', 'melbourne', 'germany', 'france'
}
GENERIC_LOCATIONS = {'', 'unknown', 'none', 'nan', 'not found', 'not specified', 'india', 'indian'}


def clean(value):
    return '' if pd.isna(value) else str(value).strip()


def normalized_location(value):
    value = clean(value).lower()
    value = re.sub(r'[^a-z\s]', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def place_match(location, candidates):
    return any(re.search(r'(?<!\w)' + re.escape(p) + r'(?!\w)', location) for p in candidates)


def geo_status(location):
    loc = normalized_location(location)
    if loc in GENERIC_LOCATIONS:
        return 'location_unresolved'
    if place_match(loc, FOREIGN_PLACES) and not place_match(loc, INDIAN_PLACES):
        return 'outside_india_review'
    if place_match(loc, INDIAN_PLACES):
        return 'india_location_candidate'
    return 'location_unresolved'


def normalize_text(value):
    value = clean(value).lower()
    value = re.sub(r'https?://\S+|\[url\]|\[user\]', ' ', value)
    value = re.sub(r'[^a-z0-9\s]', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def time_value(row):
    # Only explicit extracted event dates can support event-date clustering.
    # Tweet publication date is recorded for context but is not event occurrence time.
    return pd.to_datetime(clean(row.get('event_date')), errors='coerce', utc=True)


def choose_text(row):
    evidence = clean(row.get('event_evidence'))
    return evidence if len(evidence) >= 25 else clean(row.get('text'))


def build(version):
    source = INPUT_DIR / f'event_mentions_{version}.csv'
    if not source.exists():
        print(f'[{version}] Missing {source}; skipped')
        return
    df = pd.read_csv(source, dtype={'post_id': 'string'}, low_memory=False)
    needed = {'post_id', 'event_type', 'event_location', 'event_date', 'extraction_status'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'{source.name}: missing columns {sorted(missing)}')
    df = df[df['extraction_status'].astype('string').eq('event_candidate')].copy().reset_index(drop=True)
    if df.empty:
        print(f'[{version}] No event_candidate mentions; skipped')
        return
    df['geo_review_status'] = df['event_location'].map(geo_status)
    df['normalized_location'] = df['event_location'].map(normalized_location)
    df['normalized_evidence'] = df.apply(lambda r: normalize_text(choose_text(r)), axis=1)
    df['parsed_event_date'] = df.apply(time_value, axis=1)
    df['consolidation_status'] = 'needs_location_review'
    df.loc[df.geo_review_status.eq('outside_india_review'), 'consolidation_status'] = 'outside_india_review'
    df.loc[df.geo_review_status.eq('india_location_candidate'), 'consolidation_status'] = 'needs_date_review'
    eligible = df.index[
        df.geo_review_status.eq('india_location_candidate')
        & df.parsed_event_date.notna()
        & df.normalized_evidence.str.len().ge(25)
    ].tolist()
    parent = {i: i for i in eligible}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Block by exact normalized extracted place and event type. Avoid global
    # clustering across different incidents sharing generic protest vocabulary.
    blocks = defaultdict(list)
    for i in eligible:
        r = df.loc[i]
        blocks[(r['normalized_location'], clean(r['event_type']).lower())].append(i)

    for _, indices in blocks.items():
        if len(indices) < 2:
            continue
        texts = [df.at[i, 'normalized_evidence'] for i in indices]
        matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words='english').fit_transform(texts)
        sims = cosine_similarity(matrix)
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                ia, ib = indices[a], indices[b]
                if clean(df.at[ia, 'post_id']) == clean(df.at[ib, 'post_id']):
                    continue  # separate mentions from one post are not independent corroboration
                gap = abs((df.at[ia, 'parsed_event_date'] - df.at[ib, 'parsed_event_date']).total_seconds()) / 86400
                if gap <= MAX_DAY_GAP and sims[a, b] >= SIMILARITY_THRESHOLD:
                    union(ia, ib)

    groups = defaultdict(list)
    for i in eligible:
        groups[find(i)].append(i)
    ordered = sorted(groups.values(), key=lambda ids: min(ids))
    cluster_rows = []
    for num, ids in enumerate(ordered, start=1):
        cluster_id = f'{version.upper()}-IND-{num:04d}'
        subset = df.loc[ids]
        for i in ids:
            df.at[i, 'incident_cluster_id'] = cluster_id
            df.at[i, 'consolidation_status'] = 'provisional_cluster'
        unique_posts = sorted(set(subset['post_id'].dropna().astype(str)))
        cluster_rows.append({
            'incident_cluster_id': cluster_id,
            'event_type': clean(subset.iloc[0]['event_type']),
            'event_location': clean(subset.iloc[0]['event_location']),
            'event_date_earliest': subset['parsed_event_date'].min().date().isoformat(),
            'event_date_latest': subset['parsed_event_date'].max().date().isoformat(),
            'mention_count': len(ids),
            'supporting_post_count': len(unique_posts),
            'supporting_post_ids': json.dumps(unique_posts),
            'representative_evidence': choose_text(subset.iloc[0])[:600],
            'verification_status': 'unverified_provisional',
            'location_scope': 'india_candidate_not_geocoded'
        })
    df['incident_cluster_id'] = df.get('incident_cluster_id', pd.Series(index=df.index, dtype='object')).fillna('')
    df['parsed_event_date'] = df['parsed_event_date'].dt.strftime('%Y-%m-%d')
    out_mentions = OUTPUT_DIR / f'event_mentions_consolidated_{version}_v1.csv'
    out_clusters = OUTPUT_DIR / f'incident_clusters_{version}_v1.csv'
    out_review = OUTPUT_DIR / f'incident_location_date_review_{version}_v1.csv'
    out_summary = OUTPUT_DIR / f'incident_consolidation_summary_{version}_v1.json'
    outputs = (out_mentions, out_clusters, out_review, out_summary)
    existing = [p for p in outputs if p.exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite: {existing}. Rename/remove old outputs intentionally.')
    df.to_csv(out_mentions, index=False)
    pd.DataFrame(cluster_rows, columns=['incident_cluster_id','event_type','event_location','event_date_earliest','event_date_latest','mention_count','supporting_post_count','supporting_post_ids','representative_evidence','verification_status','location_scope']).to_csv(out_clusters, index=False)
    review = df[df['consolidation_status'].ne('provisional_cluster')].copy()
    review.to_csv(out_review, index=False)
    summary = {
        'version': version,
        'candidate_mentions': len(df),
        'provisional_clusters': len(cluster_rows),
        'clustered_mentions': len(eligible),
        'multi_post_clusters': sum(r['supporting_post_count'] > 1 for r in cluster_rows),
        'needs_review_mentions': len(review),
        'status_counts': df['consolidation_status'].value_counts().to_dict(),
        'note': 'Conservative candidate consolidation only; not verified events or an India geocoder.'
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\n[{version}] {json.dumps(summary, indent=2)}')
    print('Saved:', *[str(p) for p in outputs], sep='\n  ')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for version in VERSIONS:
        build(version)


if __name__ == '__main__':
    main()
