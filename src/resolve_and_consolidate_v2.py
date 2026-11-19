
"""Stage 12.1: rule-based context recovery and cautious incident grouping.
Run: python src/resolve_and_consolidate_v2.py
Requires pandas and scikit-learn. Original files are never overwritten.
"""
from pathlib import Path
from collections import defaultdict
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'data' / 'extracted'
DEST = ROOT / 'data' / 'consolidated'

# Place patterns: longer and more specific first; extend as necessary.
PLACES = {
    'Gharonda, Karnal, Haryana': [r'\bgharonda\b'],
    'Kartarpur, Jalandhar, Punjab': [r'\bkartarpur\b'],
    'ITO, Delhi': [r'\b(?:ito|i\.?t\.?o\.?)\s*,?\s*delhi\b', r'\bito\b'],
    'Red Fort, Delhi': [r'\bred fort\b', r'\blal qila\b'],
    'Singhu Border, Delhi': [r'\bsinghu(?: border)?\b'],
    'Tikri Border, Delhi': [r'\btikri(?: border)?\b'],
    'Ghazipur Border, Delhi': [r'\bghazipur(?: border)?\b'],
    'Lakhimpur Kheri, Uttar Pradesh': [r'\blakhimpur(?: kheri)?\b'],
    'Karnal, Haryana': [r'\bkarnal\b'],
    'Jalandhar, Punjab': [r'\bjalandhar\b'],
    'Chandigarh, India': [r'\bchandigarh\b'],
    'Delhi, India': [r'\b(?:new )?delhi\b'],
    'Mumbai, Maharashtra': [r'\bmumbai\b'],
    'Jaipur, Rajasthan': [r'\bjaipur\b'],
    'Noida, Uttar Pradesh': [r'\bnoida\b'],
    'Ghaziabad, Uttar Pradesh': [r'\bghaziabad\b'],
    'Muzaffarnagar, Uttar Pradesh': [r'\bmuzaffarnagar\b'],
    'Bengaluru, Karnataka': [r'\bbengaluru\b', r'\bbangalore\b'],
    'Kolkata, West Bengal': [r'\bkolkata\b'],
    'Patna, Bihar': [r'\bpatna\b'],
    'Chennai, Tamil Nadu': [r'\bchennai\b'],
    'Hyderabad, Telangana': [r'\bhyderabad\b'],
    'Bhopal, Madhya Pradesh': [r'\bbhopal\b'],
    'Ahmedabad, Gujarat': [r'\bahmedabad\b'],
    'Punjab, India': [r'\bpunjab\b'],
    'Haryana, India': [r'\bharyana\b'],
    'Uttar Pradesh, India': [r'\buttar pradesh\b'],
    'Rajasthan, India': [r'\brajasthan\b'],
}

FOREIGN = re.compile(
    r'\b(?:london|toronto|canada|united states|usa|washington|'
    r'new york|sydney|pakistan|australia|united kingdom)\b',
    re.I
)

MONTHS = (
    r'jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|'
    r'may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|'
    r'oct(?:ober)?|nov(?:ember)?|dec(?:ember)?'
)

DATE_PATTERN = re.compile(
    rf'\b(?:\d{{1,2}}\s+(?:{MONTHS})\s*\d{{0,4}}|'
    rf'(?:{MONTHS})\s+\d{{1,2}}(?:,?\s*\d{{4}})?)\b',
    re.I
)

EVENT_VERBS = re.compile(
    r'\b(?:protest(?:ed|ing|s)?|clash(?:es|ed)?|'
    r'lathi\s*charg\w*|tear\s*gas|rally|march(?:ed|ing)?|'
    r'blockad\w*|detain\w*|arrest\w*|attack\w*|'
    r'demonstrat\w*)\b',
    re.I
)

FUTURE = re.compile(
    r'\b(?:will|tomorrow|next week|scheduled to|'
    r'plans? to|going to|expected to)\b',
    re.I
)

PAST = re.compile(
    r'\b(?:last year|years ago|anniversary|'
    r'remember when|back in \d{4})\b',
    re.I
)

ONGOING = re.compile(
    r'\b(?:still|ongoing|continues?|continuing|'
    r'have been|has been)\b',
    re.I
)


def val(x):
    return '' if pd.isna(x) else str(x).strip()


def event_clause(text):
    text = re.sub(
        r'https?://\S+|\[URL\]|\[USER\]',
        ' ',
        val(text),
        flags=re.I
    )

    pieces = re.split(
        r'(?<=[.!?;])\s+|\s+[|]\s+',
        text
    )

    matches = [
        p for p in pieces
        if EVENT_VERBS.search(p)
    ]

    return matches[0] if matches else text[:350]


def candidates(text):
    hits = []

    for name, patterns in PLACES.items():
        if any(re.search(p, text, re.I) for p in patterns):
            hits.append(name)

    # Remove broader place when a specific contained place was found.
    selected = []

    for name in hits:
        if not any(
            name != other
            and name.split(',')[0].lower()
            in other.lower().split(', ')[1:]
            for other in hits
        ):
            selected.append(name)

    # Specific locations should win over a broad state/city.
    if any(x.startswith('Gharonda,') for x in selected):
        selected = [
            x for x in selected
            if x != 'Karnal, Haryana'
        ]

    if any(x.startswith('Kartarpur,') for x in selected):
        selected = [
            x for x in selected
            if x != 'Jalandhar, Punjab'
        ]

    if any(x.endswith(', Delhi') for x in selected):
        selected = [
            x for x in selected
            if x != 'Delhi, India'
        ]

    return list(dict.fromkeys(selected))


def location_for(row):
    evidence = val(row.get('event_evidence'))
    full = val(row.get('text'))
    clause = event_clause(full)

    # Use event evidence first, without blindly borrowing
    # unrelated locations from the full tweet.
    for source, segment in [
        ('event_evidence', evidence),
        ('event_clause', clause)
    ]:
        found = candidates(segment)

        if len(found) == 1:
            return found[0], source, 'rule_candidate'

        if len(found) > 1:
            return '', source, 'multiple_places_review'

    existing = val(row.get('event_location'))
    found = candidates(existing)

    if len(found) == 1:
        return (
            found[0],
            'existing_extraction',
            'existing_candidate_review'
        )

    if FOREIGN.search(clause) and not candidates(clause):
        return (
            '',
            'event_clause',
            'possible_foreign_event_review'
        )

    if candidates(full):
        return '', 'full_text', 'unlinked_place_review'

    return '', 'none', 'missing_location'


def date_for(row):
    post = pd.to_datetime(
        val(row.get('post_date')) or val(row.get('date')),
        errors='coerce',
        utc=True
    )

    evidence = val(row.get('event_evidence'))
    clause = event_clause(val(row.get('text')))
    explicit = val(row.get('event_date'))

    if explicit:
        parsed = pd.to_datetime(
            explicit,
            errors='coerce',
            utc=True,
            dayfirst=True
        )

        if pd.notna(parsed):
            return (
                parsed.date().isoformat(),
                'extracted_explicit',
                'candidate'
            )

    for segment in [evidence, clause]:
        m = DATE_PATTERN.search(segment)

        if m:
            raw = m.group(0)

            if (
                not re.search(r'\b\d{4}\b', raw)
                and pd.notna(post)
            ):
                raw += ' ' + str(post.year)

            parsed = pd.to_datetime(
                raw,
                errors='coerce',
                utc=True,
                dayfirst=True
            )

            if pd.notna(parsed):
                return (
                    parsed.date().isoformat(),
                    'text_explicit',
                    'candidate'
                )

        if pd.notna(post):
            if re.search(r'\byesterday\b', segment, re.I):
                return (
                    (post - pd.Timedelta(days=1))
                    .date()
                    .isoformat(),
                    'relative_yesterday',
                    'candidate'
                )

            if re.search(
                r'\btoday\b|\bthis morning\b|\btonight\b',
                segment,
                re.I
            ):
                return (
                    post.date().isoformat(),
                    'relative_today',
                    'candidate'
                )

    return '', 'unknown_event_date', 'unresolved'


def occurrence_for(row):
    s = event_clause(val(row.get('text')))

    if FUTURE.search(s):
        return 'planned_or_future_review'

    if PAST.search(s):
        return 'historical_review'

    if ONGOING.search(s):
        return 'ongoing_reported'

    return 'reported_occurrence_unverified'


def normalized_text(s):
    s = re.sub(
        r'https?://\S+|\[URL\]|\[USER\]',
        ' ',
        val(s).lower(),
        flags=re.I
    )

    return re.sub(
        r'\s+',
        ' ',
        re.sub(r'[^a-z0-9\s]', ' ', s)
    ).strip()


def process(version):
    path = SOURCE / f'event_mentions_{version}.csv'

    if not path.exists():
        print(f'{version}: source not found: {path}')
        return

    df = pd.read_csv(
        path,
        dtype={'post_id': 'string'},
        low_memory=False
    )

    required = {
        'post_id',
        'text',
        'event_type',
        'extraction_status'
    }

    if required - set(df.columns):
        raise ValueError(
            f'{path.name} missing '
            f'{sorted(required - set(df.columns))}'
        )

    df = df[
        df.extraction_status.astype('string')
        .eq('event_candidate')
    ].copy().reset_index(drop=True)

    if df.empty:
        print(f'{version}: no candidates')
        return

    # STEP 1: Resolve locations
    resolved = df.apply(
        location_for,
        axis=1,
        result_type='expand'
    )

    df[
        [
            'resolved_event_location',
            'location_source',
            'location_resolution_status'
        ]
    ] = resolved

    # STEP 2: Resolve dates
    dates = df.apply(
        date_for,
        axis=1,
        result_type='expand'
    )

    df[
        [
            'resolved_event_date',
            'event_date_source',
            'event_date_status'
        ]
    ] = dates

    # STEP 3: Determine occurrence status
    df['resolved_occurrence_status'] = df.apply(
        occurrence_for,
        axis=1
    )

    df['reference_post_date'] = pd.to_datetime(
        df.get(
            'post_date',
            pd.Series(index=df.index, dtype='object')
        ),
        errors='coerce',
        utc=True
    ).dt.strftime('%Y-%m-%d')

    df['cluster_id_v2'] = ''
    df['cluster_status_v2'] = 'unresolved_context'

    # STEP 4: Build clustering blocks
    # Publication date is a weak reference, NOT an event date.
    blocks = defaultdict(list)

    for i, r in df.iterrows():
        if (
            not val(r.resolved_event_location)
            or r.location_resolution_status != 'rule_candidate'
        ):
            continue

        if r.resolved_occurrence_status in (
            'planned_or_future_review',
            'historical_review'
        ):
            continue

        day = (
            val(r.resolved_event_date)
            or val(r.reference_post_date)
        )

        if not day:
            continue

        blocks[
            (
                r.resolved_event_location,
                val(r.event_type).lower(),
                day
            )
        ].append(i)

    # STEP 5: Text similarity grouping
    # Avoid transitive chaining: compare every member to seed.
    groups = []

    for _, indices in blocks.items():
        texts = [
            normalized_text(
                event_clause(df.at[i, 'text'])
            )
            for i in indices
        ]

        if not any(texts):
            continue

        if len(indices) == 1:
            groups.append(indices)
            continue

        matrix = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english'
        ).fit_transform(texts)

        similarities = cosine_similarity(matrix)

        unused = set(range(len(indices)))

        while unused:
            seed = min(unused)
            cluster = [seed]
            unused.remove(seed)

            for j in sorted(unused):
                if similarities[seed, j] >= 0.72:
                    cluster.append(j)

            unused.difference_update(cluster)

            groups.append([
                indices[j] for j in cluster
            ])

    # STEP 6: Create provisional incident records
    records = []

    for number, ids in enumerate(
        sorted(groups, key=lambda g: min(g)),
        1
    ):
        cid = f'{version.upper()}-CTX-{number:04d}'
        rows = df.loc[ids]

        post_ids = sorted(
            set(rows.post_id.dropna().astype(str))
        )

        for i in ids:
            df.at[i, 'cluster_id_v2'] = cid

            df.at[i, 'cluster_status_v2'] = (
                'provisional_singleton'
                if len(ids) == 1
                else 'provisional_group'
            )

        records.append({
            'cluster_id': cid,
            'event_type': val(rows.iloc[0].event_type),
            'location_candidate': val(
                rows.iloc[0].resolved_event_location
            ),
            'event_date_candidate': val(
                rows.iloc[0].resolved_event_date
            ),
            'reference_post_date': val(
                rows.iloc[0].reference_post_date
            ),
            'date_basis': val(
                rows.iloc[0].event_date_source
            ),
            'mention_count': len(ids),
            'supporting_post_count': len(post_ids),
            'supporting_post_ids': json.dumps(post_ids),
            'representative_text': val(
                rows.iloc[0].text
            )[:450],
            'verification_status': 'unverified_rule_based'
        })

    # STEP 7: Save results
    DEST.mkdir(
        parents=True,
        exist_ok=True
    )

    outputs = {
        DEST / f'event_mentions_context_{version}_v2.csv': df,

        DEST / f'incident_clusters_{version}_v2.csv':
            pd.DataFrame(
                records,
                columns=[
                    'cluster_id',
                    'event_type',
                    'location_candidate',
                    'event_date_candidate',
                    'reference_post_date',
                    'date_basis',
                    'mention_count',
                    'supporting_post_count',
                    'supporting_post_ids',
                    'representative_text',
                    'verification_status'
                ]
            ),

        DEST / f'event_context_review_{version}_v2.csv':
            df[
                df.cluster_status_v2.eq(
                    'unresolved_context'
                )
            ].copy()
    }

    summary_path = (
        DEST /
        f'event_context_summary_{version}_v2.json'
    )

    # Do not overwrite existing outputs
    for output in [*outputs, summary_path]:
        if output.exists():
            raise FileExistsError(
                'Output already exists, refusing overwrite: '
                f'{output}'
            )

    for output, data in outputs.items():
        data.to_csv(
            output,
            index=False
        )

    summary = {
        'version': version,
        'candidate_mentions': len(df),

        'resolved_location_candidates': int(
            df.resolved_event_location.ne('').sum()
        ),

        'resolved_event_date_candidates': int(
            df.resolved_event_date.ne('').sum()
        ),

        'provisional_clusters_including_singletons':
            len(records),

        'multi_post_clusters': sum(
            r['supporting_post_count'] > 1
            for r in records
        ),

        'unresolved_context_mentions': int(
            df.cluster_status_v2.eq(
                'unresolved_context'
            ).sum()
        ),

        'location_status_counts':
            df.location_resolution_status
            .value_counts()
            .to_dict(),

        'date_source_counts':
            df.event_date_source
            .value_counts()
            .to_dict(),

        'warning': (
            'Rule-based hypotheses; no human verification '
            'or independent geocoding.'
        )
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8'
    )

    print(
        f'\n{version.upper()} SUMMARY\n'
        + json.dumps(summary, indent=2)
    )

    print(
        'Saved:',
        *(str(x) for x in [*outputs, summary_path]),
        sep='\n  '
    )


if __name__ == '__main__':
    for v in ('v4', 'v5'):
        process(v)
