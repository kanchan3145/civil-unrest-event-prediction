
from pathlib import Path
import re
import json
import html
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'data/evaluation/minilm_candidate_predictions_v1.csv'
OUT = ROOT / 'data/extracted'
VERSION = 'v4'

RULES = {
    'vehicle_attack': [
        r'\b(?:cars?|vehicles?|trucks?|suvs?)\b.{0,100}\b(?:crush(?:ed|ing)?|ran over|rammed|mowed down)\b',
        r'\b(?:crush(?:ed|ing)?|ran over|rammed|mowed down)\b.{0,100}\b(?:farmers?|protesters?|people|civilians?)\b'
    ],
    'riot_or_violence': [
        r'\b(?:riots?|rioting)\b.{0,110}\b(?:injur(?:ed|ies)|killed|died|violence|broke out)\b',
        r'\b(?:injur(?:ed|ies)|killed|died)\b.{0,110}\b(?:riots?|rioting)\b'
    ],
    'clash': [
        r'\bclash(?:es|ed|ing)?\b',
        r'\bscuffle(?:s)?\b'
    ],
    'police_action': [
        r'\b(?:police|cops|security forces)\b.{0,100}\b(?:fired|deployed|used|using)\b.{0,40}\b(?:tear gas|water cannons?)\b',
        r'\b(?:tear gas|water cannons?|lathi[ -]?charg(?:e|ed|ing))\b',
        r'\b(?:police|cops)\b.{0,80}\b(?:arrested|detained|dispersed)\b.{0,70}\b(?:farmers?|protesters?|demonstrators?)\b'
    ],
    'blockade': [
        r'\b(?:farmers?|protesters?)\b.{0,90}\b(?:blocked|blockaded)\b.{0,70}\b(?:roads?|highways?|borders?|traffic)\b',
        r'\b(?:roads?|highways?|borders?)\b.{0,60}\b(?:blocked|blockaded)\b'
    ],
    'march': [
        r'\b(?:farmers?|protesters?|workers?|demonstrators?)\b.{0,85}\b(?:marched|marching)\b',
        r'\b(?:marched|marching)\b.{0,85}\b(?:farmers?|protesters?|workers?)\b',
        r'\b(?:took out|held|staged)\b.{0,40}\b(?:tractor rally|march|procession)\b'
    ],
    'rally': [
        r'\b(?:held|staged|organized|organised)\b.{0,35}\brall(?:y|ies)\b',
        r'\brall(?:y|ies)\b.{0,35}\b(?:was held|took place|began)\b'
    ],
    'sit_in': [
        r'\b(?:held|staged|began|continued)\b.{0,40}\b(?:sit[ -]?in|dharna)\b'
    ],
    'strike': [
        r'\b(?:workers?|employees?|farmers?)\b.{0,80}\b(?:went on strike|staged a strike)\b',
        r'\b(?:strike|bandh)\b.{0,50}\b(?:began|started|was observed|was held)\b'
    ],
    'protest': [
        r'\b(?:farmers?|workers?|students?|protesters?|demonstrators?|people)\b.{0,90}\b(?:protesting|protested|demonstrating)\b',
        r'\b(?:held|staged|joined|continued)\b.{0,40}\b(?:protest|demonstration|agitation)\b',
        r'\b(?:farmers?|protesters?)\b.{0,70}\b(?:gathered|assembled)\b'
    ],
}

PLANNED = [
    r'\b(?:will|shall|going to)\s+(?:march|protest|hold|stage|organize|organise|block|join)\b',
    r'\b(?:planned|planning|plans|proposed|scheduled|upcoming)\b.{0,65}\b(?:march|rally|protest|strike|demonstration)\b'
]

CANCELLED = [
    r'\b(?:march|rally|protest|strike|demonstration)\b.{0,65}\b(?:cancelled|canceled|postponed|called off)\b',
    r'\b(?:cancelled|canceled|postponed|called off)\b.{0,65}\b(?:march|rally|protest|strike|demonstration)\b'
]

NEGATED = [
    r"\b(?:did not|didn't|never)\s+(?:march|protest|clash|attack|happen|occur)\b",
    r'\bno\s+(?:clash|protest|violence|incident)\b'
]

HYPOTHETICAL = [
    r'\b(?:might|may|could|would)\s+(?:march|protest|clash|attack|block|riot)\b',
    r'\blikely to\s+(?:march|protest|clash|attack|block|riot)\b',
    r'\b(?:rumou?r|unconfirmed)\b'
]

HISTORICAL = [
    r'\b(?:years ago|last year|anniversary|in memory of)\b'
]

ALLEGATION = [
    r'\b(?:alleged|allegedly|claims?|accus(?:ed|ations?)|reportedly)\b',
    r'\b(?:state apparatus|government)\b.{0,70}\b(?:weaponized|committing|carrying out)\b'
]

LOCATIONS = {
    'ITO, Delhi': [r'\b(?:ito intersection|central delhi)\b'],
    'Ghazipur Border': [r'\b(?:ghazipur|gazipur) border\b'],
    'Singhu Border': [r'\bsinghu(?: border)?\b'],
    'Tikri Border': [r'\btikri(?: border)?\b'],
    'New Delhi': [r'\bnew delhi\b'],
    'Delhi': [r'\b(?:delhi|dehli)\b'],
    'Ghaziabad': [r'\bghaziabad\b'],
    'Punjab': [r'\bpunjab\b'],
    'Haryana': [r'\bharyana\b'],
    'Uttar Pradesh': [r'\buttar pradesh\b'],
    'Uttarakhand': [r'\buttarakhand\b'],
    'Mumbai': [r'\bmumbai\b'],
    'Kolkata': [r'\bkolkata\b'],
    'Bengaluru': [r'\b(?:bengaluru|bangalore)\b'],
    'Chandigarh': [r'\bchandigarh\b'],
    'Downing Street': [r'\bdowning street\b'],
}

AUDIT_IDS = [
    '1426909019340369920',
    '1345603554568859648',
    '1354248484937986049',
    '1458053635087142918',
    '1353995986679558144',
    '1353995755929788416',
    '1445690167625277444',
    '1331975238939070465',
    '1395585442163007489',
    '1445248697181216768',
    '1354073743627149312',
    '1453480427755352066',
    '1333882880691417088',
    '1354284290515861505',
    '1355177937595555845',
    '1354274850521575427',
    '1387288908711354369',
    '1336021209260019712',
    '1445964306349117440'
]


def any_match(text, patterns):
    return any(re.search(p, text, re.I) for p in patterns)


def normalize(value):
    return re.sub(
        r'\s+', ' ',
        re.sub(
            r'\[(?:URL|USER)\]|https?://\S+',
            ' ',
            html.unescape(str(value)),
            flags=re.I
        )
    ).strip()


def clauses(text):
    # Split text into sentences and contrast clauses.
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    output = []

    for sentence in sentences:
        for clause in re.split(
            r'\s*;\s*|\s*,\s*(?=(?:but|while|however|whereas)\b)|\s+\b(?:but|however|whereas)\b\s+',
            sentence,
            flags=re.I
        ):
            clause = clause.strip()
            if clause:
                output.append(clause)

    return output


def status_for(clause):
    if '?' in clause or re.match(
        r'^(?:where|when|why|how|did|does|is|are|what)\b',
        clause, re.I
    ):
        return 'question'

    if any_match(clause, CANCELLED):
        return 'cancelled_or_postponed'

    if any_match(clause, NEGATED):
        return 'negated'

    if any_match(clause, HYPOTHETICAL):
        return 'hypothetical'

    if any_match(clause, PLANNED):
        return 'planned_or_future'

    if any_match(clause, HISTORICAL):
        return 'historical_reference'

    if any_match(clause, ALLEGATION):
        return 'allegation_needs_review'

    return 'reported_language'


def military_only(clause):
    # Identify likely interstate military conflicts.
    return bool(
        re.search(
            r'\b(?:soldiers?|troops?|army|military)\b',
            clause, re.I
        )
        and re.search(
            r'\b(?:china|chinese|pakistan|pakistani|border conflict)\b',
            clause, re.I
        )
    )


def extract_location(clause):
    found = []

    for name, patterns in LOCATIONS.items():
        for pattern in patterns:
            m = re.search(pattern, clause, re.I)
            if m:
                found.append((name, m.start(), m.end()))
                break

    if any(
        x[0] in ('New Delhi', 'ITO, Delhi')
        for x in found
    ):
        found = [
            x for x in found
            if x[0] != 'Delhi'
        ]

    names = list(dict.fromkeys(x[0] for x in found))

    if not names:
        return None, 'missing', []

    if len(names) > 1:
        return None, 'multiple_mentions_review', names

    name, start, _ = found[0]
    prefix = clause[max(0, start - 40):start]

    if re.search(
        r'\b(?:towards?|heading to|en route to|marching to|entered? to)\s*$',
        prefix, re.I
    ):
        return None, 'destination_only', names

    if re.search(
        r'\b(?:at|in|near|outside|on)\s*$',
        prefix, re.I
    ):
        return name, 'possible_event_location_review', names

    return None, 'unlinked_location_mention', names


def extract_date(clause):
    m = re.search(r'\b20\d{2}-\d{2}-\d{2}\b', clause)

    if not m:
        return None

    try:
        return pd.to_datetime(
            m.group(),
            format='%Y-%m-%d',
            errors='raise'
        ).strftime('%Y-%m-%d')

    except ValueError:
        return None


def detect_types(clause):
    found = [
        typ for typ, pats in RULES.items()
        if any_match(clause, pats)
    ]

    # Prefer specific events over generic protest mentions.
    primary = next(
        (
            x for x in found
            if x not in ('protest', 'march', 'rally')
        ),
        None
    )

    if primary:
        return [primary], found

    return found[:1], found


# --------------------------------------------------
# V4: PARTICIPANT AND EVENT CONTEXT VALIDATION
# --------------------------------------------------

PARTICIPANTS = (
    r"(?:farmers?|protesters?|demonstrators?|"
    r"agitating farmers?|kisans?|police|cops)"
)

DIRECT_ACTION = re.compile(
    r"\b(?:clash(?:ed|es|ing)?|"
    r"lathi[ -]?charg(?:e|ed|ing)|"
    r"tear gas|water cannons?|"
    r"march(?:ed|ing)?|"
    r"protest(?:ed|ing)?|"
    r"sit[ -]?in|dharna|rail roko|"
    r"blocked|blockaded|injured|killed|"
    r"crushed|ran over|detained|arrested)\b",
    re.I
)

UNRELATED = re.compile(
    r"\b(?:pakistan|islamists?|china|chinese|"
    r"galwan|military|soldiers?)\b",
    re.I
)

PARTICIPANT = re.compile(
    r"\b(?:farmers?|protesters?|demonstrators?|kisans?)\b",
    re.I
)


def explicit_participation(clause):
    # Hashtags alone do not establish event participation.
    clean = re.sub(r"#\S+", " ", clause)

    return bool(
        re.search(
            r"(?:" + PARTICIPANT.pattern + r")"
            r".{0,95}"
            r"(?:" + DIRECT_ACTION.pattern + r")"
            r"|"
            r"(?:" + DIRECT_ACTION.pattern + r")"
            r".{0,95}"
            r"(?:" + PARTICIPANT.pattern + r")",
            clean,
            re.I
        )
    )


def event_context(clause):
    # Prevent unrelated clashes from becoming farmer events.
    clean = re.sub(r"#\S+", " ", clause)

    if re.search(
        r'\b(?:pakistan|islamists?|china|chinese|galwan)\b',
        clean,
        re.I
    ):
        return False

    if UNRELATED.search(clean):
        if not re.search(
            r'\b(?:farmers?|protesters?)\b'
            r'.{0,65}'
            r'\b(?:clash|attacked|injured|killed|march|protest)'
            r'|'
            r'\b(?:clash|attacked|injured|killed|march)\w*\b'
            r'.{0,65}'
            r'\b(?:farmers?|protesters?)\b',
            clean,
            re.I
        ):
            return False

    return explicit_participation(clean)


# --------------------------------------------------
# V4: EVENT EXTRACTION
# --------------------------------------------------

def extract_one(row):
    text = normalize(row['text'])
    post_id = str(row['post_id'])
    target = int(row['minilm_predicted_target'])
    probability = float(row['minilm_positive_probability'])

    results = []

    for clause_no, clause in enumerate(clauses(text)):
        primary_types, all_types = detect_types(clause)

        # Detect reported Rail Roko participation.
        if (
            not primary_types
            and re.search(r"\brail[ -]?roko\b", clause, re.I)
            and re.search(
                r"\b(?:took part|participated|joined|held|observed)\b",
                clause,
                re.I
            )
        ):
            primary_types = ['blockade']
            all_types = ['blockade']

        # Detect additional protest descriptions.
        if (
            not primary_types
            and re.search(
                r"\b(?:holding|held|staged)\b"
                r".{0,70}\bprotest\b",
                clause,
                re.I
            )
        ):
            primary_types = ['protest']
            all_types = ['protest']

        if not primary_types:
            continue

        status = status_for(clause)

        location, loc_status, mentions = extract_location(clause)

        related = event_context(clause)

        for typ in primary_types:

            if military_only(clause) or not related:
                decision = (
                    'exclude_unrelated_conflict'
                    if military_only(clause)
                    else 'needs_review'
                )

            elif status != 'reported_language':
                decision = 'needs_review'

            elif (
                re.search(
                    r"\b(?:ahead of|scheduled|plans? to|will|likely to)\b",
                    clause,
                    re.I
                )
                and re.search(
                    r"\b(?:deployed|march|protest|rally)\b",
                    clause,
                    re.I
                )
            ):
                decision = 'needs_review'

            elif (
                typ == 'protest'
                and re.search(
                    r"\b(?:just protesting|were protesting for their rights)\b",
                    clause,
                    re.I
                )
            ):
                decision = 'needs_review'

            elif (
                typ == 'protest'
                and re.search(
                    r"\b(?:violent attacks?|murder)\b",
                    clause,
                    re.I
                )
            ):
                decision = 'needs_review'

            elif (
                typ == 'march'
                and not re.search(
                    r"\b(?:held|staged|marched|marching|took out)\b",
                    clause,
                    re.I
                )
            ):
                decision = 'needs_review'

            else:
                # The classifier is supporting evidence,
                # not an automatic rejection rule.
                decision = 'event_candidate'

            results.append({
                'post_id': post_id,
                'text': str(row['text']),
                'post_date': row.get('date'),
                'clause_index': clause_no,
                'event_type': typ,
                'other_actions_in_clause': json.dumps(
                    [x for x in all_types if x != typ]
                ),
                'occurrence_status': status,
                'event_location': location,
                'location_status': loc_status,
                'location_mentions': json.dumps(mentions),
                'event_date': extract_date(clause),
                'event_evidence': clause[:600],
                'classifier_probability': probability,
                'classifier_predicted_target': target,
                'extraction_status': decision,
                'human_verified_event': False,
                'extraction_method': 'evidence_aware_v4',
                'incident_id': None,
                'deduplicated': False,
            })

    return results


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    print('=' * 66)
    print('STAGE 11.3 - EVIDENCE-AWARE EXTRACTION V4')
    print('=' * 66)

    if not SOURCE.exists():
        raise FileNotFoundError(
            f'Run Stage 10 first: {SOURCE}'
        )

    names = {
        'tweets': OUT / 'structured_event_candidates_v4.csv',
        'mentions': OUT / 'event_mentions_v4.csv',
        'candidates': OUT / 'extracted_events_v4.csv',
        'review': OUT / 'event_extraction_review_v4.csv',
        'audit': OUT / 'event_candidate_audit_v4.csv',
        'summary': OUT / 'event_extraction_summary_v4.json',
    }

    # Protect existing output files.
    for path in names.values():
        if path.exists():
            raise FileExistsError(
                f'Output exists, not overwriting: {path}'
            )

    df = pd.read_csv(
        SOURCE,
        dtype={'post_id': 'string'},
        low_memory=False
    )

    required = {
        'post_id',
        'text',
        'minilm_positive_probability',
        'minilm_predicted_target'
    }

    if required - set(df.columns):
        raise ValueError(
            f'Missing columns: {required - set(df.columns)}'
        )

    if (
        df['post_id'].isna().any()
        or df['post_id'].duplicated().any()
        or df['text'].isna().any()
    ):
        raise ValueError(
            'Missing/duplicate post_id or missing text'
        )

    p = pd.to_numeric(
        df['minilm_positive_probability'],
        errors='coerce'
    )

    y = pd.to_numeric(
        df['minilm_predicted_target'],
        errors='coerce'
    )

    if (
        p.isna().any()
        or not p.between(0, 1).all()
        or y.isna().any()
        or not y.isin([0, 1]).all()
    ):
        raise ValueError(
            'Invalid model probabilities or predictions'
        )

    df['minilm_positive_probability'] = p
    df['minilm_predicted_target'] = y.astype(int)

    mentions = []
    tweet_rows = []

    for _, row in df.iterrows():

        records = extract_one(row)
        mentions.extend(records)

        decisions = [
            r['extraction_status']
            for r in records
        ]

        priority = [
            'event_candidate',
            'needs_review',
            'classifier_negative_review',
            'exclude_unrelated_conflict'
        ]

        decision = next(
            (s for s in priority if s in decisions),
            (
                'insufficient_evidence'
                if int(row['minilm_predicted_target'])
                else 'not_selected_by_classifier'
            )
        )

        tweet_rows.append({
            'post_id': str(row['post_id']),
            'text': row['text'],
            'post_date': row.get('date'),
            'classifier_probability': float(
                row['minilm_positive_probability']
            ),
            'classifier_predicted_target': int(
                row['minilm_predicted_target']
            ),
            'event_mention_count': len(records),
            'extraction_status': decision,
            'event_types': json.dumps(
                list(dict.fromkeys(
                    r['event_type'] for r in records
                ))
            ),
            'human_verified_event': False,
        })

    mention_columns = [
        'post_id',
        'text',
        'post_date',
        'clause_index',
        'event_type',
        'other_actions_in_clause',
        'occurrence_status',
        'event_location',
        'location_status',
        'location_mentions',
        'event_date',
        'event_evidence',
        'classifier_probability',
        'classifier_predicted_target',
        'extraction_status',
        'human_verified_event',
        'extraction_method',
        'incident_id',
        'deduplicated'
    ]

    mdf = pd.DataFrame(
        mentions,
        columns=mention_columns
    )

    tdf = pd.DataFrame(tweet_rows)

    cdf = mdf[
        mdf.extraction_status.eq('event_candidate')
    ].copy()

    rdf = mdf[
        mdf.extraction_status.isin([
            'needs_review',
            'classifier_negative_review'
        ])
    ].copy()

    audit = tdf[
        tdf.post_id.isin(AUDIT_IDS)
    ].copy()

    audit['audit_candidate_number'] = audit.post_id.map({
        x: i for i, x in enumerate(AUDIT_IDS, 1)
    })

    audit = audit.sort_values(
        'audit_candidate_number'
    )

    audit['human_review_decision'] = ''
    audit['human_review_notes'] = ''

    report = {
        'stage': '11.3-v4',
        'total_tweets': len(tdf),
        'classifier_positive': int(y.eq(1).sum()),
        'event_mentions': len(mdf),
        'candidate_mentions': len(cdf),
        'candidate_tweets': int(
            tdf.extraction_status.eq('event_candidate').sum()
        ),
        'review_mentions': len(rdf),
        'known_audit_found': len(audit),
        'tweet_status_counts': (
            tdf.extraction_status.value_counts().to_dict()
        ),
        'mention_status_counts': (
            mdf.extraction_status.value_counts().to_dict()
        ),
        'event_type_counts': (
            cdf.event_type.value_counts().to_dict()
        ),
        'verified_incident_count': 0,
        'deduplicated': False,
        'labels_modified': False,
        'warning': (
            'Candidates are unverified. Clause rules can '
            'still miss mixed events; location is only a '
            'mention or possible location. Do not aggregate '
            'as confirmed incidents.'
        )
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    tdf.to_csv(
        names['tweets'],
        index=False,
        encoding='utf-8-sig'
    )

    mdf.to_csv(
        names['mentions'],
        index=False,
        encoding='utf-8-sig'
    )

    cdf.to_csv(
        names['candidates'],
        index=False,
        encoding='utf-8-sig'
    )

    rdf.to_csv(
        names['review'],
        index=False,
        encoding='utf-8-sig'
    )

    audit.to_csv(
        names['audit'],
        index=False,
        encoding='utf-8-sig'
    )

    names['summary'].write_text(
        json.dumps(report, indent=2),
        encoding='utf-8'
    )

    print(
        '\nTWEET-LEVEL STATUS:\n',
        tdf.extraction_status.value_counts().to_string()
    )

    print(
        '\nMENTION-LEVEL STATUS:\n',
        mdf.extraction_status.value_counts().to_string()
    )

    print(
        '\nCANDIDATE EVENT TYPES:\n',
        cdf.event_type.value_counts().to_string()
    )

    print(
        '\nKNOWN AUDIT CASES:\n',
        audit[
            [
                'audit_candidate_number',
                'extraction_status',
                'event_types'
            ]
        ].to_string(index=False)
    )

    print('\nSaved:')

    for path in names.values():
        print(path)

    print('\nNo source labels or old outputs were changed.')
    print('STAGE 11.3 COMPLETED SUCCESSFULLY')


if __name__ == '__main__':
    main()
