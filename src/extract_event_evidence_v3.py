
"""Stage 5 V3: rule-based civil-unrest event evidence extraction.

Input:  data/annotation/event_detection_test.csv
Output: data/annotation/structured_event_evidence_v3.csv

Manual labels are never read or modified by this script.
"""
from pathlib import Path
import html
import re
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / 'data' / 'annotation' / 'event_detection_test.csv'
OUTPUT_FILE = PROJECT_DIR / 'data' / 'annotation' / 'structured_event_evidence_v3.csv'

# Each pattern represents an event expression, not an isolated topic keyword.
EVENT_PATTERNS = {
    'clash': [
        r'\bclash(?:es|ed|ing)?\s+(?:with|between|amongst|amid)\b',
        r'\bclash(?:es)?\s+(?:at|during)\b',
        r'\b(?:violent|deadly)\s+clash(?:es)?\b',
        r'\b(?:police|farmers|protesters|demonstrators|media personnel)\s+clash(?:ed|ing)?\s+with\b',
    ],
    'injury_or_assault': [
        r'\b(?:cop|officer|police|farmer|farmers|protester|protesters|doctor|doctors|volunteer|volunteers|people|person)\b.{0,55}\b(?:injured|wounded|beaten|attacked|killed)\b',
        r'\b(?:injured|wounded|beaten|attacked|killed)\b.{0,55}\b(?:police|farmer|farmers|protester|protesters|doctor|doctors|volunteer|volunteers)\b',
        r'\b(?:brutally\s+)?beaten\s+by\s+police\b',
        r'\b(?:broken arms?|cracked skull)\b',
    ],
    'police_action': [
        r'\b(?:police|security forces|cops)\s+(?:fired|used|deployed)\s+(?:tear\s*gas|water cannons?|rubber bullets?)\b',
        r'\b(?:tear[ -]?gass?ed|lathi[ -]?charged)\b',
        r'\b(?:police|authorities)\s+(?:arrested|detained|booked|charged)\b',
        r'\b(?:police|authorities)\s+books?\s+(?:unknown\s+)?(?:people|persons|protesters|farmers)\b',
        r'\b(?:farmers|protesters|demonstrators)\s+(?:were|are|have been)\s+(?:arrested|detained|tear[ -]?gass?ed)\b',
        r'\b(?:fired|used)\s+tear\s*gas\s+shells?\b',
    ],
    'blockade_or_damage': [
        r'\b(?:farmers|protesters|demonstrators|activists)\s+(?:have\s+)?(?:blocked|block|occupied|stormed)\s+(?:the\s+)?(?:roads?|highways?|tracks?|railways?|buildings?|border|parliament)\b',
        r'\b(?:roads?|highways?|railway\s+tracks?)\s+(?:were|are|have been)\s+blocked\b',
        r'\b(?:set\s+fire\s+to|torched|vandalized|vandalised)\s+(?:vehicles?|buildings?|property)\b',
    ],
    'protest_activity': [
        r'\b(?:farmers|protesters|demonstrators|workers|students)\s+(?:held|staged|organized|organised|began|started|launched|joined|took part in)\s+(?:a\s+|the\s+)?(?:protest|rally|march|strike|demonstration|tractor\s+(?:rally|march|parade))\b',
        r'\b(?:protest|rally|march|strike|demonstration|tractor\s+(?:rally|march|parade))\s+(?:was|were|is|are)\s+(?:held|staged|organized|organised|underway)\b',
        r'\b(?:farmers|protesters|demonstrators)\s+(?:marched|gathered|assembled|protested)\s+(?:at|in|near|outside|towards?|to)\b',
        r'\b(?:today.s|historic|massive)\s+tractor\s+(?:march|rally|parade)\b',
        r'\btractor\s+(?:rally|march|parade)\s+(?:at|in|on)\b',
        r'\b(?:tractor\s+)?(?:march|rally|parade)\s+on\s+(?:jan(?:uary)?\s*)?26\b',
    ],
    'government_action': [
        r'\b(?:centre|center|government|pm|prime minister)\s+(?:decides?|decided|announces?|announced)\s+to\s+repeal\b',
        r'\b(?:farm\s+laws?)\s+(?:were|have been)\s+repealed\b',
        r'\b(?:chief minister|cm)\s+(?:cancels?|cancelled)\s+(?:a\s+|the\s+)?(?:farmers?\s+)?meet(?:ing)?\b',
    ],
    'support_action': [
        r'\b(?:temple|mandir|gurdwara|gurudwara)\b.{0,80}\b(?:offered|provided|opened)\b.{0,65}\b(?:facility|accommodation|shelter|food|meals?)\b',
        r'\b(?:preparing|prepared|serving|served|distributing|distributed)\s+(?:rotis?|food|meals?|langar)\b',
    ],
    'public_statement': [
        r'\b(?:rihanna|greta thunberg)\b.{0,90}\b(?:supported|supporting|tweeted|wrote)\b',
        r'\b(?:supported|supporting)\s+farmer\s+protests?\b',
    ],
}

PLANNED_PATTERNS = [
    r'\b(?:will|shall)\s+(?:hold|stage|organize|organise|observe|begin|start|march|protest)\b',
    r'\b(?:to|set to|scheduled to|planning to|plans? to)\s+(?:hold|stage|organize|organise|observe|begin|start|march|protest)\b',
    r'\b(?:will be|to be)\s+(?:held|organized|organised|observed)\b',
    r'\b(?:upcoming|planned|scheduled)\s+(?:protest|rally|march|strike|demonstration)\b',
    r'\bfarmers\s+are\s+coming\s+to\s+parliament\s+to\s+protest\b',
    r'\bfarmers\s+to\s+observe\b',
]

HISTORICAL_PATTERNS = [
    r'\b(?:yesterday|last\s+(?:week|month|year)|previously|earlier)\b',
    r'\b(?:had\s+(?:marched|protested|clashed))\b',
]

UNCERTAINTY_PATTERNS = [
    r'\b(?:allegedly|reportedly|unconfirmed|rumou?r|not\s+(?:100\s*%\s*)?sure)\b',
]

PROTEST_CONTEXT = re.compile(
    r'\b(?:farmers?|protest(?:ers?|s|ing)?|demonstrat(?:ion|ors?)|rally|march|tractor|police|strike|unrest|clash(?:es)?|riot(?:s)?)\b',
    re.I,
)

# Reject some clear negations of protest-activity expressions.
NEGATION = re.compile(
    r'\b(?:did\s+not|didn.t|never|fake|false\s+claim|no\s+evidence\s+of)\b',
    re.I
)


def clean_text(value):
    if pd.isna(value):
        return ''

    text = html.unescape(str(value))
    text = re.sub(r'https?://\S+|\b(?:www\.)\S+', ' ', text)
    text = re.sub(r'\[URL\]|\[USER\]', ' ', text, flags=re.I)

    return re.sub(r'\s+', ' ', text).strip()


def matches(text, patterns):
    hits = []

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            value = match.group(0)

            if value.lower() not in [h.lower() for h in hits]:
                hits.append(value)

    return hits


def analyze_post(row):
    text = clean_text(row.get('text', ''))

    evidence = {
        name: matches(text, patterns)
        for name, patterns in EVENT_PATTERNS.items()
    }

    categories = [
        name for name, hits in evidence.items() if hits
    ]

    actions = [
        hit for hits in evidence.values() for hit in hits
    ]

    planned = matches(text, PLANNED_PATTERNS)
    historical = matches(text, HISTORICAL_PATTERNS)
    uncertainty = matches(text, UNCERTAINTY_PATTERNS)

    context = bool(PROTEST_CONTEXT.search(text))

    incident_categories = {
        'clash',
        'injury_or_assault',
        'police_action',
        'blockade_or_damage'
    }

    has_incident = bool(
        incident_categories.intersection(categories)
    )

    has_activity = 'protest_activity' in categories

    has_other_action = bool(
        set(categories) & {
            'government_action',
            'support_action',
            'public_statement'
        }
    )

    if not text:
        decision = 'needs_review'
        status = 'unclear'
        reason = 'Empty post text'

    elif planned and not has_incident:
        decision = 'planned_or_future'
        status = 'planned_or_future'
        reason = 'Explicit future or planned action'

    elif has_incident and context:
        decision = 'event_candidate'
        status = (
            'historical' if historical
            else 'reported_occurrence'
        )
        reason = 'Concrete incident expression detected'

    elif has_activity and not NEGATION.search(text):
        decision = 'event_candidate'
        status = (
            'historical' if historical
            else 'reported_occurrence'
        )
        reason = 'Specific protest activity expression detected'

    elif has_other_action:
        decision = 'related_action'
        status = (
            'historical' if historical
            else 'reported_occurrence'
        )
        reason = (
            'Concrete related action, not necessarily '
            'a civil-unrest incident'
        )

    elif uncertainty and context:
        decision = 'needs_review'
        status = 'unclear'
        reason = 'Unverified or ambiguous claim'

    elif context:
        decision = 'non_event'
        status = 'no_specific_occurrence'
        reason = (
            'Topic or opinion without concrete event expression'
        )

    else:
        decision = 'non_event'
        status = 'no_specific_occurrence'
        reason = 'No concrete event expression'

    return {
        'topic_categories': '; '.join(categories),
        'action_evidence': '; '.join(actions),
        'consequence_evidence': '; '.join(
            evidence['injury_or_assault']
        ),
        'incident_detail_evidence': '; '.join(
            evidence['injury_or_assault']
            + evidence['police_action']
            + evidence['blockade_or_damage']
        ),
        'location_evidence': '',
        'time_evidence': '',
        'future_evidence': '; '.join(planned),
        'historical_evidence': '; '.join(historical),
        'explanatory_evidence': '',
        'demand_evidence': '',
        'actual_occurrence_evidence': decision in (
            'event_candidate', 'related_action'
        ),
        'occurrence_status': status,
        'event_evidence_decision': decision,
        'evidence_score': min(
            10,
            2 * len(categories) + int(has_incident) * 2
        ),
        'classifier_prediction': row.get(
            'predicted_label', ''
        ),
        'event_evidence_reason': reason,
    }


def main():
    print('=' * 65)
    print('CIVIL UNREST PROJECT - STAGE 5 V3')
    print('=' * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f'Input file not found: {INPUT_FILE}'
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={'post_id': 'string'}
    )

    required = {'post_id', 'text'}

    if not required.issubset(df.columns):
        raise ValueError(
            f'Missing columns: {sorted(required - set(df.columns))}'
        )

    if (
        df['post_id'].isna().any()
        or df['post_id'].duplicated().any()
    ):
        raise ValueError(
            'Missing or duplicate post_id values in input'
        )

    evidence = pd.DataFrame([
        analyze_post(row)
        for _, row in df.iterrows()
    ])

    # Prevent duplicate evidence columns.
    result = pd.concat(
        [
            df.drop(
                columns=evidence.columns,
                errors='ignore'
            ).reset_index(drop=True),

            evidence.reset_index(drop=True)
        ],
        axis=1
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding='utf-8-sig'
    )

    print('Input records:', len(result))

    print('\nEvidence decisions:')
    print(
        result['event_evidence_decision']
        .value_counts(dropna=False)
        .to_string()
    )

    if 'sampling_group' in result.columns:
        print('\nBy sampling group:')
        print(
            pd.crosstab(
                result['sampling_group'],
                result['event_evidence_decision']
            ).to_string()
        )

    print('\nSaved:', OUTPUT_FILE)
    print('STAGE 5 V3 COMPLETED')


if __name__ == '__main__':
    main()
