
from pathlib import Path
import json
import re
import html

import pandas as pd


# ============================================================
# STAGE 11.2 - EVIDENCE-AWARE EVENT EXTRACTION V2
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "evaluation"
    / "minilm_candidate_predictions_v1.csv"
)

OUTPUT_DIR = ROOT / "data" / "extracted"

ALL_FILE = OUTPUT_DIR / "structured_event_candidates_v2.csv"
EVENT_FILE = OUTPUT_DIR / "extracted_events_v2.csv"
REVIEW_FILE = OUTPUT_DIR / "event_extraction_review_v2.csv"
SUMMARY_FILE = OUTPUT_DIR / "event_extraction_summary_v2.json"
AUDIT_FILE = OUTPUT_DIR / "event_candidate_audit_v2.csv"


# ============================================================
# 1. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = html.unescape(str(text))

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    text = re.sub(
        r"\[(?:URL|USER)\]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def matches(text, pattern):

    return bool(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )
    )


def find_span(text, pattern):

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE
    )


def split_sentences(text):

    # Keep questions separate from reported actions.
    parts = re.split(
        r"(?<=[.!?])\s+|[\n]+",
        text
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


# ============================================================
# 2. EVENT TYPE RULES
#
# Patterns favor actions over generic topic keywords.
# Order is used only for displaying primary event type.
# ============================================================

EVENT_RULES = {
    "vehicle_attack": [
        r"\b(?:car|cars|vehicle|vehicles|suv|truck)\b.{0,70}\b(?:crushed|ran over|rammed|mowed down)\b",
        r"\b(?:crushed|ran over|rammed|mowed down)\b.{0,70}\b(?:farmer|farmers|protester|protesters|people)\b",
    ],
    "clash": [
        r"\bclash(?:es|ed|ing)?\b",
        r"\bscuffle(?:s)?\b",
        r"\b(?:farmers|protesters|demonstrators)\b.{0,80}\b(?:fought|confronted)\b.{0,40}\bpolice\b",
    ],
    "police_action": [
        r"\b(?:police|cops|security forces)\b.{0,100}\b(?:fired|deployed|used)\b.{0,30}\b(?:tear gas|water cannons?)\b",
        r"\b(?:tear gas|water cannons?|lathi charge|lathi charged|baton charge)\b",
        r"\b(?:police|cops)\b.{0,70}\b(?:arrested|detained|dispersed)\b.{0,50}\b(?:farmers|protesters|demonstrators)\b",
    ],
    "blockade": [
        r"\b(?:farmers|protesters|demonstrators)\b.{0,80}\b(?:blocked|blockaded)\b.{0,50}\b(?:road|highway|border|traffic)\b",
        r"\b(?:road|highway|border)\b.{0,40}\b(?:blocked|blockaded)\b",
    ],
    "march": [
        r"\b(?:farmers|protesters|demonstrators|workers)\b.{0,70}\b(?:marched|marching)\b",
        r"\b(?:marched|marching)\b.{0,70}\b(?:farmers|protesters|demonstrators|workers)\b",
        r"\b(?:took out|held|staged)\b.{0,30}\b(?:tractor rally|march|procession)\b",
        r"\btractor rally\b.{0,60}\b(?:held|took place|began|started)\b",
    ],
    "rally": [
        r"\b(?:held|staged|organized|organised)\b.{0,30}\brally\b",
        r"\brally\b.{0,30}\b(?:was held|took place|began)\b",
    ],
    "sit_in": [
        r"\b(?:held|staged|began|continued)\b.{0,30}\b(?:sit-in|dharna)\b",
        r"\b(?:sit-in|dharna)\b.{0,30}\b(?:held|continued|began)\b",
    ],
    "strike": [
        r"\b(?:workers|employees|farmers)\b.{0,70}\b(?:went on strike|staged a strike|observed a bandh)\b",
        r"\b(?:strike|bandh)\b.{0,40}\b(?:began|started|was observed|was held)\b",
    ],
    "protest": [
        r"\b(?:farmers|workers|students|protesters|demonstrators|people)\b.{0,90}\b(?:protesting|protested|demonstrating)\b",
        r"\b(?:protesting|protested|demonstrating)\b.{0,90}\b(?:farmers|workers|students|people)\b",
        r"\b(?:held|staged|joined|continued)\b.{0,30}\b(?:protest|demonstration|agitation)\b",
        r"\b(?:farmers|protesters)\b.{0,60}\b(?:gathered|assembled)\b",
    ],
}

EVENT_PRIORITY = list(EVENT_RULES.keys())


# ============================================================
# 3. CONTEXT RULES
# ============================================================

MILITARY_CONTEXT = [
    r"\b(?:soldiers|troops|army|armies|military)\b",
    r"\b(?:border war|armed forces|military clash)\b",
]

MILITARY_CONFLICT = [
    r"\b(?:china|chinese|pakistan|pakistani)\b.{0,100}\b(?:soldiers|troops|army|military|clash)\b",
    r"\b(?:soldiers|troops|army|military)\b.{0,100}\b(?:china|chinese|pakistan|pakistani)\b",
    r"\b(?:soldiers|troops)\b.{0,100}\b(?:killed|clashed|fought)\b",
]

FUTURE_PATTERNS = [
    r"\b(?:will|shall|going to)\s+(?:march|protest|hold|stage|organize|organise|block|join)\b",
    r"\b(?:planned|planning|plans|proposed|scheduled|upcoming)\b.{0,60}\b(?:march|rally|protest|strike|demonstration)\b",
    r"\b(?:march|rally|protest|strike)\b.{0,40}\b(?:planned|scheduled|tomorrow|next week)\b",
]

CANCEL_PATTERNS = [
    r"\b(?:march|rally|protest|strike|demonstration)\b.{0,60}\b(?:cancelled|canceled|postponed|called off)\b",
    r"\b(?:cancelled|canceled|postponed|called off)\b.{0,60}\b(?:march|rally|protest|strike|demonstration)\b",
]

HYPOTHETICAL_PATTERNS = [
    r"\b(?:might|may|could|would)\s+(?:march|protest|clash|attack|block|riot)\b",
    r"\blikely to\s+(?:march|protest|clash|attack|block|riot)\b",
    r"\b(?:rumou?r|unconfirmed)\b",
]

NEGATION_PATTERNS = [
    r"\b(?:did not|didn't|never)\s+(?:march|protest|clash|attack|happen|occur)\b",
    r"\bno\s+(?:clash|protest|violence|incident)\b",
]

HISTORICAL_PATTERNS = [
    r"\b(?:years ago|last year|anniversary|in memory of)\b",
]

VAGUE_PATTERNS = [
    r"\b(?:where are|what if|why would|did they|is there)\b",
    r"\b(?:should|must|ought to)\s+(?:protest|march|rally)\b",
]


# ============================================================
# 4. LOCATION RECOGNITION
#
# A location mention is NOT automatically an event location.
# ============================================================

LOCATIONS = {
    "New Delhi": [r"\bnew delhi\b"],
    "Delhi": [r"\bdelhi\b", r"\bdehli\b"],
    "ITO, Delhi": [r"\bito intersection\b", r"\bcentral delhi\b"],
    "Ghazipur Border": [
        r"\bghazipur border\b",
        r"\bgazipur border\b",
    ],
    "Singhu Border": [r"\bsinghu(?: border)?\b"],
    "Tikri Border": [r"\btikri(?: border)?\b"],
    "Ghaziabad": [r"\bghaziabad\b"],
    "Punjab": [r"\bpunjab\b"],
    "Haryana": [r"\bharyana\b"],
    "Uttar Pradesh": [r"\buttar pradesh\b"],
    "Uttarakhand": [r"\buttarakhand\b"],
    "Mumbai": [r"\bmumbai\b"],
    "Kolkata": [r"\bkolkata\b"],
    "Bengaluru": [r"\bbengaluru\b", r"\bbangalore\b"],
    "Chandigarh": [r"\bchandigarh\b"],
    "Downing Street": [r"\bdowning street\b"],
}

# Do not infer coordinates or geocode at this stage.


def extract_locations(sentence):

    found = []

    for name, patterns in LOCATIONS.items():

        for pattern in patterns:

            match = find_span(sentence, pattern)

            if match:
                found.append({
                    "name": name,
                    "start": match.start(),
                    "end": match.end(),
                })
                break

    # Prefer specific locations over broader overlapping names.
    specific = [
        "New Delhi",
        "ITO, Delhi",
        "Ghazipur Border",
    ]

    if any(
        item["name"] in specific
        for item in found
    ):
        found = [
            item for item in found
            if item["name"] != "Delhi"
        ]

    return found


def location_role(sentence, location):

    name = location["name"]

    patterns = LOCATIONS[name]

    for pattern in patterns:

        escaped = pattern

        # Directional context: destination, not occurrence.
        if matches(
            sentence,
            rf"\b(?:towards|toward|to|heading to|en route to)\s+{escaped}"
        ):
            return "destination"

    if matches(
        sentence,
        r"\b(?:compared with|compared to|like in)\b"
    ):
        return "contextual_mention"

    return "mentioned_location"


def resolve_location(sentence):

    found = extract_locations(sentence)

    if not found:
        return None, "missing", []

    names = list(dict.fromkeys(
        item["name"] for item in found
    ))

    if len(names) > 1:
        return None, "multiple_mentions", names

    role = location_role(
        sentence,
        found[0]
    )

    if role == "destination":
        return None, "destination_only", names

    if role == "contextual_mention":
        return None, "contextual_mention", names

    # Mentioned_location means a possible location,
    # not a verified event location.
    return names[0], "mentioned_location", names


# ============================================================
# 5. EVENT OCCURRENCE STATUS
# ============================================================

def occurrence_status(sentence):

    sentence = sentence.strip()

    if "?" in sentence or matches(
        sentence,
        r"^(?:where|when|why|how|did|does|is|are|what)\b"
    ):
        return "question_or_uncertain"

    if any(
        matches(sentence, p)
        for p in CANCEL_PATTERNS
    ):
        return "cancelled_or_postponed"

    if any(
        matches(sentence, p)
        for p in NEGATION_PATTERNS
    ):
        return "negated"

    if any(
        matches(sentence, p)
        for p in HYPOTHETICAL_PATTERNS
    ):
        return "hypothetical"

    if any(
        matches(sentence, p)
        for p in FUTURE_PATTERNS
    ):
        return "planned_or_future"

    if any(
        matches(sentence, p)
        for p in HISTORICAL_PATTERNS
    ):
        return "historical_reference"

    if any(
        matches(sentence, p)
        for p in VAGUE_PATTERNS
    ):
        return "uncertain"

    return "reported_occurrence"


# ============================================================
# 6. EXTRACT EVENT TYPES FROM ONE SENTENCE
# ============================================================

def detect_event_types(sentence):

    detected = []

    for event_type, patterns in EVENT_RULES.items():

        if any(
            matches(sentence, pattern)
            for pattern in patterns
        ):
            detected.append(event_type)

    # Clash is more specific than a generic protest,
    # but both can be retained as separate event mentions.
    return detected


def is_unrelated_military_conflict(sentence):

    if not any(
        matches(sentence, p)
        for p in MILITARY_CONTEXT
    ):
        return False

    return any(
        matches(sentence, p)
        for p in MILITARY_CONFLICT
    )


# ============================================================
# 7. EVENT DATE
# ============================================================

def extract_explicit_date(sentence):

    match = re.search(
        r"\b20\d{2}-\d{2}-\d{2}\b",
        sentence
    )

    if not match:
        return None

    try:
        date = pd.to_datetime(
            match.group(0),
            format="%Y-%m-%d",
            errors="raise"
        )

        return date.strftime("%Y-%m-%d")

    except (ValueError, TypeError):
        return None


# ============================================================
# 8. EXTRACT EVENT MENTIONS FROM ONE TWEET
# ============================================================

def extract_tweet(row):

    post_id = str(row["post_id"])

    text = normalize_text(
        row["text"]
    )

    probability = float(
        row["minilm_positive_probability"]
    )

    predicted_target = int(
        row["minilm_predicted_target"]
    )

    sentences = split_sentences(text)

    mentions = []

    for sentence_index, sentence in enumerate(sentences):

        event_types = detect_event_types(
            sentence
        )

        if not event_types:
            continue

        status = occurrence_status(
            sentence
        )

        military = is_unrelated_military_conflict(
            sentence
        )

        location, location_status, location_mentions = (
            resolve_location(sentence)
        )

        explicit_date = extract_explicit_date(
            sentence
        )

        for event_type in event_types:

            if military:
                decision = "exclude_unrelated_conflict"

            elif status != "reported_occurrence":
                decision = "needs_review"

            elif predicted_target != 1:
                decision = "classifier_negative_review"

            elif event_type == "protest" and matches(
                sentence,
                r"\b(?:just protesting|were protesting for their rights)\b"
            ):
                decision = "needs_review"

            elif location_status in [
                "multiple_mentions",
                "destination_only",
                "contextual_mention",
            ]:
                decision = "needs_review"

            else:
                decision = "event_candidate"

            mentions.append({
                "post_id": post_id,
                "text": text,
                "post_date": row.get("date", None),
                "sentence_index": sentence_index,
                "event_type": event_type,
                "occurrence_status": status,
                "event_location": location,
                "location_status": location_status,
                "location_mentions": json.dumps(
                    location_mentions,
                    ensure_ascii=False
                ),
                "event_date": explicit_date,
                "event_evidence": sentence[:600],
                "classifier_probability": probability,
                "classifier_predicted_target": predicted_target,
                "extraction_status": decision,
                "human_verified_event": False,
                "extraction_method": "evidence_aware_v2",
            })

    return mentions


# ============================================================
# 9. SUMMARIZE EACH TWEET
# ============================================================

def summarize_tweet(row, mentions):

    decisions = [
        item["extraction_status"]
        for item in mentions
    ]

    if "event_candidate" in decisions:
        status = "event_candidate"

    elif "needs_review" in decisions:
        status = "needs_review"

    elif "classifier_negative_review" in decisions:
        status = "classifier_negative_review"

    elif "exclude_unrelated_conflict" in decisions:
        status = "exclude_unrelated_conflict"

    elif int(row["minilm_predicted_target"]) == 1:
        status = "insufficient_evidence"

    else:
        status = "not_selected_by_classifier"

    return {
        "post_id": str(row["post_id"]),
        "text": str(row["text"]),
        "post_date": row.get("date", None),
        "classifier_probability": float(
            row["minilm_positive_probability"]
        ),
        "classifier_predicted_target": int(
            row["minilm_predicted_target"]
        ),
        "event_mention_count": len(mentions),
        "event_types": json.dumps(
            list(dict.fromkeys(
                item["event_type"]
                for item in mentions
            ))
        ),
        "extraction_status": status,
        "human_verified_event": False,
    }


# ============================================================
# 10. AUDIT KNOWN PROBLEM CASES
# ============================================================

KNOWN_AUDIT_IDS = [
    "1426909019340369920",
    "1345603554568859648",
    "1354248484937986049",
    "1458053635087142918",
    "1353995986679558144",
    "1353995755929788416",
    "1445690167625277444",
    "1331975238939070465",
    "1395585442163007489",
    "1445248697181216768",
    "1354073743627149312",
    "1453480427755352066",
    "1333882880691417088",
    "1354284290515861505",
    "1355177937595555845",
    "1354274850521575427",
    "1387288908711354369",
    "1336021209260019712",
    "1445964306349117440",
]


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 11.2 - EVIDENCE-AWARE EVENT EXTRACTION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    outputs = [
        ALL_FILE,
        EVENT_FILE,
        REVIEW_FILE,
        SUMMARY_FILE,
        AUDIT_FILE,
    ]

    for output in outputs:
        if output.exists():
            raise FileExistsError(
                f"Existing output preserved: {output}"
            )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    required = {
        "post_id",
        "text",
        "minilm_positive_probability",
        "minilm_predicted_target",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing input columns: {missing}"
        )

    if df["post_id"].isna().any():
        raise ValueError("Missing post IDs.")

    if df["post_id"].duplicated().any():
        raise ValueError("Duplicate post IDs.")

    if df["text"].isna().any():
        raise ValueError("Missing tweet text.")

    probabilities = pd.to_numeric(
        df["minilm_positive_probability"],
        errors="coerce"
    )

    targets = pd.to_numeric(
        df["minilm_predicted_target"],
        errors="coerce"
    )

    if (
        probabilities.isna().any()
        or not probabilities.between(0, 1).all()
    ):
        raise ValueError(
            "Invalid classifier probabilities."
        )

    if (
        targets.isna().any()
        or not targets.isin([0, 1]).all()
    ):
        raise ValueError(
            "Invalid classifier predictions."
        )

    df["minilm_positive_probability"] = probabilities
    df["minilm_predicted_target"] = targets.astype(int)

    print("\nInput tweets:", len(df))

    all_mentions = []
    tweet_summaries = []

    for _, row in df.iterrows():

        mentions = extract_tweet(row)

        all_mentions.extend(mentions)

        tweet_summaries.append(
            summarize_tweet(row, mentions)
        )

    summary_df = pd.DataFrame(tweet_summaries)

    mention_columns = [
        "post_id",
        "text",
        "post_date",
        "sentence_index",
        "event_type",
        "occurrence_status",
        "event_location",
        "location_status",
        "location_mentions",
        "event_date",
        "event_evidence",
        "classifier_probability",
        "classifier_predicted_target",
        "extraction_status",
        "human_verified_event",
        "extraction_method",
    ]

    mentions_df = pd.DataFrame(
        all_mentions,
        columns=mention_columns
    )

    events_df = mentions_df[
        mentions_df["extraction_status"]
        == "event_candidate"
    ].copy()

    review_df = mentions_df[
        mentions_df["extraction_status"].isin([
            "needs_review",
            "classifier_negative_review",
        ])
    ].copy()

    audit_df = summary_df[
        summary_df["post_id"].isin(
            KNOWN_AUDIT_IDS
        )
    ].copy()

    audit_df["audit_candidate_number"] = (
        audit_df["post_id"]
        .map({
            post_id: i
            for i, post_id in enumerate(
                KNOWN_AUDIT_IDS,
                start=1
            )
        })
    )

    audit_df = audit_df.sort_values(
        "audit_candidate_number"
    )

    print("\nTWEET-LEVEL EXTRACTION STATUS:")
    print(
        summary_df["extraction_status"]
        .value_counts()
        .to_string()
    )

    print("\nEVENT MENTION DECISIONS:")
    print(
        mentions_df["extraction_status"]
        .value_counts()
        .to_string()
    )

    print("\nEVENT CANDIDATE TYPES:")
    print(
        events_df["event_type"]
        .value_counts()
        .to_string()
    )

    print("\nKNOWN 19-CANDIDATE AUDIT:")
    print(
        audit_df[
            [
                "audit_candidate_number",
                "extraction_status",
                "event_types",
                "event_mention_count",
            ]
        ].to_string(index=False)
    )

    report = {
        "stage": "11.2",
        "method": "evidence_aware_v2",
        "total_tweets": int(len(df)),
        "classifier_positive": int(
            (df["minilm_predicted_target"] == 1).sum()
        ),
        "event_mentions": int(len(mentions_df)),
        "candidate_event_mentions": int(len(events_df)),
        "candidate_tweets": int(
            summary_df["extraction_status"]
            .eq("event_candidate")
            .sum()
        ),
        "review_mentions": int(len(review_df)),
        "known_audit_tweets_found": int(len(audit_df)),
        "tweet_status_counts": {
            str(k): int(v)
            for k, v in summary_df[
                "extraction_status"
            ].value_counts().items()
        },
        "event_type_counts": {
            str(k): int(v)
            for k, v in events_df[
                "event_type"
            ].value_counts().items()
        },
        "human_verified_events": 0,
        "source_labels_modified": False,
        "note": (
            "Automatic candidate extraction only. "
            "No factual verification or geocoding. "
            "Sentence-level rules may still mix distinct "
            "clauses, duplicate events, or misattribute "
            "locations and occurrence statuses."
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df.to_csv(
        ALL_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    events_df.to_csv(
        EVENT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    review_df.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    audit_df.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4
        )

    print("\nOUTPUT FILES:")
    for output in outputs:
        print(output)

    print("\nOriginal labels were NOT modified.")
    print("STAGE 11.2 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
