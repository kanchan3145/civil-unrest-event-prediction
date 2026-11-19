
from pathlib import Path
import json
import re

import pandas as pd


# ============================================================
# STAGE 11 - STRUCTURED CIVIL UNREST EVENT EXTRACTION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "evaluation"
    / "minilm_candidate_predictions_v1.csv"
)

OUTPUT_DIR = ROOT / "data" / "extracted"

ALL_OUTPUT = (
    OUTPUT_DIR / "structured_event_candidates_v1.csv"
)

EVENT_OUTPUT = (
    OUTPUT_DIR / "extracted_events_v1.csv"
)

REVIEW_OUTPUT = (
    OUTPUT_DIR / "event_extraction_review_v1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "event_extraction_summary_v1.json"
)


# ============================================================
# 1. EVENT PATTERNS
# ============================================================

EVENT_PATTERNS = {
    "clash": [
        r"\bclash(?:es|ed|ing)?\b",
        r"\bviolent confrontation\b",
        r"\bscuffle(?:s)?\b",
    ],
    "police_action": [
        r"\btear[\s-]?gas\b",
        r"\blathi[\s-]?charg(?:e|ed|ing)\b",
        r"\bbaton[\s-]?charg(?:e|ed|ing)\b",
        r"\bpolice (?:fired|arrested|detained|beat|attacked)\b",
    ],
    "blockade": [
        r"\bblockad(?:e|ed|ing)\b",
        r"\broad(?:s)? blocked\b",
        r"\bblocked (?:the )?(?:road|highway|border)\b",
        r"\broadblock(?:s)?\b",
    ],
    "march": [
        r"\bmarch(?:es|ed|ing)?\b",
        r"\btractor rall(?:y|ies)\b",
        r"\bprocession(?:s)?\b",
    ],
    "rally": [
        r"\brall(?:y|ies)\b",
        r"\bdemonstration(?:s)?\b",
    ],
    "sit_in": [
        r"\bsit[\s-]?in(?:s)?\b",
        r"\bdharna\b",
    ],
    "strike": [
        r"\bstrike(?:s)?\b",
        r"\bstriking workers\b",
        r"\bbandh\b",
        r"\brail roko\b",
    ],
    "protest": [
        r"\bprotest(?:s|ed|ers|ing)?\b",
        r"\bagitat(?:ion|ions|ing|ed|ors?)\b",
    ],
}

# Stronger evidence of a described action.
# These patterns do not by themselves verify an event.
ACTION_PATTERNS = [
    r"\b(?:held|staged|organized|organised)\s+(?:a |the )?(?:protest|rally|march|demonstration)\b",
    r"\b(?:gathered|assembled|converged)\b",
    r"\b(?:blocked|blockaded|occupied)\b",
    r"\b(?:clashed|arrested|detained|injured)\b",
    r"\b(?:fired|deployed)\s+tear[\s-]?gas\b",
    r"\b(?:marched|marching)\b",
    r"\b(?:protesting|demonstrating|agitating)\b",
    r"\b(?:pelting|pelted)\s+stones\b",
    r"\b(?:ran|drove)\s+(?:a |the )?tractor\s+over\b",
]

PLANNED_PATTERNS = [
    r"\b(?:will|shall|going to)\s+(?:hold|stage|organize|organise|begin|start|launch|join|march|protest|block)\b",
    r"\b(?:plan|plans|planned|planning)\s+to\b",
    r"\b(?:scheduled|expected|set)\s+to\b",
    r"\b(?:proposed|upcoming)\s+(?:march|protest|rally|strike)\b",
    r"\b(?:tomorrow|next week|next month)\b",
]

HYPOTHETICAL_PATTERNS = [
    r"\b(?:might|may|could|would)\s+(?:protest|clash|march|block|attack|detain)\b",
    r"\b(?:if|unless)\s+.{0,80}\b(?:protest|clash|riot|march)\b",
    r"\blikely to\b",
    r"\b(?:rumou?r|unconfirmed|allegedly)\b",
]

NEGATED_PATTERNS = [
    r"\b(?:did not|didn't|never)\s+(?:happen|occur|take place|protest|march|clash)\b",
    r"\b(?:no|without)\s+(?:protest|clash|violence|incident)\b",
    r"\b(?:cancelled|canceled|postponed|called off)\b",
]

HISTORICAL_PATTERNS = [
    r"\b(?:last year|years ago|months ago)\b",
    r"\b(?:anniversary|in memory of)\b",
    r"\b(?:previously|earlier this year)\b",
]


# ============================================================
# 2. LOCATION PATTERNS
# ============================================================

# Small explicit gazetteer for initial India-focused data.
# This is not a full geographic NER system.

LOCATION_PATTERNS = {
    "New Delhi": [
        r"\bnew delhi\b",
    ],
    "Delhi": [
        r"\bdelhi\b",
    ],
    "Ghaziabad": [
        r"\bghaziabad\b",
        r"\bghazipur(?: border)?\b",
        r"\bgazipur(?: border)?\b",
    ],
    "Singhu Border": [
        r"\bsinghu(?: border)?\b",
    ],
    "Tikri Border": [
        r"\btikri(?: border)?\b",
    ],
    "Punjab": [
        r"\bpunjab\b",
    ],
    "Haryana": [
        r"\bharyana\b",
    ],
    "Uttar Pradesh": [
        r"\buttar pradesh\b",
    ],
    "Uttarakhand": [
        r"\buttarakhand\b",
    ],
    "Bajpur": [
        r"\bbajpur\b",
    ],
    "Mumbai": [
        r"\bmumbai\b",
    ],
    "Kolkata": [
        r"\bkolkata\b",
    ],
    "Bengaluru": [
        r"\bbengaluru\b",
        r"\bbangalore\b",
    ],
    "Chandigarh": [
        r"\bchandigarh\b",
    ],
}

# Avoid treating these common social-media phrases
# as evidence of a specific event location.
LOCATION_CONTEXT_WARNINGS = [
    r"\b(?:across|throughout)\s+india\b",
    r"\b(?:in|from)\s+other countries\b",
    r"\b(?:compared to|like|similar to)\b",
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def find_matches(text, patterns):

    matches = []

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            matches.append(match.group(0))

    return matches


def split_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def identify_event_type(text):

    found_types = []

    for event_type, patterns in EVENT_PATTERNS.items():

        if find_matches(text, patterns):
            found_types.append(event_type)

    if not found_types:
        return None, []

    # The dictionary order gives specific categories
    # precedence over the generic protest category.
    return found_types[0], found_types


def identify_location(text):

    matches = []

    for location, patterns in LOCATION_PATTERNS.items():

        if find_matches(text, patterns):
            matches.append(location)

    # Suppress the broader Delhi label when
    # New Delhi is explicitly mentioned.
    if "New Delhi" in matches and "Delhi" in matches:
        matches.remove("Delhi")

    if not matches:
        return None, "missing", []

    if len(matches) > 1:
        return None, "ambiguous", matches

    if find_matches(text, LOCATION_CONTEXT_WARNINGS):
        return matches[0], "needs_review", matches

    return matches[0], "explicit_mention", matches


def identify_occurrence_status(text):

    # These are conservative linguistic indicators,
    # not factual verification.

    if find_matches(text, NEGATED_PATTERNS):
        return "negated_or_cancelled"

    if find_matches(text, HYPOTHETICAL_PATTERNS):
        return "hypothetical_or_uncertain"

    if find_matches(text, PLANNED_PATTERNS):
        return "planned_or_future"

    if find_matches(text, HISTORICAL_PATTERNS):
        return "historical_report"

    if find_matches(text, ACTION_PATTERNS):
        return "reported_occurrence"

    return "uncertain"


def extract_event_date(text):

    # Only extract explicit ISO-style calendar dates.
    # Relative dates such as "yesterday" are not
    # resolved without reliable event-time context.

    match = re.search(
        r"\b(20\d{2}-\d{2}-\d{2})\b",
        text,
    )

    if not match:
        return None, "not_explicitly_resolved"

    candidate = match.group(1)

    try:
        parsed = pd.to_datetime(
            candidate,
            format="%Y-%m-%d",
            errors="raise",
        )

        return (
            parsed.strftime("%Y-%m-%d"),
            "explicit_date_mention",
        )

    except (ValueError, TypeError):
        return None, "invalid_date_mention"


def extract_evidence(text):

    sentences = split_sentences(text)

    # Prefer a sentence that contains both an event
    # term and a stronger action indicator.
    for sentence in sentences:

        event_type, _ = identify_event_type(sentence)

        if event_type and find_matches(
            sentence,
            ACTION_PATTERNS,
        ):
            return sentence[:500]

    # Otherwise retain a sentence containing
    # an event term as weaker evidence.
    for sentence in sentences:

        event_type, _ = identify_event_type(sentence)

        if event_type:
            return sentence[:500]

    return None


# ============================================================
# 4. EXTRACT ONE TWEET
# ============================================================

def extract_one(row):

    text = str(row["text"])

    probability = float(
        row["minilm_positive_probability"]
    )

    predicted_target = int(
        row["minilm_predicted_target"]
    )

    event_type, all_types = identify_event_type(text)

    event_location, location_status, locations = (
        identify_location(text)
    )

    occurrence_status = identify_occurrence_status(
        text
    )

    event_date, date_status = extract_event_date(
        text
    )

    event_evidence = extract_evidence(text)

    has_action_evidence = bool(
        find_matches(text, ACTION_PATTERNS)
    )

    has_event_term = event_type is not None

    # A positive classifier prediction is necessary,
    # but not sufficient, for automatic candidate extraction.
    if predicted_target != 1:

        extraction_status = "not_selected_by_classifier"

    elif not has_event_term:

        extraction_status = "insufficient_evidence"

    elif occurrence_status != "reported_occurrence":

        extraction_status = "needs_review"

    elif not has_action_evidence:

        extraction_status = "needs_review"

    elif location_status in [
        "ambiguous",
        "needs_review",
    ]:

        extraction_status = "needs_review"

    else:

        extraction_status = "event_candidate"

    # Even event_candidate is not a verified event.
    # Missing event dates and locations remain missing.

    return {
        "post_id": str(row["post_id"]),
        "text": text,
        "post_date": row.get("date", None),
        "classifier_probability": probability,
        "classifier_predicted_target": predicted_target,
        "event_type": event_type,
        "all_event_types": json.dumps(all_types),
        "event_location": event_location,
        "location_status": location_status,
        "location_mentions": json.dumps(locations),
        "event_date": event_date,
        "event_date_status": date_status,
        "occurrence_status": occurrence_status,
        "has_action_evidence": has_action_evidence,
        "event_evidence": event_evidence,
        "extraction_status": extraction_status,
        "human_verified_event": False,
        "extraction_method": "lightweight_rule_based_v1",
    }


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 11 - STRUCTURED EVENT EXTRACTION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    # Preserve previous experiment outputs.
    for path in [
        ALL_OUTPUT,
        EVENT_OUTPUT,
        REVIEW_OUTPUT,
        SUMMARY_OUTPUT,
    ]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    print("\nLoading MiniLM predictions...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"},
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
            f"Missing columns: {missing}"
        )

    if df["post_id"].isna().any():
        raise ValueError("Missing post IDs.")

    if df["post_id"].duplicated().any():
        raise ValueError("Duplicate post IDs.")

    if df["text"].isna().any():
        raise ValueError("Missing tweet text.")

    probabilities = pd.to_numeric(
        df["minilm_positive_probability"],
        errors="coerce",
    )

    targets = pd.to_numeric(
        df["minilm_predicted_target"],
        errors="coerce",
    )

    if (
        probabilities.isna().any()
        or (~probabilities.between(0, 1)).any()
    ):
        raise ValueError(
            "Invalid classifier probabilities."
        )

    if (
        targets.isna().any()
        or (~targets.isin([0, 1])).any()
    ):
        raise ValueError(
            "Invalid classifier targets."
        )

    df["minilm_positive_probability"] = probabilities
    df["minilm_predicted_target"] = targets.astype(int)

    print("Input tweets:", len(df))

    print("\nExtracting event information...")

    records = []

    for _, row in df.iterrows():
        records.append(extract_one(row))

    result = pd.DataFrame(records)

    events = result[
        result["extraction_status"] == "event_candidate"
    ].copy()

    review = result[
        result["extraction_status"].isin([
            "needs_review",
            "insufficient_evidence",
        ])
    ].copy()

    print("\nEXTRACTION STATUS DISTRIBUTION:")
    print(
        result["extraction_status"]
        .value_counts()
        .to_string()
    )

    print("\nEVENT TYPE DISTRIBUTION:")
    print(
        events["event_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nOCCURRENCE STATUS DISTRIBUTION:")
    print(
        result["occurrence_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nLOCATION STATUS DISTRIBUTION:")
    print(
        result["location_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    summary = {
        "stage": "11",
        "method": "lightweight_rule_based_v1",
        "total_tweets": int(len(result)),
        "classifier_positive": int(
            (result["classifier_predicted_target"] == 1).sum()
        ),
        "event_candidates": int(len(events)),
        "review_cases": int(len(review)),
        "not_selected_by_classifier": int(
            (
                result["extraction_status"]
                == "not_selected_by_classifier"
            ).sum()
        ),
        "extraction_status_counts": {
            str(k): int(v)
            for k, v in result[
                "extraction_status"
            ].value_counts().items()
        },
        "human_verified_event_records": 0,
        "source_labels_modified": False,
        "note": (
            "Event candidates are not confirmed incidents. "
            "Event dates and locations require independent "
            "validation. Regex rules may miss or misclassify "
            "contextual and multi-event tweets."
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        ALL_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    events.to_csv(
        EVENT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    review.to_csv(
        REVIEW_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\nOUTPUT FILES:")
    print(ALL_OUTPUT)
    print(EVENT_OUTPUT)
    print(REVIEW_OUTPUT)
    print(SUMMARY_OUTPUT)

    print("\nOriginal labels were NOT modified.")
    print("STAGE 11 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
