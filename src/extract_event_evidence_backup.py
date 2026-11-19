
"""
Civil Unrest Event Extraction and Prediction from Social Media Data

STAGE 5: IMPROVED STRUCTURED EVENT EVIDENCE EXTRACTION

Input:
    data/annotation/event_detection_test.csv

Output:
    data/annotation/structured_event_evidence_v2.csv
"""

from pathlib import Path
import re
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "annotation"
    / "event_detection_test.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "data" / "annotation"
    / "structured_event_evidence_v2.csv"
)


# --------------------------------------------------
# 2. EVIDENCE PATTERNS
# --------------------------------------------------

CATEGORY_PATTERNS = {
    "protest": [
        r"\bprotest(?:s|ed|ing|er|ers)?\b",
        r"\bdemonstration(?:s)?\b"
    ],
    "strike": [
        r"\bstrike(?:s|d|ing)?\b",
        r"\bbharat bandh\b",
        r"\bchakka jam\b"
    ],
    "march": [
        r"\bmarch(?:es|ed|ing)?\b",
        r"\brally(?:ies)?\b"
    ],
    "clash": [
        r"\bclash(?:es|ed|ing)?\b",
        r"\bconfrontation(?:s)?\b"
    ],
    "riot": [
        r"\briot(?:s|ed|ing)?\b",
        r"\blooting\b",
        r"\bvandalism\b"
    ]
}

# Concrete actions associated with an incident.
ACTION_PATTERNS = [
    r"\bheld a protest\b",
    r"\bstaged a protest\b",
    r"\bprotested\b",
    r"\bmarched\b",
    r"\bheld a march\b",
    r"\bwent on strike\b",
    r"\bstrike observed\b",
    r"\bclashed\b",
    r"\battacked\b",
    r"\bblocked (?:the )?road\b",
    r"\bblocked roads\b",
    r"\bbroke through\b",
    r"\bstormed\b",
    r"\boccupied\b",
    r"\bset fire\b",
    r"\bpolice fired\b",
    r"\btear gas shells?\b",
    r"\bwater cannons?\b",
    r"\bpolice lathi.?charged\b",
    r"\blathi.?charged\b",
    r"\blooted\b",
    r"\bset vehicles? on fire\b"
]

# Concrete consequences or incident details.
CONSEQUENCE_PATTERNS = [
    r"\binjured\b",
    r"\bwounded\b",
    r"\bkilled\b",
    r"\bdied\b",
    r"\bdead\b",
    r"\bhospitali[sz]ed\b",
    r"\barrested\b",
    r"\bdetained\b",
    r"\bdamaged\b",
    r"\bvandalism\b",
    r"\btear gas\b",
    r"\blathi.?charge\b",
    r"\bproperty damage\b",
    r"\bpolice personnel injured\b",
    r"\bvehicles? (?:were )?burnt\b",
    r"\bvehicles? (?:were )?burned\b"
]

# Strong incident details that can support a report
# even when the text does not mention a protest category.
INCIDENT_DETAIL_PATTERNS = [
    r"\binjured\b",
    r"\bwounded\b",
    r"\bkilled\b",
    r"\bdied\b",
    r"\barrested\b",
    r"\bdetained\b",
    r"\bpolice fired\b",
    r"\btear gas\b",
    r"\blathi.?charg(?:e|ed)\b",
    r"\bblocked (?:the )?road\b",
    r"\bblocked roads\b",
    r"\bproperty damage\b",
    r"\bset fire\b",
    r"\blooted\b",
    r"\bvehicles? (?:were )?burnt\b",
    r"\bvehicles? (?:were )?burned\b"
]

FUTURE_PATTERNS = [
    r"\bwill be (?:held|organised|organized|observed)\b",
    r"\bwill take place\b",
    r"\bwill begin\b",
    r"\bwill start\b",
    r"\bscheduled to\b",
    r"\bplanned to\b",
    r"\bhas been planned\b",
    r"\bgoing to (?:hold|organise|organize|stage)\b",
    r"\bto be held\b",
    r"\bupcoming\b"
]

HISTORICAL_PATTERNS = [
    r"\blast year\b",
    r"\blast month\b",
    r"\blast week\b",
    r"\byesterday\b",
    r"\bpreviously\b",
    r"\bearlier\b",
    r"\bthe day we marched\b",
    r"\bwhen we protested\b",
    r"\bwe had marched\b",
    r"\bwe had protested\b"
]

EXPLANATORY_PATTERNS = [
    r"\bhope this clarifies\b",
    r"\bto clarify\b",
    r"\bfor anyone wondering\b",
    r"\bexplains why\b",
    r"\bthe reason is\b"
]

DEMAND_PATTERNS = [
    r"\bshould\b",
    r"\bmust\b",
    r"\bneed to\b",
    r"\bdemand(?:s|ed)?\b",
    r"\bsupport\b",
    r"\bstand with\b",
    r"\bappeal to\b"
]

# Only recognize explicit location expressions.
# These are still textual clues, not verified geographic locations.
LOCATION_PATTERNS = [
    r"\bin front of the [a-z][a-z ]{1,30}",
    r"\bat the [a-z][a-z ]{1,30}",
    r"\bnear the [a-z][a-z ]{1,30}",
    r"\bin [A-Z][a-z]+(?: [A-Z][a-z]+){0,2}"
]

TIME_PATTERNS = [
    r"\btoday\b",
    r"\byesterday\b",
    r"\bon monday\b",
    r"\bon tuesday\b",
    r"\bon wednesday\b",
    r"\bon thursday\b",
    r"\bon friday\b",
    r"\bon saturday\b",
    r"\bon sunday\b",
    r"\b\d{1,2}(?:st|nd|rd|th)? "
    r"(?:january|february|march|april|may|june|"
    r"july|august|september|october|november|december)\b"
]


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def find_matches(text, patterns):

    matches = []

    for pattern in patterns:
        found = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for item in found:

            if isinstance(item, tuple):
                item = " ".join(item)

            if item and item not in matches:
                matches.append(item)

    return matches


def find_categories(text):

    found = []

    for category, patterns in CATEGORY_PATTERNS.items():

        if find_matches(text, patterns):
            found.append(category)

    return found


# --------------------------------------------------
# 4. ANALYZE EACH POST
# --------------------------------------------------

def analyze_post(row):

    original_text = str(row["text"])
    text = original_text.lower()

    categories = find_categories(text)

    actions = find_matches(
        text, ACTION_PATTERNS
    )

    consequences = find_matches(
        text, CONSEQUENCE_PATTERNS
    )

    incident_details = find_matches(
        text, INCIDENT_DETAIL_PATTERNS
    )

    future_evidence = find_matches(
        text, FUTURE_PATTERNS
    )

    historical_evidence = find_matches(
        text, HISTORICAL_PATTERNS
    )

    explanatory_evidence = find_matches(
        text, EXPLANATORY_PATTERNS
    )

    demand_evidence = find_matches(
        text, DEMAND_PATTERNS
    )

    location_evidence = find_matches(
        text, LOCATION_PATTERNS
    )

    time_evidence = find_matches(
        text, TIME_PATTERNS
    )

    has_category = bool(categories)
    has_action = bool(actions)
    has_consequence = bool(consequences)
    has_incident_detail = bool(incident_details)

    is_future = bool(future_evidence)
    is_historical = bool(historical_evidence)

    is_explanatory = bool(explanatory_evidence)
    is_demand = bool(demand_evidence)

    # A specific incident requires a category,
    # a concrete action, and an incident detail.
    specific_event = (
        has_category
        and has_action
        and has_incident_detail
    )

    # An incident without a detected category still
    # requires an action and a concrete incident detail.
    direct_incident = (
        has_action
        and has_incident_detail
    )

    actual_occurrence = (
        specific_event or direct_incident
    )

    # --------------------------------------------------
    # DECISION LOGIC
    # --------------------------------------------------

    if is_future and is_historical:

        decision = "needs_review"
        status = "unclear"
        reason = "Both future and historical language found"

    elif is_future and actual_occurrence:

        decision = "needs_review"
        status = "unclear"
        reason = "Future language conflicts with incident evidence"

    elif is_future:

        decision = "planned_or_future"
        status = "planned_or_future"
        reason = "Future or planned event language found"

    elif is_historical:

        decision = "historical_event"
        status = "historical"
        reason = "Historical reference found; occurrence needs verification"

    elif actual_occurrence:

        decision = "event_candidate"
        status = "reported_occurrence"
        reason = "Concrete action and incident detail found"

    elif is_explanatory or is_demand:

        decision = "needs_review"
        status = "unclear"
        reason = "Opinion, support, demand, or explanatory language found"

    elif has_category or has_action or has_consequence:

        decision = "needs_review"
        status = "unclear"
        reason = "Event-related language found without sufficient incident detail"

    else:

        decision = "non_event"
        status = "unclear"
        reason = "No event-related evidence detected"

    # Rule-based evidence score, not a probability.
    evidence_score = (
        2 * int(has_category)
        + 2 * int(has_action)
        + 2 * int(has_incident_detail)
        + int(bool(location_evidence))
        + int(bool(time_evidence))
    )

    return {
        "topic_categories": "; ".join(categories),
        "action_evidence": "; ".join(actions),
        "consequence_evidence": "; ".join(consequences),
        "incident_detail_evidence": "; ".join(incident_details),
        "location_evidence": "; ".join(location_evidence),
        "time_evidence": "; ".join(time_evidence),
        "future_evidence": "; ".join(future_evidence),
        "historical_evidence": "; ".join(historical_evidence),
        "explanatory_evidence": "; ".join(explanatory_evidence),
        "demand_evidence": "; ".join(demand_evidence),
        "actual_occurrence_evidence": actual_occurrence,
        "occurrence_status": status,
        "event_evidence_decision": decision,
        "evidence_score": evidence_score,
        "classifier_prediction": row.get(
            "predicted_label", ""
        ),
        "event_evidence_reason": reason
    }


# --------------------------------------------------
# 5. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 65)
    print("CIVIL UNREST PROJECT - STAGE 5")
    print("IMPROVED STRUCTURED EVENT EVIDENCE EXTRACTION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    if "text" not in df.columns:
        raise ValueError(
            "Input dataset has no text column."
        )

    print("\nInput records:", len(df))
    print("\nExtracting improved event evidence...")

    evidence_df = pd.DataFrame(
        [analyze_post(row) for _, row in df.iterrows()]
    )

    result = pd.concat(
        [
            df.reset_index(drop=True),
            evidence_df.reset_index(drop=True)
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
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 65)
    print("IMPROVED EVENT EVIDENCE SUMMARY")
    print("=" * 65)

    print("\nEvidence decisions:")
    print(
        result["event_evidence_decision"]
        .value_counts()
        .to_string()
    )

    print("\nOccurrence status:")
    print(
        result["occurrence_status"]
        .value_counts()
        .to_string()
    )

    print("\nDecisions by sampling group:")
    print(
        pd.crosstab(
            result["sampling_group"],
            result["event_evidence_decision"]
        ).to_string()
    )

    print("\nSaved output:")
    print(OUTPUT_FILE)

    print("\nSTAGE 5 COMPLETED")


if __name__ == "__main__":
    main()
