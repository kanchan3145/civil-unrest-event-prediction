
from pathlib import Path
import json
import re

import pandas as pd


# ============================================================
# STAGE 17: STRUCTURED EVENT ARGUMENT EXTRACTION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data/consolidated"
    / "event_mentions_context_v5_v2.csv"
)

OUTPUT_DIR = ROOT / "data/structured"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR / "structured_events_v1.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "structured_events_summary_v1.json"
)


# ============================================================
# 1. EVENT VOCABULARY
# ============================================================

EVENT_TRIGGERS = {
    "protest": [
        r"\bprotest(?:s|ed|ing)?\b",
        r"\bdemonstrat(?:ion|ions|ed|ing)\b",
        r"\bagitation\b"
    ],
    "strike": [
        r"\bstrike\b",
        r"\bstrikes\b",
        r"\bshutdown\b",
        r"\bbandh\b"
    ],
    "march": [
        r"\bmarch(?:ed|ing)?\b",
        r"\brally\b",
        r"\bprocession\b"
    ],
    "clash": [
        r"\bclash(?:es|ed)?\b",
        r"\bconfrontation\b",
        r"\bscuffle\b"
    ],
    "riot": [
        r"\briot(?:s|ing)?\b",
        r"\bviolent unrest\b"
    ]
}

ACTOR_PATTERNS = {
    "farmers": r"\bfarmers?\b",
    "police": r"\bpolice\b",
    "students": r"\bstudents?\b",
    "workers": r"\bworkers?\b",
    "trade unions": r"\btrade unions?\b",
    "protesters": r"\bprotesters?\b",
    "security forces": r"\bsecurity forces?\b"
}

TARGET_PATTERNS = {
    "government": r"\bgovernment\b|\bgovt\b",
    "police": r"\bpolice\b",
    "farm laws": r"\bfarm laws?\b",
    "administration": r"\badministration\b",
    "political leadership": r"\bprime minister\b|\bchief minister\b"
}

ESCALATION_PATTERNS = {
    "violence": r"\bviolence\b|\bviolent\b",
    "clashes": r"\bclashes?\b",
    "police_action": (
        r"\blathi[\s-]?charge\b|"
        r"\btear gas\b|"
        r"\bpolice firing\b"
    ),
    "injuries": r"\binjur(?:y|ies|ed)\b",
    "fatalities": r"\bkill(?:ed|ing)?\b|\bdeaths?\b",
    "mobilization": (
        r"\bthousands?\b|"
        r"\bmassive rally\b|"
        r"\blarge crowd\b"
    )
}

SCALE_PATTERNS = [
    (r"\b\d[\d,]*\s+(?:people|protesters|farmers|workers)\b",
     "numeric"),
    (r"\bthousands?\b", "thousands"),
    (r"\bhundreds?\b", "hundreds"),
    (r"\blarge crowd\b|\bmassive gathering\b", "large")
]

FUTURE_PATTERN = (
    r"\bwill\b|\bplanned\b|\bplanning\b|"
    r"\bscheduled\b|\bexpected to\b|"
    r"\bgoing to\b|\bproposed\b"
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_matches(text, patterns):
    matches = []

    for label, pattern in patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(label)

    return matches


def extract_event_type_and_trigger(text):
    for event_type, patterns in EVENT_TRIGGERS.items():
        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if match:
                return event_type, match.group(0)

    return None, None


def extract_scale(text):
    for pattern, scale_type in SCALE_PATTERNS:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:
            return scale_type, match.group(0)

    return None, None


def clean_value(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.lower() in [
        "", "nan", "none", "unknown", "null"
    ]:
        return None

    return value


def get_existing_value(row, candidates):
    for column in candidates:
        if column in row.index:
            value = clean_value(row[column])
            if value is not None:
                return value

    return None


def get_text_column(df):
    candidates = [
        "text",
        "renderedContent",
        "tweet_text",
        "clean_text",
        "post_text"
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "No text column found. Available columns: "
        + ", ".join(df.columns)
    )


# ============================================================
# 3. LOAD EXISTING EVENT MENTIONS
# ============================================================

print("\nSTAGE 17: STRUCTURED EVENT EXTRACTION")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Stage 12 input not found: {INPUT_FILE}"
    )

if OUTPUT_FILE.exists() or SUMMARY_FILE.exists():
    raise FileExistsError(
        "Stage 17 output already exists. "
        "Existing files were not overwritten."
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)

text_column = get_text_column(df)

print("Input mentions:", len(df))
print("Text column:", text_column)


# ============================================================
# 4. EXTRACT STRUCTURED ATTRIBUTES
# ============================================================

print("\nExtracting structured event attributes...")

records = []

for index, row in df.iterrows():

    text = clean_value(row[text_column]) or ""

    event_type, trigger = (
        extract_event_type_and_trigger(text)
    )

    actors = find_matches(
        text, ACTOR_PATTERNS
    )

    targets = find_matches(
        text, TARGET_PATTERNS
    )

    escalation = find_matches(
        text, ESCALATION_PATTERNS
    )

    scale_type, scale_evidence = (
        extract_scale(text)
    )

    # Use existing extracted event location/date.
    # Do not use the author's profile location.
    location = get_existing_value(
        row,
        [
            "resolved_location",
            "event_location",
            "extracted_location",
            "location"
        ]
    )

    event_time = get_existing_value(
        row,
        [
            "resolved_event_date",
            "event_date",
            "extracted_event_date"
        ]
    )

    future_signal = bool(
        re.search(
            FUTURE_PATTERN,
            text,
            flags=re.IGNORECASE
        )
    )

    # A simple completeness score, NOT calibrated
    # probability or model confidence.
    fields_present = sum([
        bool(event_type),
        bool(actors),
        bool(location),
        bool(event_time),
        bool(targets),
        bool(scale_type),
        bool(escalation)
    ])

    completeness = round(
        fields_present / 7,
        3
    )

    records.append({
        "mention_id": index,
        "text": text,
        "event_type": event_type,
        "event_trigger": trigger,
        "actors": "|".join(actors),
        "location": location,
        "event_time": event_time,
        "targets": "|".join(targets),
        "scale_category": scale_type,
        "scale_evidence": scale_evidence,
        "escalation_indicators": "|".join(escalation),
        "future_or_planned_signal": future_signal,
        "argument_completeness": completeness,
        "extraction_method": (
            "rule_based_argument_enrichment"
        )
    })


structured = pd.DataFrame(records)


# ============================================================
# 5. SAVE RESULTS
# ============================================================

structured.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

summary = {
    "stage": 17,
    "input_mentions": len(df),
    "structured_records": len(structured),
    "event_types_detected": int(
        structured["event_type"].notna().sum()
    ),
    "locations_available": int(
        structured["location"].notna().sum()
    ),
    "event_times_available": int(
        structured["event_time"].notna().sum()
    ),
    "actors_detected": int(
        structured["actors"].fillna("").ne("").sum()
    ),
    "targets_detected": int(
        structured["targets"].fillna("").ne("").sum()
    ),
    "escalation_signals_detected": int(
        structured["escalation_indicators"]
        .fillna("").ne("").sum()
    ),
    "average_argument_completeness": round(
        float(
            structured["argument_completeness"].mean()
        ),
        3
    ),
    "limitations": [
        "Rule-based argument enrichment, not token-level NER",
        "Actors and targets are lexical mentions, not verified roles",
        "Future signals do not prove an event is planned",
        "Missing locations and dates are not invented",
        "Completeness is not model confidence",
        "Event records are candidate mentions, not verified incidents"
    ]
}

SUMMARY_FILE.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nSTAGE 17 COMPLETE")
print(json.dumps(summary, indent=2))

print("\nSaved:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)
