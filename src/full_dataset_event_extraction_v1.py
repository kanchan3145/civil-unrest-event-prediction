
from pathlib import Path
import pandas as pd
import re
import json
import csv
import time


# ============================================================
# STAGE 18 — FULL DATASET EVENT CANDIDATE EXTRACTION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "raw" / "tweets.csv"

OUTPUT_DIR = ROOT / "data" / "structured"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "full_event_candidates_v1.csv"
SUMMARY_FILE = OUTPUT_DIR / "full_event_candidates_summary_v1.json"

CHUNK_SIZE = 10000


# ============================================================
# 1. EVENT PATTERNS
# ============================================================

EVENT_PATTERNS = {
    "clash": [
        r"\bclash(?:es|ed|ing)?\b",
        r"\bconfrontation\b",
        r"\bscuffle\b"
    ],
    "riot": [
        r"\briot(?:s|ing)?\b",
        r"\bviolent unrest\b"
    ],
    "strike": [
        r"\bstrike(?:s)?\b",
        r"\bbandh\b",
        r"\bwalkout\b"
    ],
    "march": [
        r"\bmarch(?:ed|ing)?\b",
        r"\brall(?:y|ies)\b",
        r"\bprocession\b",
        r"\btractor rally\b"
    ],
    "protest": [
        r"\bprotest(?:s|ed|ing|ers)?\b",
        r"\bdemonstrat(?:ion|ions|e|ed|ing|ors)\b",
        r"\bagitation\b",
        r"\bsit[\s-]?in\b",
        r"\bdharna\b"
    ]
}

ACTOR_PATTERNS = {
    "farmers": r"\bfarmers?\b",
    "students": r"\bstudents?\b",
    "workers": r"\bworkers?\b",
    "police": r"\bpolice\b",
    "trade unions": r"\btrade unions?\b",
    "protesters": r"\bprotesters?\b",
    "security forces": r"\bsecurity forces?\b",
    "activists": r"\bactivists?\b"
}

TARGET_PATTERNS = {
    "government": r"\bgovernment\b|\bgovt\b",
    "police": r"\bpolice\b",
    "farm laws": r"\bfarm laws?\b",
    "administration": r"\badministration\b",
    "political leadership": (
        r"\bprime minister\b|\bchief minister\b"
    )
}

ESCALATION_PATTERNS = {
    "violence": r"\bviolence\b|\bviolent\b",
    "clashes": r"\bclash(?:es|ed)?\b",
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
    (
        r"\b\d[\d,]*\s+"
        r"(?:people|protesters|farmers|workers)\b",
        "numeric"
    ),
    (r"\bthousands?\b", "thousands"),
    (r"\bhundreds?\b", "hundreds"),
    (
        r"\blarge crowd\b|\bmassive gathering\b",
        "large"
    )
]

FUTURE_PATTERN = (
    r"\bwill\b|\bplanned\b|\bplanning\b|"
    r"\bscheduled\b|\bexpected to\b|"
    r"\bgoing to\b|\bproposed\b"
)

NEGATION_PATTERN = (
    r"\bno protest\b|\bno strike\b|"
    r"\bno rally\b|\bno violence\b"
)

EVENT_PATTERNS = {
    key: [re.compile(p, re.IGNORECASE) for p in patterns]
    for key, patterns in EVENT_PATTERNS.items()
}

ACTOR_PATTERNS = {
    key: re.compile(pattern, re.IGNORECASE)
    for key, pattern in ACTOR_PATTERNS.items()
}

TARGET_PATTERNS = {
    key: re.compile(pattern, re.IGNORECASE)
    for key, pattern in TARGET_PATTERNS.items()
}

ESCALATION_PATTERNS = {
    key: re.compile(pattern, re.IGNORECASE)
    for key, pattern in ESCALATION_PATTERNS.items()
}

SCALE_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), label)
    for pattern, label in SCALE_PATTERNS
]

FUTURE_PATTERN = re.compile(FUTURE_PATTERN, re.IGNORECASE)
NEGATION_PATTERN = re.compile(NEGATION_PATTERN, re.IGNORECASE)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def extract_event(text):
    matches = []

    for event_type, patterns in EVENT_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)

            if match:
                matches.append(
                    (event_type, match.group(0))
                )
                break

    if not matches:
        return None, None, ""

    primary_type, primary_trigger = matches[0]

    all_types = "|".join(
        event_type for event_type, _ in matches
    )

    return primary_type, primary_trigger, all_types


def extract_labels(text, patterns):
    return "|".join(
        label
        for label, pattern in patterns.items()
        if pattern.search(text)
    )


def extract_scale(text):
    for pattern, category in SCALE_PATTERNS:
        match = pattern.search(text)

        if match:
            return category, match.group(0)

    return "", ""


def extract_record(row):
    text = row.get("renderedContent", "")

    if pd.isna(text):
        text = ""

    text = str(text).strip()

    if not text:
        return None

    event_type, trigger, all_types = extract_event(text)

    if not event_type:
        return None

    actors = extract_labels(text, ACTOR_PATTERNS)
    targets = extract_labels(text, TARGET_PATTERNS)
    escalation = extract_labels(
        text, ESCALATION_PATTERNS
    )

    scale_category, scale_evidence = extract_scale(text)

    future_signal = bool(FUTURE_PATTERN.search(text))
    negation_signal = bool(NEGATION_PATTERN.search(text))

    return {
        "tweetId": row.get("tweetId", ""),
        "tweetUrl": row.get("tweetUrl", ""),
        "post_date": row.get("date", ""),
        "text": text,
        "event_type": event_type,
        "event_trigger": trigger,
        "all_event_types": all_types,
        "actors": actors,
        "targets": targets,
        "scale_category": scale_category,
        "scale_evidence": scale_evidence,
        "escalation_indicators": escalation,
        "future_or_planned_signal": future_signal,
        "negation_signal": negation_signal,
        "event_location": "",
        "event_time": "",
        "extraction_method": "rule_based_candidate"
    }


# ============================================================
# 3. MAIN PROCESSING
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("STAGE 18 — FULL DATASET EVENT EXTRACTION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Tweet dataset not found: {INPUT_FILE}"
        )

    if OUTPUT_FILE.exists() or SUMMARY_FILE.exists():
        raise FileExistsError(
            "Stage 18 outputs already exist. "
            "Existing files will not be overwritten."
        )

    start_time = time.time()

    total_tweets = 0
    total_candidates = 0
    chunk_number = 0

    event_counts = {}

    columns = [
        "tweetId",
        "tweetUrl",
        "post_date",
        "text",
        "event_type",
        "event_trigger",
        "all_event_types",
        "actors",
        "targets",
        "scale_category",
        "scale_evidence",
        "escalation_indicators",
        "future_or_planned_signal",
        "negation_signal",
        "event_location",
        "event_time",
        "extraction_method"
    ]

    with OUTPUT_FILE.open(
        "x",
        newline="",
        encoding="utf-8"
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=columns
        )

        writer.writeheader()

        for chunk in pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            dtype=str,
            keep_default_na=False
        ):

            chunk_number += 1
            total_tweets += len(chunk)

            records = []

            for row in chunk.to_dict("records"):
                record = extract_record(row)

                if record is not None:
                    records.append(record)

                    event_type = record["event_type"]

                    event_counts[event_type] = (
                        event_counts.get(event_type, 0) + 1
                    )

            if records:
                writer.writerows(records)

            total_candidates += len(records)

            if chunk_number % 10 == 0:
                print(
                    f"Chunks: {chunk_number} | "
                    f"Tweets: {total_tweets:,} | "
                    f"Candidates: {total_candidates:,}",
                    flush=True
                )

    elapsed = round(time.time() - start_time, 2)

    summary = {
        "stage": 18,
        "total_tweets_processed": total_tweets,
        "total_event_candidates": total_candidates,
        "candidate_percentage": round(
            100 * total_candidates / max(total_tweets, 1),
            2
        ),
        "event_type_counts": event_counts,
        "processing_seconds": elapsed,
        "limitations": [
            "Keyword candidates are not verified events",
            "Event type may be ambiguous",
            "Actors and targets are lexical mentions",
            "Post date is not necessarily event date",
            "Event location and time remain unresolved",
            "No DistilBERT inference performed in this stage"
        ]
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("STAGE 18 COMPLETE")
    print("=" * 65)

    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
