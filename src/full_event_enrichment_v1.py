
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import re

import pandas as pd

from event_role_extraction_v2 import extract_event_roles


# ============================================================
# STAGE 20 - FULL-SCALE STRUCTURED EVENT ENRICHMENT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "structured"
    / "full_event_predictions_v1.csv"
)

OUTPUT_DIR = ROOT / "data" / "structured"

OUTPUT_FILE = (
    OUTPUT_DIR / "enriched_event_candidates_v1.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "enriched_event_summary_v1.json"
)

REVIEW_FILE = (
    OUTPUT_DIR / "enriched_event_review_sample_v1.csv"
)

CHUNK_SIZE = 2000
REVIEW_SAMPLE_SIZE = 100
RANDOM_SEED = 42

NEW_COLUMNS = [
    "role_event_actors",
    "role_event_targets",
    "role_responding_entities",
    "role_unresolved_entities",
    "role_extraction_method",
    "role_evidence_status",
    "event_type_enriched",
    "event_trigger_enriched",
    "scale_category_enriched",
    "escalation_indicators_enriched",
    "event_location_enriched",
    "event_location_status",
    "event_time_enriched",
    "event_time_status",
    "occurrence_review_status",
    "structured_record_status"
]


# ============================================================
# 1. TEMPORAL AND LOCATION EVIDENCE
# ============================================================

def has_explicit_event_time(text):
    """
    Identify limited explicit date expressions.

    This does NOT resolve relative dates or prove
    that the date refers to the unrest event.
    """

    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
        (
            r"\b(?:on\s+)?\d{1,2}\s+"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|"
            r"Sep|Oct|Nov|Dec)"
            r"[a-z]*\s+\d{4}\b"
        )
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:
            return match.group(0), "date_mention_unresolved"

    return "", "missing"


def extract_location_evidence(row):
    """
    Preserve existing location evidence, if any.

    Stage 18 did not reliably resolve locations.
    A location string is not automatically a
    confirmed event location.
    """

    value = str(
        row.get("event_location", "")
    ).strip()

    if value and value.lower() not in {
        "unknown", "nan", "none", "null"
    }:
        return value, "existing_unverified"

    return "", "missing"


# ============================================================
# 2. OCCURRENCE SIGNALS
# ============================================================

def get_occurrence_status(row):

    planned = str(
        row.get("future_or_planned_signal", "")
    ).strip().lower()

    negated = str(
        row.get("negation_signal", "")
    ).strip().lower()

    def active(value):
        return value in {
            "true", "1", "yes"
        }

    if active(negated):
        return "negation_review"

    if active(planned):
        return "planned_or_future_review"

    return "occurrence_unverified"


# ============================================================
# 3. ENRICH ONE CANDIDATE
# ============================================================

def enrich_record(row):

    text = str(
        row.get("text", "")
    )

    roles = extract_event_roles(text)

    event_time, time_status = (
        has_explicit_event_time(text)
    )

    event_location, location_status = (
        extract_location_evidence(row)
    )

    actor = roles.get(
        "event_actors", ""
    )

    target = roles.get(
        "event_targets", ""
    )

    responder = roles.get(
        "responding_entities", ""
    )

    unresolved = roles.get(
        "unresolved_entities", ""
    )

    if actor or target or responder:
        role_status = "explicit_pattern_match"
    else:
        role_status = "roles_unresolved"

    result = {
        "role_event_actors": actor,
        "role_event_targets": target,
        "role_responding_entities": responder,
        "role_unresolved_entities": unresolved,
        "role_extraction_method": "context_rules_v2",
        "role_evidence_status": role_status,

        "event_type_enriched": str(
            row.get("event_type", "")
        ),

        "event_trigger_enriched": str(
            row.get("event_trigger", "")
        ),

        "scale_category_enriched": str(
            row.get("scale_category", "")
        ),

        "escalation_indicators_enriched": str(
            row.get("escalation_indicators", "")
        ),

        "event_location_enriched": event_location,
        "event_location_status": location_status,

        "event_time_enriched": event_time,
        "event_time_status": time_status,

        "occurrence_review_status": (
            get_occurrence_status(row)
        ),

        "structured_record_status": (
            "model_positive_unverified"
        )
    }

    return result


# ============================================================
# 4. MAIN PROCESSING
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 20 - STRUCTURED EVENT ENRICHMENT")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for path in [
        OUTPUT_FILE,
        SUMMARY_FILE,
        REVIEW_FILE
    ]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    input_columns = list(
        pd.read_csv(
            INPUT_FILE,
            nrows=0
        ).columns
    )

    required = {
        "text",
        "minilm_predicted_target",
        "minilm_positive_probability"
    }

    missing = required - set(input_columns)

    if missing:
        raise ValueError(
            f"Missing input columns: {sorted(missing)}"
        )

    duplicated = (
        set(NEW_COLUMNS) & set(input_columns)
    )

    if duplicated:
        raise ValueError(
            f"Columns already exist: {sorted(duplicated)}"
        )

    total_read = 0
    total_selected = 0

    role_status_counts = Counter()
    occurrence_counts = Counter()
    location_counts = Counter()
    time_counts = Counter()
    event_type_counts = Counter()

    first_write = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=False,
            low_memory=False
        ),
        start=1
    ):

        total_read += len(chunk)

        selected = chunk.loc[
            chunk["minilm_predicted_target"] == "1"
        ].copy()

        if selected.empty:
            continue

        enriched = [
            enrich_record(row)
            for row in selected.to_dict("records")
        ]

        extra_df = pd.DataFrame(
            enriched,
            index=selected.index
        )

        result = pd.concat(
            [selected, extra_df],
            axis=1
        )

        result.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8"
        )

        first_write = False

        total_selected += len(result)

        role_status_counts.update(
            result["role_evidence_status"]
        )

        occurrence_counts.update(
            result["occurrence_review_status"]
        )

        location_counts.update(
            result["event_location_status"]
        )

        time_counts.update(
            result["event_time_status"]
        )

        event_type_counts.update(
            result["event_type_enriched"]
        )

        if chunk_number % 10 == 0:
            print(
                f"Chunks read: {chunk_number} | "
                f"Tweets scanned: {total_read:,} | "
                f"Model-positive: {total_selected:,}",
                flush=True
            )

    if total_selected == 0:
        raise ValueError(
            "No model-positive candidates found."
        )

    # ========================================================
    # 5. CREATE REVIEW SAMPLE
    # ========================================================

    print(
        "\nCreating deterministic review sample...",
        flush=True
    )

    positive_df = pd.read_csv(
        OUTPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    review_sample = positive_df.sample(
        n=min(
            REVIEW_SAMPLE_SIZE,
            len(positive_df)
        ),
        random_state=RANDOM_SEED
    )

    review_sample.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 6. SUMMARY
    # ========================================================

    summary = {
        "stage": 20,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_file": str(INPUT_FILE),
        "total_classified_candidates_read": total_read,
        "total_model_positive_candidates": total_selected,
        "role_evidence_status": dict(
            role_status_counts
        ),
        "occurrence_review_status": dict(
            occurrence_counts
        ),
        "location_status": dict(
            location_counts
        ),
        "time_status": dict(
            time_counts
        ),
        "event_type_counts": dict(
            event_type_counts
        ),
        "review_sample_size": len(review_sample),
        "important_notes": [
            "Predicted incidents are not confirmed events.",
            "Actor and target rules are conservative heuristics.",
            "A date mention is not a resolved event date.",
            "Existing location strings remain unverified.",
            "Tweet post_date is not event occurrence date.",
            "Review sample has not been human annotated.",
            "No event deduplication performed."
        ]
    }

    with SUMMARY_FILE.open(
        "x",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2
        )

    print("\n" + "=" * 65)
    print("STAGE 20 COMPLETED")
    print("=" * 65)

    print(json.dumps(
        summary,
        indent=2
    ))

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)
    print(REVIEW_FILE)


if __name__ == "__main__":
    main()
