
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json
import re

import pandas as pd


# ============================================================
# STAGE 22 - LOCATION AND TEMPORAL EVIDENCE VALIDATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "structured"

INPUT_FILE = DATA_DIR / "event_location_time_v1.csv"

OUTPUT_FILE = DATA_DIR / "validated_event_evidence_v1.csv"
SUMMARY_FILE = DATA_DIR / "validated_event_evidence_summary_v1.json"
REVIEW_FILE = DATA_DIR / "validated_event_evidence_review_v1.csv"

CHUNK_SIZE = 2000
RANDOM_SEED = 42

EVENT_PATTERN = re.compile(
    r"\b(?:protest(?:s|ed|ing|ers)?|"
    r"march(?:es|ed|ing)?|"
    r"rall(?:y|ies|ied)|"
    r"strike|strikes|striking|"
    r"clash(?:es|ed|ing)?|"
    r"riot(?:s|ing)?|"
    r"demonstrat(?:ion|ions|ed|ing|ors)|"
    r"violence|"
    r"lathi[\s-]?charg(?:e|ed|ing))\b",
    re.IGNORECASE
)

LOCATION_PREPOSITIONS = re.compile(
    r"\b(?:in|at|near|outside|across|around|"
    r"through|on|towards|toward)\s+"
    r"(?:the\s+)?$",
    re.IGNORECASE
)

ORIGIN_PREPOSITIONS = re.compile(
    r"\b(?:from|of|belonging to|based in)\s+"
    r"(?:the\s+)?$",
    re.IGNORECASE
)

DATE_RELATIONS = re.compile(
    r"\b(?:on|since|during|yesterday|today|tomorrow)\b",
    re.IGNORECASE
)

FUTURE_PATTERN = re.compile(
    r"\b(?:will|would|shall|going to|"
    r"scheduled to|plans? to|planning to|"
    r"expected to|upcoming|tomorrow|next week)\b",
    re.IGNORECASE
)

NEGATION_PATTERN = re.compile(
    r"\b(?:did not|didn't|never|"
    r"no protest|no strike|"
    r"cancelled|canceled|called off)\b",
    re.IGNORECASE
)

PAST_PATTERN = re.compile(
    r"\b(?:yesterday|last week|last month|"
    r"earlier|previously|had protested|"
    r"had marched|was held|were held)\b",
    re.IGNORECASE
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def split_locations(value):
    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


def find_mentions(text, phrase):
    pattern = re.compile(
        r"(?<!\w)" + re.escape(phrase) + r"(?!\w)",
        re.IGNORECASE
    )
    return list(pattern.finditer(text))


def find_event_positions(text):
    return [
        (match.start(), match.end())
        for match in EVENT_PATTERN.finditer(text)
    ]


def nearest_event_distance(start, end, positions):
    if not positions:
        return None

    return min(
        max(event_start - end, start - event_end, 0)
        for event_start, event_end in positions
    )


# ============================================================
# LOCATION VALIDATION
# ============================================================

def validate_location(text, locations):

    if not locations:
        return {
            "validated_event_location": "",
            "location_validation_status": "missing",
            "location_validation_evidence": ""
        }

    event_positions = find_event_positions(text)

    if not event_positions:
        return {
            "validated_event_location": "",
            "location_validation_status": "no_event_trigger",
            "location_validation_evidence": ""
        }

    candidates = []

    for location in locations:
        for mention in find_mentions(text, location):

            before = text[max(0, mention.start() - 35):mention.start()]
            after = text[mention.end():mention.end() + 35]

            # Only accept a directly preceding location relation.
            relation = LOCATION_PREPOSITIONS.search(before)
            origin = ORIGIN_PREPOSITIONS.search(before)

            if origin:
                continue

            if not relation:
                continue

            distance = nearest_event_distance(
                mention.start(),
                mention.end(),
                event_positions
            )

            # Require a nearby event expression.
            if distance is None or distance > 90:
                continue

            candidates.append({
                "location": location,
                "distance": distance,
                "evidence": (
                    before[-35:] + location + after[:35]
                ).strip()
            })

    if not candidates:
        return {
            "validated_event_location": "",
            "location_validation_status": "no_explicit_event_location_relation",
            "location_validation_evidence": ""
        }

    candidates.sort(key=lambda item: item["distance"])

    best = candidates[0]
    unique_locations = {
        item["location"].lower()
        for item in candidates
    }

    # Preserve ambiguity when different locations
    # have similarly strong event relations.
    if len(unique_locations) > 1:
        close_candidates = [
            item for item in candidates
            if item["distance"] <= best["distance"] + 15
        ]

        if len({
            item["location"].lower()
            for item in close_candidates
        }) > 1:
            return {
                "validated_event_location": "",
                "location_validation_status": "ambiguous_event_locations",
                "location_validation_evidence": " | ".join(
                    item["evidence"]
                    for item in close_candidates[:3]
                )
            }

    return {
        "validated_event_location": best["location"],
        "location_validation_status": "explicit_relation_candidate_unverified",
        "location_validation_evidence": best["evidence"]
    }


# ============================================================
# TEMPORAL VALIDATION
# ============================================================

def validate_time(row):

    text = str(row.get("text", ""))

    date_candidate = str(
        row.get("event_date_candidate", "")
    ).strip()

    date_expression = str(
        row.get("event_date_expression", "")
    ).strip()

    if not date_candidate:
        return {
            "validated_event_date": "",
            "temporal_validation_status": "missing_or_ambiguous",
            "temporal_validation_evidence": date_expression
        }

    if not date_expression:
        return {
            "validated_event_date": "",
            "temporal_validation_status": "no_date_expression",
            "temporal_validation_evidence": ""
        }

    event_positions = find_event_positions(text)

    if not event_positions:
        return {
            "validated_event_date": "",
            "temporal_validation_status": "no_event_trigger",
            "temporal_validation_evidence": date_expression
        }

    # Find the expression in the original tweet.
    # The Stage 21 expression can contain multiple
    # alternatives separated by "|".

    expressions = [
        item.strip()
        for item in date_expression.split("|")
        if item.strip()
    ]

    distances = []

    for expression in expressions:
        for match in find_mentions(text, expression):
            distance = nearest_event_distance(
                match.start(),
                match.end(),
                event_positions
            )

            if distance is not None:
                distances.append(
                    (distance, expression)
                )

    if not distances:
        return {
            "validated_event_date": "",
            "temporal_validation_status": "date_not_found_in_text",
            "temporal_validation_evidence": date_expression
        }

    distance, nearest_expression = min(distances)

    if distance > 90:
        return {
            "validated_event_date": "",
            "temporal_validation_status": "date_not_near_event",
            "temporal_validation_evidence": nearest_expression
        }

    # These checks identify review conditions.
    # They are not definitive occurrence labels.

    if NEGATION_PATTERN.search(text):
        status = "negation_context_review"

    elif FUTURE_PATTERN.search(text):
        status = "future_context_review"

    elif PAST_PATTERN.search(text):
        status = "past_context_candidate_unverified"

    else:
        status = "event_near_date_candidate_unverified"

    return {
        "validated_event_date": date_candidate,
        "temporal_validation_status": status,
        "temporal_validation_evidence": nearest_expression
    }


# ============================================================
# PROCESSING
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 22 - LOCATION AND TEMPORAL VALIDATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    for output in [
        OUTPUT_FILE,
        SUMMARY_FILE,
        REVIEW_FILE
    ]:
        if output.exists():
            raise FileExistsError(
                f"Output already exists: {output}"
            )

    location_counts = Counter()
    temporal_counts = Counter()

    total = 0
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

        records = []

        for _, row in chunk.iterrows():

            text = str(row.get("text", ""))

            locations = split_locations(
                row.get("location_mentions_ner", "")
            )

            location_result = validate_location(
                text,
                locations
            )

            time_result = validate_time(row)

            record = {
                **location_result,
                **time_result,
                "validation_method": (
                    "explicit_relation_rules_v1"
                )
            }

            records.append(record)

            location_counts.update([
                location_result[
                    "location_validation_status"
                ]
            ])

            temporal_counts.update([
                time_result[
                    "temporal_validation_status"
                ]
            ])

        extra = pd.DataFrame(
            records,
            index=chunk.index
        )

        result = pd.concat(
            [chunk, extra],
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
        total += len(result)

        print(
            f"Chunks: {chunk_number} | "
            f"Processed: {total:,}",
            flush=True
        )

    print("\nCreating review sample...", flush=True)

    df = pd.read_csv(
        OUTPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    review_columns = [
        "tweetId",
        "post_date",
        "text",
        "event_type",
        "location_mentions_ner",
        "event_location_candidate",
        "validated_event_location",
        "location_validation_status",
        "location_validation_evidence",
        "event_date_expression",
        "event_date_candidate",
        "validated_event_date",
        "temporal_validation_status",
        "temporal_validation_evidence"
    ]

    available_columns = [
        column for column in review_columns
        if column in df.columns
    ]

    # Review both positive and unresolved cases.
    groups = [
        df[
            df["location_validation_status"]
            == "explicit_relation_candidate_unverified"
        ],
        df[
            df["temporal_validation_status"]
            .isin([
                "event_near_date_candidate_unverified",
                "past_context_candidate_unverified",
                "future_context_review",
                "negation_context_review"
            ])
        ],
        df[
            df["location_validation_status"]
            == "ambiguous_event_locations"
        ],
        df[
            df["location_validation_status"]
            == "no_explicit_event_location_relation"
        ]
    ]

    samples = []

    for group in groups:
        if len(group) > 0:
            samples.append(
                group.sample(
                    n=min(25, len(group)),
                    random_state=RANDOM_SEED
                )
            )

    if samples:
        review = pd.concat(
            samples,
            ignore_index=True
        ).drop_duplicates(
            subset=["tweetId"]
        )
    else:
        review = df.sample(
            n=min(100, len(df)),
            random_state=RANDOM_SEED
        )

    review[available_columns].to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "stage": 22,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_processed": total,
        "location_validation_status": dict(
            location_counts
        ),
        "temporal_validation_status": dict(
            temporal_counts
        ),
        "review_sample_size": len(review),
        "important_notes": [
            "All extracted locations remain unverified.",
            "Temporal proximity does not prove event date.",
            "Future and negation checks are broad review signals.",
            "Multiple event locations may remain ambiguous.",
            "No incident deduplication has been performed.",
            "No regional prediction has been performed."
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
    print("STAGE 22 COMPLETE")
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
