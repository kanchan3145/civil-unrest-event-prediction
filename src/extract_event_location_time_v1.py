
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import re

import pandas as pd
import spacy


# ============================================================
# STAGE 21 - EVENT LOCATION AND TIME EVIDENCE
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "structured"
    / "enriched_event_candidates_v1.csv"
)

OUTPUT_DIR = ROOT / "data" / "structured"

OUTPUT_FILE = (
    OUTPUT_DIR / "event_location_time_v1.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "event_location_time_summary_v1.json"
)

REVIEW_FILE = (
    OUTPUT_DIR / "event_location_time_review_v1.csv"
)

CHUNK_SIZE = 500
NLP_BATCH_SIZE = 32
RANDOM_SEED = 42

NEW_COLUMNS = [
    "location_mentions_ner",
    "event_location_candidate",
    "event_location_evidence",
    "event_location_resolution_status",
    "event_date_candidate",
    "event_date_expression",
    "event_date_resolution_status",
    "event_date_evidence",
    "event_datetime_precision",
    "location_time_extraction_method"
]


# ============================================================
# 1. LOCATION PATTERNS
# ============================================================

# These expressions are intentionally conservative.
# They are not a complete gazetteer of Indian locations.

LOCATION_CUES = re.compile(
    r"\b(?:in|at|near|outside|across|"
    r"through|around|from|towards|toward)\s+"
    r"(?:the\s+)?",
    re.IGNORECASE
)

EVENT_WORDS = re.compile(
    r"\b(?:protest(?:ed|ing|s|ers)?|"
    r"march(?:ed|ing|es)?|"
    r"rall(?:y|ies|ied)|"
    r"strike|strikes|"
    r"clash(?:es|ed)?|"
    r"riot(?:s|ing)?|"
    r"demonstrat(?:ion|ions|ed|ing)|"
    r"violence)\b",
    re.IGNORECASE
)


def extract_location(doc):

    mentions = []

    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC", "FAC"}:
            name = ent.text.strip()

            if name and name not in mentions:
                mentions.append(name)

    # Only one location mention is accepted as
    # an unambiguous mention-level candidate.
    # This is not proof of event occurrence there.

    if len(mentions) == 0:
        return {
            "location_mentions_ner": "",
            "event_location_candidate": "",
            "event_location_evidence": "",
            "event_location_resolution_status": "missing"
        }

    if len(mentions) > 1:
        return {
            "location_mentions_ner": "|".join(mentions),
            "event_location_candidate": "",
            "event_location_evidence": "",
            "event_location_resolution_status": "multiple_mentions_unresolved"
        }

    location = mentions[0]

    # Require an event term and a location cue
    # close to the recognized place name.

    start = doc.text.lower().find(
        location.lower()
    )

    context = doc.text[
        max(0, start - 100):
        min(len(doc.text), start + len(location) + 100)
    ]

    if (
        EVENT_WORDS.search(context)
        and LOCATION_CUES.search(context)
    ):
        return {
            "location_mentions_ner": location,
            "event_location_candidate": location,
            "event_location_evidence": context,
            "event_location_resolution_status": "contextual_candidate_unverified"
        }

    return {
        "location_mentions_ner": location,
        "event_location_candidate": "",
        "event_location_evidence": "",
        "event_location_resolution_status": "mention_only_unresolved"
    }


# ============================================================
# 2. DATE EXTRACTION
# ============================================================

MONTHS = (
    "January February March April May June July "
    "August September October November December"
)

DATE_PATTERNS = [
    re.compile(
        r"\b\d{4}-\d{2}-\d{2}\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\b\d{1,2}\s+"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
        r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
        r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{4}\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
        r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
        r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE
    )
]

RELATIVE_PATTERN = re.compile(
    r"\b(today|yesterday|tomorrow)\b",
    re.IGNORECASE
)


def parse_post_date(value):

    if not value:
        return None

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        utc=True
    )

    if pd.isna(parsed):
        return None

    return parsed.date()


def parse_absolute_date(expression):

    formats = [
        "%Y-%m-%d",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%B %d, %Y",
        "%b %d %Y",
        "%b %d, %Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(
                expression, fmt
            ).date()
        except ValueError:
            continue

    return None


def extract_event_date(text, post_date):

    matches = []

    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(
                (match.start(), match.group(0))
            )

    relative = list(
        RELATIVE_PATTERN.finditer(text)
    )

    if matches:

        unique_expressions = list(
            dict.fromkeys(
                value for _, value in sorted(matches)
            )
        )

        parsed_dates = []

        for expression in unique_expressions:
            parsed = parse_absolute_date(expression)

            if parsed is not None:
                parsed_dates.append(parsed)

        unique_dates = set(parsed_dates)

        if len(unique_dates) == 1:
            date_value = next(iter(unique_dates))

            return {
                "event_date_candidate": date_value.isoformat(),
                "event_date_expression": "|".join(
                    unique_expressions
                ),
                "event_date_resolution_status": "absolute_date_mention_unverified",
                "event_date_evidence": "|".join(
                    unique_expressions
                ),
                "event_datetime_precision": "day"
            }

        return {
            "event_date_candidate": "",
            "event_date_expression": "|".join(
                unique_expressions
            ),
            "event_date_resolution_status": "ambiguous_absolute_dates",
            "event_date_evidence": "|".join(
                unique_expressions
            ),
            "event_datetime_precision": "unknown"
        }

    if relative:

        expressions = list(
            dict.fromkeys(
                match.group(0).lower()
                for match in relative
            )
        )

        if len(expressions) != 1:
            return {
                "event_date_candidate": "",
                "event_date_expression": "|".join(
                    expressions
                ),
                "event_date_resolution_status": "multiple_relative_dates",
                "event_date_evidence": "|".join(
                    expressions
                ),
                "event_datetime_precision": "unknown"
            }

        if post_date is None:
            return {
                "event_date_candidate": "",
                "event_date_expression": expressions[0],
                "event_date_resolution_status": "relative_date_without_post_date",
                "event_date_evidence": expressions[0],
                "event_datetime_precision": "unknown"
            }

        offsets = {
            "yesterday": -1,
            "today": 0,
            "tomorrow": 1
        }

        expression = expressions[0]

        candidate = (
            post_date + timedelta(
                days=offsets[expression]
            )
        )

        return {
            "event_date_candidate": candidate.isoformat(),
            "event_date_expression": expression,
            "event_date_resolution_status": "relative_to_post_date_unverified",
            "event_date_evidence": expression,
            "event_datetime_precision": "day"
        }

    return {
        "event_date_candidate": "",
        "event_date_expression": "",
        "event_date_resolution_status": "missing",
        "event_date_evidence": "",
        "event_datetime_precision": "unknown"
    }


# ============================================================
# 3. PROCESS DATASET
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 21 - LOCATION AND EVENT-TIME EXTRACTION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    for path in [
        OUTPUT_FILE,
        SUMMARY_FILE,
        REVIEW_FILE
    ]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    print("Loading spaCy English model...", flush=True)

    nlp = spacy.load(
        "en_core_web_sm",
        disable=["parser", "lemmatizer"]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_rows = 0
    location_counts = Counter()
    date_counts = Counter()

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

        texts = (
            chunk["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        documents = nlp.pipe(
            texts,
            batch_size=NLP_BATCH_SIZE,
            n_process=1
        )

        extracted = []

        for (_, row), doc in zip(
            chunk.iterrows(),
            documents
        ):

            location = extract_location(doc)

            post_date = parse_post_date(
                row.get("post_date", "")
            )

            temporal = extract_event_date(
                str(row["text"]),
                post_date
            )

            record = {
                **location,
                **temporal,
                "location_time_extraction_method": (
                    "spacy_ner_and_date_rules_v1"
                )
            }

            extracted.append(record)

            location_counts.update([
                location[
                    "event_location_resolution_status"
                ]
            ])

            date_counts.update([
                temporal[
                    "event_date_resolution_status"
                ]
            ])

        extra_df = pd.DataFrame(
            extracted,
            index=chunk.index
        )

        result = pd.concat(
            [chunk, extra_df],
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

        total_rows += len(result)

        if chunk_number % 5 == 0:
            print(
                f"Chunks: {chunk_number} | "
                f"Processed: {total_rows:,}",
                flush=True
            )

    print("\nCreating review sample...", flush=True)

    df = pd.read_csv(
        OUTPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    review = df.sample(
        n=min(100, len(df)),
        random_state=RANDOM_SEED
    )

    review.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "stage": 21,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_processed": total_rows,
        "location_resolution_status": dict(
            location_counts
        ),
        "date_resolution_status": dict(
            date_counts
        ),
        "limitations": [
            "NER location mentions may be incorrect.",
            "Contextual candidates are not verified event locations.",
            "Date mentions may refer to unrelated events.",
            "Relative dates use the UTC posting date.",
            "Multiple date mentions remain unresolved.",
            "No geocoding or unique-event consolidation performed.",
            "No regional event counts are produced."
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
    print("STAGE 21 COMPLETE")
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
