
from pathlib import Path
import json
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# STAGE 37 — FULL-CORPUS STRUCTURED EVENT EXTRACTION
#
# Reuses existing structured extractions.
# Extracts remaining MiniLM-positive tweets using rules.
# Fixes mixed string/Boolean Parquet serialization.
#
# CPU-only. No retraining. No MiniLM inference.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = (
    ROOT / "outputs/stage36/full_corpus_predictions.parquet"
)

PROCESSED = (
    ROOT / "data/processed/tweets_processed.parquet"
)

EXISTING = (
    ROOT / "data/structured/full_event_candidates_v1.csv"
)

RAW_EXCLUDED = (
    ROOT
    / "outputs/stage35"
    / "raw_candidates_excluded_from_processed.parquet"
)

OUT = ROOT / "outputs/stage37"

DETAIL_FILE = OUT / "structured_event_evidence.parquet"
SUMMARY_FILE = OUT / "tweet_event_summary.parquet"
AUDIT_FILE = OUT / "stage37_summary.json"
REVIEW_FILE = OUT / "event_review_queue.csv"

ID = "tweetId"

URL_RE = (
    r"/status(?:es)?/([0-9]{1,20})(?:[/?#]|$)"
)

EVENT_FIELDS = [
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
    "extraction_method",
]

BOOLEAN_FIELDS = [
    "future_or_planned_signal",
    "negation_signal",
    "minilm_review_required",
    "minilm_human_verified",
    "verified_incident",
    "requires_review",
]

EVENT_PATTERNS = {
    "protest": (
        r"\bprotest(?:s|ed|ing|ers)?\b"
        r"|\bdemonstrat(?:ion|ions|e|ed|ing|ors)\b"
        r"|\bagitation\b|\bsit[- ]in\b"
    ),
    "strike": (
        r"\bstrike(?:s|rs)?\b"
        r"|\bwalkout\b|\bhartal\b|\bbandh\b"
    ),
    "march": (
        r"\bmarch(?:ed|ing|es)?\b"
        r"|\brall(?:y|ies)\b|\bprocession\b"
    ),
    "clash": (
        r"\bclash(?:es|ed)?\b"
        r"|\bviolent confrontation\b|\bscuffle\b"
    ),
    "riot": (
        r"\briot(?:s|ing|ers)?\b|\brioting\b"
    ),
}

ESCALATION_PATTERNS = {
    "police_action": (
        r"\bpolice\b.{0,100}\b"
        r"(?:arrest|detain|charge|fire|lathi|tear gas)\b"
        r"|\b(?:arrested|detained|lathi[- ]charge)\b"
    ),
    "injury": (
        r"\binjur(?:y|ies|ed)\b|\bwounded\b"
    ),
    "fatality": (
        r"\bkilled\b|\bdeaths?\b"
        r"|\bfatalit(?:y|ies)\b"
    ),
    "road_blockade": (
        r"\bblock(?:ed|ing|ade)?\b.{0,80}"
        r"\b(?:road|highway)\b"
        r"|\broad blockade\b"
    ),
    "violence": (
        r"\bviolence\b|\bviolent\b"
        r"|\battacked\b|\bset on fire\b"
    ),
}

FUTURE_RE = re.compile(
    r"\b(?:will|would|plan(?:s|ned|ning)? to|"
    r"scheduled to|tomorrow|next week|"
    r"proposed|upcoming)\b",
    re.IGNORECASE,
)

NEGATION_RE = re.compile(
    r"\b(?:no protest|no clash|no riot|"
    r"did not protest|never happened|"
    r"cancelled|canceled|called off)\b",
    re.IGNORECASE,
)

DATE_RE = re.compile(
    r"\b(?:on|since|from)\s+"
    r"(\d{1,2}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|"
    r"Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:\s+\d{4})?)\b",
    re.IGNORECASE,
)

LOCATION_RE = re.compile(
    r"\b(?:in|at|near|outside)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
)

COMPILED_EVENTS = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in EVENT_PATTERNS.items()
}

COMPILED_ESCALATION = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in ESCALATION_PATTERNS.items()
}


# ============================================================
# UTILITIES
# ============================================================

def log(message):
    print(message, flush=True)


def clean_value(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def convert_boolean(value):
    """
    Convert CSV strings, Python booleans and missing values
    into consistent nullable Boolean values.
    """

    if value is None:
        return pd.NA

    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return bool(value)

    text = str(value).strip().lower()

    if text in ("true", "1", "yes"):
        return True

    if text in ("false", "0", "no"):
        return False

    if text in ("", "none", "nan", "null", "<na>"):
        return pd.NA

    raise ValueError(
        f"Unexpected Boolean value: {value!r}"
    )


def normalize_parquet_types(df):
    """
    Normalize DataFrame columns before writing Parquet.

    Fixes:
    - Mixed string/Boolean values
    - Nullable Boolean columns
    - Mixed object/string columns
    - Inconsistent tweet ID types
    """

    result = df.copy()

    for column in BOOLEAN_FIELDS:
        if column not in result.columns:
            continue

        result[column] = (
            result[column]
            .map(convert_boolean)
            .astype("boolean")
        )

    if ID in result.columns:
        result[ID] = result[ID].astype("string")

    for column in result.columns:

        if column in BOOLEAN_FIELDS:
            continue

        if column == ID:
            continue

        if result[column].dtype != object:
            continue

        non_null = result[column].dropna()

        if non_null.empty:
            result[column] = result[column].astype("string")
            continue

        types = {
            type(value)
            for value in non_null
        }

        if len(types) > 1:
            result[column] = result[column].astype("string")

    return result


def atomic_parquet(df, destination):
    """
    Save a Parquet file safely after type normalization.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        ".tmp.parquet"
    )

    normalized = normalize_parquet_types(df)

    normalized.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
    )

    os.replace(temporary, destination)


def atomic_json(data, destination):
    temporary = destination.with_suffix(".tmp.json")

    temporary.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    os.replace(temporary, destination)


def join_distinct(values):
    result = []
    seen = set()

    for value in values:
        text = clean_value(value)

        if not text:
            continue

        if text not in seen:
            seen.add(text)
            result.append(text)

    return " | ".join(result)


# ============================================================
# STEP 1 — LOAD STAGE 36 POSITIVE PREDICTIONS
# ============================================================

def load_positives():

    log("\n[1/8] Loading Stage 36 positive predictions...")

    df = pd.read_parquet(
        PREDICTIONS,
        columns=[
            ID,
            "minilm_positive_probability",
            "minilm_predicted_target",
            "classification_source",
        ],
    )

    df[ID] = df[ID].astype("string")

    if df[ID].isna().any():
        raise ValueError(
            "Missing tweet IDs in Stage 36 predictions"
        )

    if df[ID].duplicated().any():
        raise ValueError(
            "Duplicate tweet IDs in Stage 36 predictions"
        )

    positives = df.loc[
        df["minilm_predicted_target"] == 1
    ].copy()

    if positives.empty:
        raise ValueError(
            "No positive predictions found"
        )

    log(f"Total classified tweets: {len(df):,}")
    log(
        f"Predicted-positive tweets: {len(positives):,}"
    )

    return positives


# ============================================================
# STEP 2 — LOAD EXISTING STRUCTURED EXTRACTIONS
# ============================================================

def load_existing(positive_ids):

    log("\n[2/8] Loading existing structured extractions...")

    columns = ["tweetUrl"] + EVENT_FIELDS

    existing = pd.read_csv(
        EXISTING,
        usecols=columns,
        dtype="string",
        low_memory=False,
    )

    existing[ID] = (
        existing["tweetUrl"]
        .str.extract(
            URL_RE,
            flags=re.IGNORECASE,
            expand=False,
        )
        .astype("string")
    )

    if existing[ID].isna().any():
        raise ValueError(
            "Existing extractions contain invalid tweet URLs"
        )

    matched = existing.loc[
        existing[ID].isin(positive_ids)
    ].copy()

    matched["extraction_source"] = (
        "reused_existing_structured_extraction"
    )

    matched["extraction_record_type"] = (
        "existing_candidate_record"
    )

    log(
        f"Existing extraction rows: {len(existing):,}"
    )

    log(
        f"Matched extraction rows: {len(matched):,}"
    )

    log(
        "Unique positive tweets with existing extraction: "
        f"{matched[ID].nunique():,}"
    )

    return matched


# ============================================================
# STEP 3 — LOAD POSITIVE TWEET TEXTS
# ============================================================

def load_positive_texts(positive_ids):

    log("\n[3/8] Loading positive tweet text...")

    parquet_file = pq.ParquetFile(PROCESSED)

    parts = []

    for batch in parquet_file.iter_batches(
        batch_size=4096,
        columns=["post_id", "text", "date"],
    ):

        frame = batch.to_pandas()

        frame["post_id"] = (
            frame["post_id"].astype("string")
        )

        selected = frame.loc[
            frame["post_id"].isin(positive_ids)
        ].copy()

        if not selected.empty:
            parts.append(selected)

    if not parts:
        raise ValueError(
            "No positive tweet texts found"
        )

    texts = pd.concat(
        parts,
        ignore_index=True,
    )

    texts = texts.rename(
        columns={
            "post_id": ID,
            "date": "post_date",
        }
    )

    texts[ID] = texts[ID].astype("string")

    if texts[ID].duplicated().any():
        raise ValueError(
            "Duplicate positive tweet IDs in processed data"
        )

    actual_ids = set(texts[ID])

    if actual_ids != positive_ids:
        missing = positive_ids - actual_ids

        raise ValueError(
            f"{len(missing)} positive tweet texts missing"
        )

    log(
        f"Positive tweet texts loaded: {len(texts):,}"
    )

    return texts


# ============================================================
# STEP 4 — RULE-BASED FALLBACK EXTRACTION
# ============================================================

def extract_fallback(tweet_id, text, post_date):

    text = clean_value(text)

    matched_events = []

    for name, pattern in COMPILED_EVENTS.items():
        if pattern.search(text):
            matched_events.append(name)

    escalations = []

    for name, pattern in COMPILED_ESCALATION.items():
        if pattern.search(text):
            escalations.append(name)

    future = bool(
        FUTURE_RE.search(text)
    )

    negation = bool(
        NEGATION_RE.search(text)
    )

    date_match = DATE_RE.search(text)
    location_match = LOCATION_RE.search(text)

    location_hint = (
        location_match.group(1)
        if location_match
        else ""
    )

    time_hint = (
        date_match.group(1)
        if date_match
        else ""
    )

    primary = (
        matched_events[0]
        if matched_events
        else "unknown"
    )

    trigger = ""

    if primary != "unknown":
        match = COMPILED_EVENTS[primary].search(text)

        if match:
            trigger = match.group(0)

    return {
        ID: tweet_id,
        "tweetUrl": "",
        "event_type": primary,
        "event_trigger": trigger,
        "all_event_types": "|".join(matched_events),
        "actors": "",
        "targets": "",
        "scale_category": "",
        "scale_evidence": "",
        "escalation_indicators": "|".join(escalations),
        "future_or_planned_signal": future,
        "negation_signal": negation,
        "event_location": location_hint,
        "event_time": time_hint,
        "extraction_method": "stage37_regex_fallback",
        "extraction_source": "stage37_rule_based",
        "extraction_record_type": "fallback_tweet_record",
        "post_date": post_date,
    }


def build_fallbacks(texts, existing):

    log("\n[4/8] Extracting unmatched positive tweets...")

    matched_ids = set(
        existing[ID].astype("string")
    )

    unmatched = texts.loc[
        ~texts[ID].isin(matched_ids)
    ].copy()

    records = []

    for tweet_id, text, post_date in unmatched[
        [ID, "text", "post_date"]
    ].itertuples(index=False, name=None):

        records.append(
            extract_fallback(
                tweet_id,
                text,
                post_date,
            )
        )

    fallback = pd.DataFrame(
        records,
        columns=[
            ID,
            "tweetUrl",
            *EVENT_FIELDS,
            "extraction_source",
            "extraction_record_type",
            "post_date",
        ],
    )

    log(
        "Tweets requiring new extraction: "
        f"{len(fallback):,}"
    )

    return fallback


# ============================================================
# STEP 5 — COMBINE EXISTING AND NEW EXTRACTIONS
# ============================================================

def build_detail(
    existing,
    fallback,
    texts,
    positives,
):

    log("\n[5/8] Combining extraction evidence...")

    existing = existing.copy()

    existing = existing.merge(
        texts[[ID, "post_date"]],
        on=ID,
        how="left",
        validate="many_to_one",
    )

    columns = [
        ID,
        "tweetUrl",
        *EVENT_FIELDS,
        "extraction_source",
        "extraction_record_type",
        "post_date",
    ]

    detail = pd.concat(
        [
            existing[columns],
            fallback[columns],
        ],
        ignore_index=True,
    )

    detail = detail.merge(
        positives[
            [
                ID,
                "minilm_positive_probability",
                "classification_source",
            ]
        ],
        on=ID,
        how="left",
        validate="many_to_one",
    )

    detail["event_type"] = (
        detail["event_type"]
        .fillna("unknown")
        .replace("", "unknown")
    )

    detail["record_id"] = np.arange(
        1,
        len(detail) + 1,
        dtype=np.int64,
    )

    if detail[
        "minilm_positive_probability"
    ].isna().any():
        raise ValueError(
            "Missing MiniLM probabilities"
        )

    if set(detail[ID].astype("string")) != set(
        positives[ID].astype("string")
    ):
        raise ValueError(
            "Not every positive tweet has extraction evidence"
        )

    # Critical fix: normalize mixed Boolean columns
    # before attempting Parquet serialization.
    detail = normalize_parquet_types(detail)

    log(
        f"Structured evidence rows: {len(detail):,}"
    )

    log(
        f"Unique positive tweets: {detail[ID].nunique():,}"
    )

    return detail


# ============================================================
# STEP 6 — BUILD ONE-ROW-PER-TWEET SUMMARY
# ============================================================

def build_summary(detail, texts, positives):

    log("\n[6/8] Building tweet-level event summary...")

    grouped = detail.groupby(
        ID,
        sort=False,
    )

    summary = grouped.agg(
        extraction_record_count=(
            "record_id",
            "size",
        ),
        event_type=(
            "event_type",
            join_distinct,
        ),
        all_event_types=(
            "all_event_types",
            join_distinct,
        ),
        actors=(
            "actors",
            join_distinct,
        ),
        targets=(
            "targets",
            join_distinct,
        ),
        event_location=(
            "event_location",
            join_distinct,
        ),
        event_time=(
            "event_time",
            join_distinct,
        ),
        escalation_indicators=(
            "escalation_indicators",
            join_distinct,
        ),
        extraction_source=(
            "extraction_source",
            join_distinct,
        ),
    ).reset_index()

    summary = summary.merge(
        positives,
        on=ID,
        how="left",
        validate="one_to_one",
    )

    summary = summary.merge(
        texts,
        on=ID,
        how="left",
        validate="one_to_one",
    )

    summary["event_evidence_status"] = (
        "requires_event_verification"
    )

    summary["verified_incident"] = False
    summary["requires_review"] = True

    summary = normalize_parquet_types(summary)

    if len(summary) != len(positives):
        raise ValueError(
            "Tweet summary count does not match "
            "positive prediction count"
        )

    if summary[ID].duplicated().any():
        raise ValueError(
            "Duplicate IDs in tweet summary"
        )

    log(
        f"Tweet-level summaries: {len(summary):,}"
    )

    return summary


# ============================================================
# STEP 7 — VALIDATION AND SAVING
# ============================================================

def save_results(
    detail,
    summary,
    existing,
    fallback,
    positives,
    elapsed,
):

    log("\n[7/8] Validating extraction coverage...")

    positive_ids = set(
        positives[ID].astype("string")
    )

    detail_ids = set(
        detail[ID].astype("string")
    )

    summary_ids = set(
        summary[ID].astype("string")
    )

    if not (
        positive_ids == detail_ids == summary_ids
    ):
        raise ValueError(
            "Extraction coverage validation failed"
        )

    if summary["text"].isna().any():
        raise ValueError(
            "Missing source text in tweet summary"
        )

    unknown_types = int(
        (
            summary["event_type"]
            .fillna("")
            .str.lower()
            .isin(["unknown", ""])
        ).sum()
    )

    fallback_unknown = int(
        (
            fallback["event_type"]
            .fillna("")
            .str.lower()
            == "unknown"
        ).sum()
    )

    excluded_count = None

    if RAW_EXCLUDED.exists():
        excluded_count = int(
            pq.ParquetFile(
                RAW_EXCLUDED
            ).metadata.num_rows
        )

    report = {
        "stage": 37,
        "stage36_positive_tweets": int(
            len(positives)
        ),
        "positive_tweets_with_existing_extractions": int(
            existing[ID].nunique()
        ),
        "reused_extraction_rows": int(
            len(existing)
        ),
        "positive_tweets_with_new_fallback_extractions": int(
            len(fallback)
        ),
        "fallback_unknown_event_types": (
            fallback_unknown
        ),
        "total_structured_evidence_rows": int(
            len(detail)
        ),
        "unique_tweet_event_summaries": int(
            len(summary)
        ),
        "unknown_tweet_level_event_types": (
            unknown_types
        ),
        "tweet_level_extraction_coverage_percent": (
            100.0
        ),
        "verified_incidents": 0,
        "verified_incidents_note": (
            "No independent incident verification "
            "was performed in Stage 37."
        ),
        "raw_excluded_candidate_tweets_preserved_separately": (
            excluded_count
        ),
        "processing_seconds": round(
            elapsed, 2
        ),
        "limitations": [
            "Positive tweets are model predictions, not verified incidents",
            "Existing extractions are reused without independent validation",
            "Fallback extraction is rule-based",
            "Fallback actors and targets remain blank",
            "Fallback location and time are unverified text hints",
            "Multiple tweets may describe one real-world incident",
            "Negation and future-event signals require review",
            "Excluded raw-source tweets remain separate",
        ],
    }

    log("\n[8/8] Saving Stage 37 outputs...")

    atomic_parquet(
        detail,
        DETAIL_FILE,
    )

    atomic_parquet(
        summary,
        SUMMARY_FILE,
    )

    atomic_json(
        report,
        AUDIT_FILE,
    )

    review_columns = [
        ID,
        "text",
        "event_type",
        "event_location",
        "event_time",
        "escalation_indicators",
        "extraction_source",
        "minilm_positive_probability",
    ]

    sample = summary.sample(
        n=min(200, len(summary)),
        random_state=42,
    )

    sample[review_columns].to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    log("\n" + "=" * 65)
    log("STAGE 37: FINAL SUMMARY")
    log("=" * 65)

    log(
        json.dumps(
            report,
            indent=2,
        )
    )

    log(f"\nSaved: {DETAIL_FILE}")
    log(f"Saved: {SUMMARY_FILE}")
    log(f"Saved: {AUDIT_FILE}")
    log(f"Saved: {REVIEW_FILE}")

    log(
        "\nSTAGE 37 COMPLETED SUCCESSFULLY"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("\n" + "=" * 65)
    log(
        "STAGE 37: FULL-CORPUS "
        "STRUCTURED EVENT EXTRACTION"
    )
    log("=" * 65)

    for path in [
        PREDICTIONS,
        PROCESSED,
        EXISTING,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    positives = load_positives()

    positive_ids = set(
        positives[ID].astype("string")
    )

    existing = load_existing(
        positive_ids
    )

    texts = load_positive_texts(
        positive_ids
    )

    fallback = build_fallbacks(
        texts,
        existing,
    )

    detail = build_detail(
        existing,
        fallback,
        texts,
        positives,
    )

    summary = build_summary(
        detail,
        texts,
        positives,
    )

    save_results(
        detail,
        summary,
        existing,
        fallback,
        positives,
        time.time() - start,
    )


if __name__ == "__main__":
    main()
