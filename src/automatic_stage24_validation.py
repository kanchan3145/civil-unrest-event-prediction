
from pathlib import Path
import hashlib
import json
import os
import time

import pandas as pd
import torch
from transformers import pipeline


# ============================================================
# STAGE 24 - AUTOMATIC PRETRAINED MODEL VALIDATION
# CPU ONLY - NO MODEL TRAINING
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "annotation"
    / "stage23_human_review_v1.csv"
)

OUTPUT_DIR = ROOT / "data" / "structured"
OUTPUT_FILE = (
    OUTPUT_DIR / "stage24_automatic_predictions.csv"
)
SUMMARY_FILE = (
    OUTPUT_DIR / "stage24_automatic_summary.json"
)

MODEL_NAME = (
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
)

# Change to 10 for a quick trial.
# None means process all records.
MAX_RECORDS = None

# Save after every completed record.
SAVE_EVERY = 1

# The model is used only for event and occurrence labels.
# Location/date fields are reused from Stage 22.
EVENT_LABELS = [
    "a report of a civil unrest event",
    "a discussion or opinion without a specific unrest event",
    "an unclear reference to civil unrest"
]

OCCURRENCE_LABELS = [
    "a civil unrest event that has already occurred",
    "a civil unrest event currently happening",
    "a civil unrest event planned for the future",
    "a historical reference to an earlier unrest event",
    "a denial or negation of an unrest event",
    "an unclear event occurrence status"
]

EVENT_MAP = {
    EVENT_LABELS[0]: "event",
    EVENT_LABELS[1]: "not_event",
    EVENT_LABELS[2]: "uncertain"
}

OCCURRENCE_MAP = {
    OCCURRENCE_LABELS[0]: "occurred",
    OCCURRENCE_LABELS[1]: "ongoing",
    OCCURRENCE_LABELS[2]: "planned",
    OCCURRENCE_LABELS[3]: "historical_reference",
    OCCURRENCE_LABELS[4]: "negated",
    OCCURRENCE_LABELS[5]: "uncertain"
}


# ============================================================
# HELPERS
# ============================================================

def get_value(row, possible_columns):
    """Read the first nonempty available column."""

    for column in possible_columns:
        if column in row.index:
            value = str(row[column]).strip()
            if value and value.lower() not in {
                "nan", "none", "null"
            }:
                return value

    return ""


def text_hash(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def save_csv_safely(df, path):
    """Save without replacing the original input."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = path.with_suffix(".tmp")

    df.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig"
    )

    os.replace(temporary, path)


def run_zero_shot(classifier, text, labels):
    """
    Return the top prediction and its model score.
    Scores are not calibrated probabilities.
    """

    result = classifier(
        text,
        candidate_labels=labels,
        hypothesis_template="This text is about {}.",
        multi_label=False,
        truncation=True,
        max_length=256
    )

    return (
        result["labels"][0],
        float(result["scores"][0])
    )


def get_location_evidence(row):
    """Reuse provisional Stage 22 location evidence."""

    location = get_value(
        row,
        [
            "validated_event_location",
            "event_location_candidate"
        ]
    )

    status = get_value(
        row,
        ["location_validation_status"]
    )

    # No validated event location should be claimed
    # when the location is only a contextual mention.
    if status != "explicit_relation_candidate_unverified":
        return "", "unresolved"

    if not location:
        return "", "unresolved"

    return location, "stage22_candidate_unverified"


def get_date_evidence(row):
    """Reuse Stage 22 event date only when provisional."""

    date = get_value(
        row,
        [
            "validated_event_date",
            "event_date_candidate"
        ]
    )

    status = get_value(
        row,
        ["temporal_validation_status"]
    )

    allowed_statuses = {
        "event_near_date_candidate_unverified",
        "past_context_candidate_unverified"
    }

    if status not in allowed_statuses:
        return "", "unresolved"

    if not date:
        return "", "unresolved"

    return date, "stage22_candidate_unverified"


def initialize_output(source):
    """
    Create or resume predictions.

    The source row number and text hash protect
    against accidentally resuming a changed file.
    """

    source = source.copy().reset_index(drop=True)

    source["stage24_row_id"] = [
        str(i) for i in range(len(source))
    ]

    source["stage24_text_hash"] = [
        text_hash(
            get_value(
                source.iloc[i],
                ["text", "renderedContent", "tweet_text"]
            )
        )
        for i in range(len(source))
    ]

    prediction_columns = [
        "model_event_label",
        "model_event_score",
        "model_occurrence_status",
        "model_occurrence_score",
        "model_event_location",
        "model_location_status",
        "model_event_date",
        "model_date_status",
        "model_multiple_events",
        "model_review_flag",
        "model_name",
        "annotation_source",
        "stage24_status"
    ]

    if OUTPUT_FILE.exists():

        print("Resuming previous Stage 24 predictions.")

        existing = pd.read_csv(
            OUTPUT_FILE,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig"
        )

        if len(existing) != len(source):
            raise ValueError(
                "Input row count changed. "
                "Do not resume with a different template."
            )

        if (
            existing["stage24_text_hash"].tolist()
            != source["stage24_text_hash"].tolist()
        ):
            raise ValueError(
                "Input text changed since the previous run."
            )

        return existing

    for column in prediction_columns:
        if column not in source.columns:
            source[column] = ""

    save_csv_safely(source, OUTPUT_FILE)

    return source


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 24 - AUTOMATIC EVENT VALIDATION")
    print("=" * 70)

    torch.set_num_threads(2)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_FILE}"
        )

    source = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    if "text" not in source.columns and (
        "renderedContent" not in source.columns
    ):
        raise ValueError(
            "No tweet text column found."
        )

    if MAX_RECORDS is not None:
        source = source.head(MAX_RECORDS).copy()

    df = initialize_output(source)

    total = len(df)

    completed = int(
        (df["stage24_status"] == "completed").sum()
    )

    print(f"\nTotal records: {total}")
    print(f"Already processed: {completed}")
    print(f"Remaining: {total - completed}")

    if completed == total:
        print("\nAll records already processed.")
        create_summary(df)
        return

    print("\nLoading pretrained DeBERTa model...")
    print("First run may require a model download.")

    classifier = pipeline(
        task="zero-shot-classification",
        model=MODEL_NAME,
        device=-1
    )

    print("\nModel loaded successfully.")
    print("Starting automatic predictions...\n")

    started = time.time()

    try:

        for index in range(total):

            if df.at[index, "stage24_status"] == "completed":
                continue

            row = df.iloc[index]

            tweet_text = get_value(
                row,
                [
                    "text",
                    "renderedContent",
                    "tweet_text"
                ]
            )

            if not tweet_text:

                df.at[index, "stage24_status"] = (
                    "missing_text"
                )

                save_csv_safely(df, OUTPUT_FILE)
                continue

            # ----------------------------------------
            # 1. EVENT CLASSIFICATION
            # ----------------------------------------

            event_raw, event_score = run_zero_shot(
                classifier,
                tweet_text,
                EVENT_LABELS
            )

            event_label = EVENT_MAP[event_raw]

            # ----------------------------------------
            # 2. OCCURRENCE CLASSIFICATION
            # ----------------------------------------

            if event_label == "not_event":

                occurrence = "not_applicable"
                occurrence_score = ""

            else:

                occurrence_raw, occurrence_score = (
                    run_zero_shot(
                        classifier,
                        tweet_text,
                        OCCURRENCE_LABELS
                    )
                )

                occurrence = OCCURRENCE_MAP[
                    occurrence_raw
                ]

            # ----------------------------------------
            # 3. LOCATION AND DATE
            # ----------------------------------------

            location, location_status = (
                get_location_evidence(row)
            )

            event_date, date_status = (
                get_date_evidence(row)
            )

            # ----------------------------------------
            # 4. REVIEW FLAGS
            # ----------------------------------------

            flags = []

            if event_score < 0.65:
                flags.append("low_event_score")

            if occurrence == "uncertain":
                flags.append("uncertain_occurrence")

            if occurrence in {
                "planned",
                "historical_reference",
                "negated"
            }:
                flags.append("not_new_occurrence")

            if not location:
                flags.append("location_unresolved")

            if not event_date:
                flags.append("date_unresolved")

            # ----------------------------------------
            # 5. SAVE PREDICTIONS
            # ----------------------------------------

            df.at[index, "model_event_label"] = (
                event_label
            )

            df.at[index, "model_event_score"] = (
                round(event_score, 4)
            )

            df.at[
                index,
                "model_occurrence_status"
            ] = occurrence

            df.at[
                index,
                "model_occurrence_score"
            ] = (
                round(occurrence_score, 4)
                if occurrence_score != ""
                else ""
            )

            df.at[index, "model_event_location"] = (
                location
            )

            df.at[index, "model_location_status"] = (
                location_status
            )

            df.at[index, "model_event_date"] = (
                event_date
            )

            df.at[index, "model_date_status"] = (
                date_status
            )

            # Multiple-event detection is not reliably
            # implemented by this model.
            df.at[index, "model_multiple_events"] = (
                "unresolved"
            )

            df.at[index, "model_review_flag"] = (
                ";".join(flags)
            )

            df.at[index, "model_name"] = MODEL_NAME

            df.at[index, "annotation_source"] = (
                "pretrained_model_unverified"
            )

            df.at[index, "stage24_status"] = (
                "completed"
            )

            completed += 1

            if completed % SAVE_EVERY == 0:
                save_csv_safely(df, OUTPUT_FILE)

            print(
                f"[{completed}/{total}] "
                f"Event={event_label} | "
                f"Occurrence={occurrence} | "
                f"Score={event_score:.3f}",
                flush=True
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        save_csv_safely(df, OUTPUT_FILE)
        create_summary(df)

        elapsed = time.time() - started

        print("\n" + "=" * 70)
        print("STAGE 24 - PROGRESS SAVED")
        print("=" * 70)
        print(f"Elapsed seconds: {elapsed:.1f}")
        print(f"Output: {OUTPUT_FILE}")
        print(f"Summary: {SUMMARY_FILE}")


# ============================================================
# SUMMARY
# ============================================================

def create_summary(df):

    done = df[
        df["stage24_status"] == "completed"
    ].copy()

    summary = {
        "stage": 24,
        "model": MODEL_NAME,
        "total_records": int(len(df)),
        "completed_records": int(len(done)),
        "remaining_records": int(
            len(df) - len(done)
        ),
        "annotation_source": (
            "pretrained_model_unverified"
        ),
        "event_predictions": (
            done["model_event_label"]
            .value_counts()
            .to_dict()
        ),
        "occurrence_predictions": (
            done["model_occurrence_status"]
            .value_counts()
            .to_dict()
        ),
        "location_candidates": int(
            done["model_event_location"]
            .ne("")
            .sum()
        ),
        "date_candidates": int(
            done["model_event_date"]
            .ne("")
            .sum()
        ),
        "limitations": [
            "No human ground truth generated",
            "Zero-shot scores are not calibrated",
            "Locations and dates are provisional",
            "Multiple-event detection not implemented",
            "No independent accuracy estimate"
        ]
    }

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nAUTOMATIC PREDICTION SUMMARY")
    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ))


if __name__ == "__main__":
    main()
