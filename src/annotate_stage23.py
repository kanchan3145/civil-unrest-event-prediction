
from pathlib import Path
from datetime import datetime, timezone
import csv
import os

import pandas as pd


# ============================================================
# STAGE 23 - TERMINAL-BASED HUMAN ANNOTATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "annotation"
    / "stage23_human_review_v1.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "annotation"
    / "stage23_terminal_annotations_v1.csv"
)

ANNOTATION_COLUMNS = [
    "human_event_label",
    "human_occurrence_status",
    "human_location_correct",
    "human_event_location",
    "human_date_correct",
    "human_event_date",
    "human_multiple_events",
    "reviewer_notes"
]

STATUS_COLUMN = "annotation_status"
TIME_COLUMN = "annotation_updated_utc"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def display_value(row, column):
    value = str(row.get(column, "")).strip()
    return value if value else "Not available"


def ask_choice(question, options):
    """
    Display choices and return the selected value.

    q = save and quit
    s = skip current tweet
    """

    print("\n" + question)

    for key, value in options.items():
        print(f"  {key} = {value}")

    while True:
        answer = input(
            "\nYour choice (s=skip, q=quit): "
        ).strip().lower()

        if answer == "q":
            return "__QUIT__"

        if answer == "s":
            return "__SKIP__"

        if answer in options:
            return options[answer]

        print("Invalid choice. Please try again.")


def ask_text(question):
    """
    Accept free-text answers.
    Enter = unknown / not identifiable.
    q = quit
    s = skip
    """

    print("\n" + question)

    answer = input(
        "Answer (Enter=blank, s=skip, q=quit): "
    ).strip()

    if answer.lower() == "q":
        return "__QUIT__"

    if answer.lower() == "s":
        return "__SKIP__"

    return answer


def ask_date(question):
    """
    Accept an optional date in YYYY-MM-DD format.
    """

    while True:
        value = ask_text(question)

        if value in {"__QUIT__", "__SKIP__", ""}:
            return value

        try:
            datetime.strptime(value, "%Y-%m-%d")
            return value

        except ValueError:
            print(
                "Invalid date. Use YYYY-MM-DD, "
                "or press Enter if unknown."
            )


def save_annotations(df):
    """
    Save safely using a temporary file.
    Existing source file is never modified.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    df.to_csv(
        temp_file,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL
    )

    os.replace(temp_file, OUTPUT_FILE)


def get_answer(question, options=None, is_date=False):

    if options is not None:
        return ask_choice(question, options)

    if is_date:
        return ask_date(question)

    return ask_text(question)


# ============================================================
# DISPLAY CURRENT TWEET
# ============================================================

def show_tweet(row, number, total):

    print("\n" + "=" * 75)
    print(f"STAGE 23 - TWEET {number} / {total}")
    print("=" * 75)

    print("\nTWEET ID:")
    print(display_value(row, "tweetId"))

    print("\nPOST DATE:")
    print(display_value(row, "post_date"))

    print("\nTWEET TEXT:")
    print(display_value(row, "text"))

    print("\n" + "-" * 75)
    print("MODEL / EXTRACTION RESULTS")
    print("-" * 75)

    fields = [
        ("Event type", "event_type"),
        ("NER locations", "location_mentions_ner"),
        ("Stage 21 location", "event_location_candidate"),
        ("Stage 22 location", "validated_event_location"),
        ("Location status", "location_validation_status"),
        ("Location evidence", "location_validation_evidence"),
        ("Date expression", "event_date_expression"),
        ("Stage 21 date", "event_date_candidate"),
        ("Stage 22 date", "validated_event_date"),
        ("Temporal status", "temporal_validation_status"),
    ]

    for label, column in fields:
        print(
            f"{label}: {display_value(row, column)}"
        )

    print("\n" + "-" * 75)
    print("HUMAN ANNOTATION")
    print("-" * 75)


# ============================================================
# ANNOTATE ONE TWEET
# ============================================================

def annotate_tweet(row):

    answers = {}

    questions = [
        (
            "human_event_label",
            "1. Does this tweet describe an actual "
            "civil unrest event?",
            {
                "e": "event",
                "n": "not_event",
                "u": "uncertain"
            }
        ),
        (
            "human_occurrence_status",
            "2. What is the occurrence status?",
            {
                "1": "occurred",
                "2": "ongoing",
                "3": "planned",
                "4": "historical_reference",
                "5": "negated",
                "6": "uncertain"
            }
        ),
        (
            "human_location_correct",
            "3. Is the Stage 22 extracted event "
            "location correct?",
            {
                "y": "yes",
                "n": "no",
                "p": "partial",
                "a": "not_applicable",
                "u": "uncertain"
            }
        )
    ]

    for column, question, options in questions:

        answer = get_answer(
            question,
            options=options
        )

        if answer in {"__QUIT__", "__SKIP__"}:
            return answer

        answers[column] = answer

    answer = get_answer(
        "4. Enter the correct event location "
        "(leave blank if unknown):"
    )

    if answer in {"__QUIT__", "__SKIP__"}:
        return answer

    answers["human_event_location"] = answer

    answer = get_answer(
        "5. Is the Stage 22 extracted event "
        "date correct?",
        options={
            "y": "yes",
            "n": "no",
            "p": "partial",
            "a": "not_applicable",
            "u": "uncertain"
        }
    )

    if answer in {"__QUIT__", "__SKIP__"}:
        return answer

    answers["human_date_correct"] = answer

    answer = get_answer(
        "6. Enter the correct event date "
        "in YYYY-MM-DD format "
        "(leave blank if unknown):",
        is_date=True
    )

    if answer in {"__QUIT__", "__SKIP__"}:
        return answer

    answers["human_event_date"] = answer

    answer = get_answer(
        "7. Does this tweet describe multiple "
        "distinct events?",
        options={
            "y": "yes",
            "n": "no",
            "u": "uncertain"
        }
    )

    if answer in {"__QUIT__", "__SKIP__"}:
        return answer

    answers["human_multiple_events"] = answer

    answer = get_answer(
        "8. Reviewer notes (optional):"
    )

    if answer in {"__QUIT__", "__SKIP__"}:
        return answer

    answers["reviewer_notes"] = answer

    return answers


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STAGE 23 - TERMINAL HUMAN ANNOTATION")
    print("=" * 75)

    if OUTPUT_FILE.exists():

        print("\nResuming saved annotations...")

        df = pd.read_csv(
            OUTPUT_FILE,
            dtype=str,
            keep_default_na=False
        )

    else:

        if not INPUT_FILE.exists():
            raise FileNotFoundError(
                f"Input file not found: {INPUT_FILE}"
            )

        print("\nLoading Stage 23 review template...")

        df = pd.read_csv(
            INPUT_FILE,
            dtype=str,
            keep_default_na=False
        )

        for column in ANNOTATION_COLUMNS:
            if column not in df.columns:
                df[column] = ""

        df[STATUS_COLUMN] = ""
        df[TIME_COLUMN] = ""

        save_annotations(df)

    if "tweetId" not in df.columns:
        raise ValueError(
            "Required tweetId column is missing."
        )

    if df["tweetId"].duplicated().any():
        raise ValueError(
            "Duplicate tweet IDs found."
        )

    for column in ANNOTATION_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    for column in [STATUS_COLUMN, TIME_COLUMN]:
        if column not in df.columns:
            df[column] = ""

    total = len(df)

    completed = (
        df[STATUS_COLUMN] == "completed"
    ).sum()

    print(f"\nTotal tweets: {total}")
    print(f"Already completed: {completed}")
    print(f"Remaining: {total - completed}")

    print("\nControls:")
    print("  q = save and quit")
    print("  s = skip current tweet")
    print("  Enter = blank for text/date questions")

    try:

        for index in range(total):

            if (
                df.at[index, STATUS_COLUMN]
                == "completed"
            ):
                continue

            row = df.iloc[index]

            show_tweet(
                row,
                index + 1,
                total
            )

            result = annotate_tweet(row)

            if result == "__QUIT__":
                save_annotations(df)
                print("\nSaved. Exiting annotation.")
                break

            if result == "__SKIP__":
                print("\nTweet skipped for now.")
                continue

            for column, value in result.items():
                df.at[index, column] = value

            df.at[index, STATUS_COLUMN] = "completed"

            df.at[index, TIME_COLUMN] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            save_annotations(df)

            completed = (
                df[STATUS_COLUMN] == "completed"
            ).sum()

            print("\nAnnotation saved successfully.")
            print(
                f"Progress: {completed}/{total}"
            )

        else:
            print("\n" + "=" * 75)
            print("ALL TWEETS ANNOTATED")
            print("=" * 75)

    except (KeyboardInterrupt, EOFError):
        save_annotations(df)
        print("\nInterrupted. Progress saved.")

    print(f"\nOutput file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
