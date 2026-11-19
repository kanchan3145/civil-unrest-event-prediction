
"""
Stage 31: Interactive structured event validation.

Run:
    python src/stage31_validate_events.py

Input:
    outputs/stage30_v2/temporal_event_records.csv

Outputs:
    outputs/stage31/validated_event_annotations.csv
    outputs/stage31/validation_summary.json

Features:
- Resume previously saved work
- Review original supporting text
- Correct event type, state and occurrence date
- Label occurrence status and supporting evidence
- Identify duplicate incident groups
- Record actors, targets, scale and escalation
- Save after every reviewed record
- Preserve uncertain information
"""

from pathlib import Path
from datetime import datetime
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT / "outputs/stage30_v2"
    / "temporal_event_records.csv"
)

OUT = ROOT / "outputs/stage31"

SAVE_FILE = OUT / "validated_event_annotations.csv"
SUMMARY_FILE = OUT / "validation_summary.json"

REQUIRED = [
    "cluster_id",
    "event_type",
    "location_candidate",
    "state",
    "reference_post_date",
    "resolved_event_date",
    "resolved_date_basis",
    "representative_text",
]

ANNOTATION_COLUMNS = [
    "review_status",
    "human_event_label",
    "human_event_type",
    "human_state",
    "human_event_date",
    "human_occurrence_status",
    "human_actor",
    "human_target",
    "human_scale",
    "human_escalation",
    "human_evidence_level",
    "human_incident_group",
    "human_notes",
    "reviewed_at",
]

EVENT_LABELS = {
    "e": "event",
    "n": "not_event",
    "u": "uncertain",
}

OCCURRENCE_OPTIONS = {
    "1": "reported_occurred",
    "2": "ongoing",
    "3": "planned_future",
    "4": "historical_reference",
    "5": "unclear",
}

EVIDENCE_OPTIONS = {
    "1": "direct_text_support",
    "2": "indirect_text_support",
    "3": "insufficient_text_support",
    "4": "externally_corroborated",
}

SCALE_OPTIONS = {
    "1": "unknown",
    "2": "small",
    "3": "medium",
    "4": "large",
}

ESCALATION_OPTIONS = {
    "1": "yes",
    "2": "no",
    "3": "unknown",
}


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def valid_date(value):
    if not value:
        return True

    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", value
    ):
        return False

    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def choose(prompt, options):
    while True:
        print()
        for key, value in options.items():
            print(f"  {key} = {value}")

        answer = input(
            f"{prompt}: "
        ).strip().lower()

        if answer in options:
            return options[answer]

        print("Invalid option. Try again.")


def ask_text(prompt, default=""):
    suffix = (
        f" [{default}]" if default else ""
    )

    answer = input(
        f"{prompt}{suffix}: "
    ).strip()

    if answer == "":
        return default

    if answer == "-":
        return ""

    return answer


def ask_date(prompt, default=""):
    while True:
        answer = ask_text(prompt, default)

        if valid_date(answer):
            return answer

        print(
            "Use YYYY-MM-DD, or '-' to clear."
        )


def load_data():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT}"
        )

    source = pd.read_csv(
        INPUT,
        dtype={"cluster_id": "string"},
        keep_default_na=False,
        low_memory=False,
    )

    missing = set(REQUIRED) - set(
        source.columns
    )

    if missing:
        raise ValueError(
            f"Missing source columns: {sorted(missing)}"
        )

    if source["cluster_id"].duplicated().any():
        raise ValueError(
            "Source contains duplicate cluster IDs."
        )

    for col in ANNOTATION_COLUMNS:
        source[col] = ""

    if SAVE_FILE.exists():

        previous = pd.read_csv(
            SAVE_FILE,
            dtype={"cluster_id": "string"},
            keep_default_na=False,
            low_memory=False,
        )

        if previous["cluster_id"].duplicated().any():
            raise ValueError(
                "Saved annotations contain duplicate IDs."
            )

        previous = previous.set_index(
            "cluster_id"
        )

        for idx, row in source.iterrows():

            cid = clean(row["cluster_id"])

            if cid not in previous.index:
                continue

            for col in ANNOTATION_COLUMNS:

                if col in previous.columns:
                    source.at[idx, col] = clean(
                        previous.at[cid, col]
                    )

    return source


def save_data(df):

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Atomic replacement avoids a partially
    # written CSV if the program is interrupted.

    temp = SAVE_FILE.with_suffix(
        ".tmp"
    )

    df.to_csv(
        temp,
        index=False,
    )

    temp.replace(SAVE_FILE)

    reviewed = df[
        df["review_status"] == "reviewed"
    ]

    summary = {
        "total_clusters": int(len(df)),
        "reviewed_clusters": int(len(reviewed)),
        "remaining_clusters": int(
            len(df) - len(reviewed)
        ),
        "human_event_labels": (
            reviewed["human_event_label"]
            .value_counts()
            .to_dict()
        ),
        "occurrence_statuses": (
            reviewed["human_occurrence_status"]
            .value_counts()
            .to_dict()
        ),
        "human_date_present": int(
            reviewed[
                "human_event_date"
            ].ne("").sum()
        ),
        "human_incident_groups": int(
            reviewed.loc[
                reviewed["human_incident_group"]
                .ne(""),
                "human_incident_group",
            ].nunique()
        ),
        "warning": (
            "Human annotation is not independent "
            "verification. Incident groups and "
            "dates require supporting evidence."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


def show_record(row, number, total):

    print("\n" + "=" * 85)
    print(
        f"RECORD {number}/{total} "
        f"| {row['cluster_id']}"
    )
    print("=" * 85)

    print(
        "Predicted event type:",
        row["event_type"],
    )

    print(
        "Location:",
        row["location_candidate"],
    )

    print(
        "Resolved state:",
        row["state"],
    )

    print(
        "Post date:",
        row["reference_post_date"],
    )

    print(
        "Suggested event date:",
        clean(row["resolved_event_date"])
        or "UNKNOWN",
    )

    print(
        "Date basis:",
        row["resolved_date_basis"],
    )

    print(
        "Duplicate text review:",
        row.get(
            "duplicate_text_review",
            "unknown",
        ),
    )

    print("\nSOURCE TEXT:\n")
    print(row["representative_text"])

    print("\n" + "-" * 85)


def review_record(row):

    result = {}

    print("\nDoes the text support an event?")

    result["human_event_label"] = choose(
        "Event label",
        EVENT_LABELS,
    )

    if result["human_event_label"] == "not_event":

        result["human_event_type"] = ""
        result["human_state"] = ""
        result["human_event_date"] = ""
        result["human_occurrence_status"] = "unclear"
        result["human_actor"] = ""
        result["human_target"] = ""
        result["human_scale"] = "unknown"
        result["human_escalation"] = "unknown"
        result["human_evidence_level"] = (
            "insufficient_text_support"
        )
        result["human_incident_group"] = ""

        result["human_notes"] = ask_text(
            "Reason for not_event"
        )

    else:

        result["human_event_type"] = ask_text(
            "Correct event type",
            clean(row["event_type"]),
        )

        result["human_state"] = ask_text(
            "Correct state (use '-' if unknown)",
            clean(row["state"]),
        )

        print(
            "\nEnter an event date ONLY when "
            "supported by the source text."
        )
        print(
            "Use '-' if the exact date is uncertain."
        )

        result["human_event_date"] = ask_date(
            "Event date YYYY-MM-DD",
            clean(row["resolved_event_date"]),
        )

        result["human_occurrence_status"] = choose(
            "Occurrence status",
            OCCURRENCE_OPTIONS,
        )

        result["human_actor"] = ask_text(
            "Actor(s), or '-' if unknown"
        )

        result["human_target"] = ask_text(
            "Target(s), or '-' if unknown"
        )

        result["human_scale"] = choose(
            "Event scale",
            SCALE_OPTIONS,
        )

        result["human_escalation"] = choose(
            "Escalation indicators present?",
            ESCALATION_OPTIONS,
        )

        result["human_evidence_level"] = choose(
            "Evidence level",
            EVIDENCE_OPTIONS,
        )

        print(
            "\nIncident group: use the SAME ID "
            "for reports of the SAME real-world "
            "incident."
        )
        print(
            "Example: INC-001"
        )
        print(
            "Leave blank when grouping is uncertain."
        )

        result["human_incident_group"] = ask_text(
            "Incident group"
        )

        result["human_notes"] = ask_text(
            "Review notes / evidence"
        )

    result["review_status"] = "reviewed"

    result["reviewed_at"] = (
        datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
    )

    return result


def main():

    df = load_data()

    print("\nSTAGE 31 EVENT VALIDATION")
    print("=" * 50)

    print("Total clusters:", len(df))

    reviewed = (
        df["review_status"] == "reviewed"
    ).sum()

    print("Previously reviewed:", reviewed)
    print("Remaining:", len(df) - reviewed)

    print("\nCommands:")
    print("  Enter = review next record")
    print("  s     = skip current record")
    print("  q     = save and quit")
    print("  r     = revisit previously reviewed records")

    revisit = False

    choice = input(
        "\nRevisit reviewed records? (y/N): "
    ).strip().lower()

    if choice == "y":
        revisit = True

    total = len(df)

    try:

        for idx in df.index:

            row = df.loc[idx]

            if (
                not revisit
                and row["review_status"] == "reviewed"
            ):
                continue

            show_record(
                row,
                idx + 1,
                total,
            )

            command = input(
                "\nEnter/s/q: "
            ).strip().lower()

            if command == "q":
                break

            if command == "s":
                continue

            annotations = review_record(row)

            for key, value in annotations.items():
                df.at[idx, key] = value

            save_data(df)

            print(
                "\nSaved:",
                row["cluster_id"],
            )

    except (KeyboardInterrupt, EOFError):
        print("\nReview interrupted.")

    finally:

        save_data(df)

        completed = int(
            (
                df["review_status"] == "reviewed"
            ).sum()
        )

        print("\n" + "=" * 50)
        print("STAGE 31 SUMMARY")
        print("Reviewed:", completed)
        print("Remaining:", total - completed)
        print("Saved:", SAVE_FILE)
        print("Summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
