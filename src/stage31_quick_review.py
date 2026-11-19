
"""
Stage 31: Correct known extraction errors and review quickly.

Run:
    python src/stage31_quick_review.py

Preserves original automatic annotations.
Saves after every decision.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/stage31"
SOURCE = OUT / "auto_event_annotations.csv"
RESULT = OUT / "quick_review_annotations.csv"

# Corrections grounded in the seven source texts.
# These are text-supported annotations, not external verification.
CORRECTIONS = {
    "V5-CTX-0003": {
        "auto_occurrence_status": "reported_occurred",
        "auto_event_type": "rally",
        "auto_event_date": "",
        "auto_incident_group": "CANDIDATE-KARTARPUR-001",
        "review_reasons": "Mixed occurred rally and future plan; date unknown",
    },
    "V5-CTX-0004": {
        "auto_occurrence_status": "reported_occurred",
        "auto_event_type": "rally",
        "auto_state": "Punjab",
        "auto_event_date": "",
        "auto_incident_group": "CANDIDATE-KARTARPUR-001",
        "review_reasons": "Same source text as 0003; not a separate Delhi rally",
    },
    "V5-CTX-0016": {
        "auto_occurrence_status": "ongoing",
        "auto_event_date": "",
        "review_reasons": "Ongoing protest and planned meetings; date assumption not accepted",
    },
    "V5-CTX-0008": {
        "auto_occurrence_status": "ongoing",
        "auto_event_date": "",
        "review_reasons": "Ongoing protest; exact start date unknown",
    },
    "V5-CTX-0010": {
        "auto_event_type": "protest_action",
        "auto_occurrence_status": "reported_occurred",
        "auto_event_date": "",
        "auto_escalation": "unknown",
        "review_reasons": "Farmers on police vehicle; police force not established",
    },
    "V5-CTX-0014": {
        "auto_occurrence_status": "reported_occurred",
        "auto_event_date": "",
        "review_reasons": "Reported clash; occurrence date not explicit",
    },
    "V5-CTX-0012": {
        "auto_occurrence_status": "reported_occurred",
        "auto_event_date": "",
        "review_reasons": "Reported ITO clash; Tuesday date requires confirmation",
    },
}


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(
        SOURCE,
        dtype={"cluster_id": "string"},
        keep_default_na=False,
    )

    if df["cluster_id"].duplicated().any():
        raise ValueError("Duplicate cluster IDs")

    if RESULT.exists():
        previous = pd.read_csv(
            RESULT,
            dtype={"cluster_id": "string"},
            keep_default_na=False,
        )
        if previous["cluster_id"].duplicated().any():
            raise ValueError("Duplicate IDs in saved review")
        df = previous
    else:
        for col in [
            "quick_review_status",
            "quick_review_notes",
        ]:
            df[col] = ""

        for idx, row in df.iterrows():
            cid = str(row["cluster_id"])
            if cid in CORRECTIONS:
                for col, new_value in CORRECTIONS[cid].items():
                    df.at[idx, col] = new_value

        # Corrections are suggestions until accepted.
        df["quick_review_status"] = ""
        df["quick_review_notes"] = ""

    OUT.mkdir(parents=True, exist_ok=True)

    def save():
        temp = RESULT.with_suffix(".tmp")
        df.to_csv(temp, index=False)
        temp.replace(RESULT)

    save()

    remaining = df[
        df["quick_review_status"] == ""
    ]

    print("\nSTAGE 31 QUICK REVIEW")
    print("Total:", len(df))
    print("Remaining:", len(remaining))
    print("a = accept suggestion")
    print("u = mark uncertain")
    print("s = skip")
    print("q = save and quit")

    for idx, row in remaining.iterrows():
        print("\n" + "=" * 75)
        print("Cluster:", row["cluster_id"])
        print("Text:", row["representative_text"])
        print("Type:", row["auto_event_type"])
        print("State:", row["auto_state"])
        print("Date:", row["auto_event_date"] or "UNKNOWN")
        print("Occurrence:", row["auto_occurrence_status"])
        print("Actors:", row["auto_actor"])
        print("Target:", row["auto_target"])
        print("Scale:", row["auto_scale"])
        print("Escalation:", row["auto_escalation"])
        print("Incident group:", row["auto_incident_group"])
        print("Review notes:", row["review_reasons"])

        while True:
            choice = input(
                "\na/u/s/q: "
            ).strip().lower()

            if choice in {"a", "u", "s", "q"}:
                break

            print("Invalid choice")

        if choice == "q":
            break

        if choice == "s":
            continue

        if choice == "a":
            df.at[idx, "quick_review_status"] = (
                "human_accepted_auto_annotation"
            )
        else:
            df.at[idx, "quick_review_status"] = (
                "human_marked_uncertain"
            )

        df.at[idx, "quick_review_notes"] = input(
            "Optional note (Enter to skip): "
        ).strip()

        save()
        print("Saved.")

    save()

    print("\nFINAL SUMMARY")
    print(
        df["quick_review_status"]
        .replace("", "not_reviewed")
        .value_counts()
        .to_string()
    )
    print("\nSaved to:", RESULT)


if __name__ == "__main__":
    main()
