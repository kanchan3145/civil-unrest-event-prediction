
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data/structured/"
    "validated_event_evidence_review_v1.csv"
)

OUTPUT_FILE = (
    ROOT / "data/annotation/"
    "stage23_human_review_v1.csv"
)

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Review file already exists: {OUTPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    if df["tweetId"].duplicated().any():
        raise ValueError("Duplicate tweet IDs found.")

    annotation_columns = [
        "human_event_label",
        "human_occurrence_status",
        "human_location_correct",
        "human_event_location",
        "human_date_correct",
        "human_event_date",
        "human_multiple_events",
        "reviewer_notes"
    ]

    for column in annotation_columns:
        df[column] = ""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 60)
    print("STAGE 23 - HUMAN REVIEW TEMPLATE CREATED")
    print("=" * 60)
    print(f"Records: {len(df)}")
    print(f"Saved: {OUTPUT_FILE}")
    print("Existing model predictions remain unchanged.")


if __name__ == "__main__":
    main()
