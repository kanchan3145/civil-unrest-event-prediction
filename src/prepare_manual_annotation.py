
"""
Civil Unrest Event Extraction and Prediction from Social Media Data

STAGE 6: PREPARE MANUAL ANNOTATION DATASET
"""

from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "annotation"
    / "structured_event_evidence_v2.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "data" / "annotation"
    / "manual_annotation.csv"
)


def main():

    print("=" * 65)
    print("CIVIL UNREST PROJECT - STAGE 6")
    print("MANUAL ANNOTATION DATASET PREPARATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "post_id",
        "text",
        "sampling_group",
        "predicted_label",
        "event_evidence_decision"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    annotation_df = df[required_columns].copy()

    # Empty columns for human reviewers.
    annotation_df["manual_label"] = ""
    annotation_df["annotation_reason"] = ""

    annotation_df["manual_label"] = (
        annotation_df["manual_label"].astype("string")
    )

    annotation_df["annotation_reason"] = (
        annotation_df["annotation_reason"].astype("string")
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    annotation_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTotal posts prepared:", len(annotation_df))

    print("\nAnnotation label options:")
    print("event      - Specific civil unrest incident reported")
    print("not_event  - Opinion, general discussion, or unrelated")
    print("uncertain  - Insufficient information")

    print("\nSaved manual annotation file:")
    print(OUTPUT_FILE)

    print("\nSTAGE 6 COMPLETED")


if __name__ == "__main__":
    main()
