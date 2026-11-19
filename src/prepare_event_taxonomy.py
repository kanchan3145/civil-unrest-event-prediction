
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "annotation"
    / "structured_event_evidence_v3.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "annotation"
    / "event_taxonomy_v1.csv"
)

CATEGORY_MAPPING = {
    "event_candidate": "actual_incident",
    "planned_or_future": "planned_event",
    "related_action": "related_action",
    "non_event": "non_event",
    "needs_review": "needs_review",
}


def main():
    print("=" * 65)
    print("CIVIL UNREST PROJECT - EVENT TAXONOMY")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": str}
    )

    required_columns = {
        "post_id",
        "text",
        "event_evidence_decision",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["post_id"].isna().any():
        raise ValueError("Missing post IDs detected.")

    if df["post_id"].duplicated().any():
        raise ValueError("Duplicate post IDs detected.")

    # Convert V3 decisions into the standardized taxonomy.
    df["event_category"] = df[
        "event_evidence_decision"
    ].map(CATEGORY_MAPPING)

    if df["event_category"].isna().any():
        unknown = df.loc[
            df["event_category"].isna(),
            "event_evidence_decision"
        ].unique()

        raise ValueError(
            f"Unknown automated decisions: {unknown}"
        )

    # Binary target for actual incident detection.
    # Missing value means the post needs review.
    df["incident_prediction_target"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64"
    )

    definite = df["event_category"] != "needs_review"

    df.loc[
        definite,
        "incident_prediction_target"
    ] = (
        df.loc[
            definite,
            "event_category"
        ] == "actual_incident"
    ).astype(int)

    # Explicitly mark where these labels came from.
    df["target_source"] = "automated_rule_v3"

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTotal posts:", len(df))

    print("\nEvent categories:")
    print(
        df["event_category"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nBinary prediction targets:")
    print(
        df["incident_prediction_target"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nEVENT TAXONOMY PREPARATION COMPLETED")


if __name__ == "__main__":
    main()
