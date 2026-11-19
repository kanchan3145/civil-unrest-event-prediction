
from pathlib import Path
import pandas as pd

# ============================================================
# STAGE 6: PREPARE INCIDENT VERIFICATION DATASET
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

ANNOTATION_DIR = PROJECT_DIR / "data" / "annotation"

MANUAL_FILE = ANNOTATION_DIR / "manual_annotation.csv"
TAXONOMY_FILE = ANNOTATION_DIR / "event_taxonomy_v1.csv"

OUTPUT_FILE = ANNOTATION_DIR / "incident_verification.csv"


def load_data():
    """Load original manual labels and automated taxonomy."""

    if not MANUAL_FILE.exists():
        raise FileNotFoundError(
            f"Manual annotation file not found: {MANUAL_FILE}"
        )

    if not TAXONOMY_FILE.exists():
        raise FileNotFoundError(
            f"Taxonomy file not found: {TAXONOMY_FILE}"
        )

    manual = pd.read_csv(
        MANUAL_FILE,
        dtype={"post_id": str}
    )

    taxonomy = pd.read_csv(
        TAXONOMY_FILE,
        dtype={"post_id": str}
    )

    for name, df in [
        ("Manual annotation", manual),
        ("Taxonomy", taxonomy)
    ]:
        if df["post_id"].isna().any():
            raise ValueError(
                f"{name} contains missing post IDs."
            )

        if df["post_id"].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate post IDs."
            )

    required_manual = {
        "post_id",
        "manual_label"
    }

    required_taxonomy = {
        "post_id",
        "event_category",
        "incident_prediction_target"
    }

    if not required_manual.issubset(manual.columns):
        raise ValueError(
            "Required manual annotation columns are missing."
        )

    if not required_taxonomy.issubset(taxonomy.columns):
        raise ValueError(
            "Required taxonomy columns are missing."
        )

    return manual, taxonomy


def assign_verification_status(row):
    """
    Identify whether an existing manual label can be
    used directly or needs incident-specific verification.

    The V3 prediction is never treated as ground truth.
    """

    manual_label = str(row["manual_label"]).strip().lower()
    category = str(row["event_category"]).strip().lower()

    if manual_label == "uncertain":
        return "manual_uncertain"

    if manual_label == "event":
        if category == "actual_incident":
            return "incident_confirmation_needed"

        return "event_type_reconciliation_needed"

    if manual_label == "not_event":
        if category == "actual_incident":
            return "possible_false_positive"

        return "manual_not_event"

    return "invalid_manual_label"


def prepare_dataset(manual, taxonomy):

    # Keep original manual annotations unchanged.
    # Avoid duplicating text or annotation columns.
    taxonomy_columns = [
        col for col in taxonomy.columns
        if col not in manual.columns or col == "post_id"
    ]

    merged = manual.merge(
        taxonomy[taxonomy_columns],
        on="post_id",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    if (merged["_merge"] != "both").any():
        raise ValueError(
            "Some manual posts were not found in the taxonomy."
        )

    merged.drop(columns=["_merge"], inplace=True)

    # Keep original human annotation for traceability.
    merged["original_manual_label"] = merged["manual_label"]

    merged["verification_status"] = merged.apply(
        assign_verification_status,
        axis=1
    )

    # This is a NEW incident-specific human label.
    # It is deliberately left empty until verified.
    merged["verified_incident_label"] = pd.Series(
        pd.NA,
        index=merged.index,
        dtype="Int64"
    )

    merged["verified_event_category"] = ""

    merged["verification_notes"] = ""

    # Mark whether a post needs further checking.
    merged["requires_incident_verification"] = (
        merged["verification_status"] != "manual_not_event"
    )

    # A negative original label can be used as a
    # provisional negative only when V3 does not
    # contradict it. This is not final ground truth.
    merged["provisional_incident_label"] = pd.Series(
        pd.NA,
        index=merged.index,
        dtype="Int64"
    )

    safe_negative = (
        merged["verification_status"] == "manual_not_event"
    )

    merged.loc[
        safe_negative,
        "provisional_incident_label"
    ] = 0

    return merged


def main():

    print("=" * 65)
    print("CIVIL UNREST PROJECT - STAGE 6")
    print("INCIDENT VERIFICATION DATASET PREPARATION")
    print("=" * 65)

    manual, taxonomy = load_data()

    print("\nManual annotation records:", len(manual))
    print("Automated taxonomy records:", len(taxonomy))

    result = prepare_dataset(manual, taxonomy)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nVerification status:")
    print(
        result["verification_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nRequires incident verification:")
    print(
        result["requires_incident_verification"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nManual label vs automated event category:")
    print(
        pd.crosstab(
            result["original_manual_label"],
            result["event_category"]
        ).to_string()
    )

    print("\nVerified incident labels:")
    print(
        result["verified_incident_label"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nSTAGE 6 PREPARATION COMPLETED")


if __name__ == "__main__":
    main()
