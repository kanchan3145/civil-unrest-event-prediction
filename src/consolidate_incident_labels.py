
from pathlib import Path
import pandas as pd

# ============================================================
# STAGE 6.2 - FINAL INCIDENT LABEL CONSOLIDATION
# ============================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotation"

INPUT_FILE = ANNOTATION_DIR / "incident_verification_reviewed.csv"

OUTPUT_FILE = ANNOTATION_DIR / "incident_labels_consolidated.csv"
REPORT_FILE = ANNOTATION_DIR / "incident_consolidation_report.csv"

VALID_CATEGORIES = {
    "actual_incident",
    "planned_event",
    "related_action",
    "non_event",
    "uncertain",
}

BINARY_MAPPING = {
    "actual_incident": 1,
    "planned_event": 0,
    "related_action": 0,
    "non_event": 0,
    "uncertain": pd.NA,
}


def main():

    print("=" * 65)
    print("CIVIL UNREST PROJECT - STAGE 6.2")
    print("INCIDENT LABEL CONSOLIDATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # Read all columns as strings to avoid accidental
    # conversion of identifiers and categorical labels.
    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=True
    )

    print(f"\nLoaded records: {len(df)}")

    required_columns = [
        "verified_event_category",
        "verified_incident_label",
        "requires_incident_verification",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if len(df) != 100:
        print(
            f"WARNING: Expected 100 records, found {len(df)}."
        )

    # Normalize categories.
    categories = (
        df["verified_event_category"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    categories = categories.replace("", pd.NA)

    invalid = (
        categories.notna()
        & ~categories.isin(VALID_CATEGORIES)
    )

    if invalid.any():
        raise ValueError(
            "Unexpected verified categories:\n"
            + str(categories[invalid].value_counts())
        )

    # Determine which records were selected for verification.
    review_required = (
        df["requires_incident_verification"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    if review_required.isna().any():
        raise ValueError(
            "Some requires_incident_verification values "
            "are missing or invalid."
        )

    # The 32 flagged records must have a completed review.
    missing_reviews = review_required & categories.isna()

    if missing_reviews.any():
        raise ValueError(
            f"{missing_reviews.sum()} flagged posts "
            "have no verified category."
        )

    # Preserve the existing verification columns.
    # Add separate consolidated columns.
    df["final_event_category"] = categories.where(
        review_required,
        pd.NA
    )

    df["final_label_status"] = "provisional_unverified"

    df.loc[
        review_required,
        "final_label_status"
    ] = "manually_verified"

    # Only reviewed records receive final binary targets.
    df["final_incident_target"] = (
        df["final_event_category"]
        .map(BINARY_MAPPING)
        .astype("Int64")
    )

    # Uncertain reviews are completed reviews,
    # but they do not provide a binary ground-truth label.
    df.loc[
        df["final_event_category"].eq("uncertain").fillna(False),
        "final_label_status"
    ] = "verified_uncertain"

    # Validate the original binary verification column
    # without modifying it.
    original_binary = pd.to_numeric(
        df["verified_incident_label"],
        errors="coerce"
    )

    comparable = (
        review_required
        & df["final_incident_target"].notna()
        & original_binary.notna()
    )

    mismatches = (
        comparable
        & (
            df["final_incident_target"].astype("Float64")
            != original_binary.astype("Float64")
        ).fillna(False)
    )

    if mismatches.any():
        print(
            "\nWARNING: Existing verified_incident_label "
            "disagrees with the reviewed category for "
            f"{mismatches.sum()} records."
        )
        print(
            "The consolidated target follows the reviewed "
            "category. Original values are preserved."
        )

    # Count labels.
    category_counts = (
        df["final_event_category"]
        .value_counts(dropna=False)
    )

    status_counts = (
        df["final_label_status"]
        .value_counts(dropna=False)
    )

    binary_counts = (
        df["final_incident_target"]
        .value_counts(dropna=False)
    )

    print("\nFINAL CATEGORY COUNTS:")
    print(category_counts.to_string())

    print("\nLABEL STATUS COUNTS:")
    print(status_counts.to_string())

    print("\nBINARY TARGET COUNTS:")
    print(binary_counts.to_string())

    # Create summary report.
    report_rows = []

    for category, count in category_counts.items():
        name = (
            "not_assigned"
            if pd.isna(category)
            else str(category)
        )

        report_rows.append({
            "section": "event_category",
            "label": name,
            "count": int(count),
        })

    for status, count in status_counts.items():
        report_rows.append({
            "section": "label_status",
            "label": str(status),
            "count": int(count),
        })

    for target, count in binary_counts.items():
        name = (
            "missing"
            if pd.isna(target)
            else str(int(target))
        )

        report_rows.append({
            "section": "binary_target",
            "label": name,
            "count": int(count),
        })

    report = pd.DataFrame(report_rows)

    # Save outputs.
    ANNOTATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)
    report.to_csv(REPORT_FILE, index=False)

    print("\n" + "=" * 65)
    print("CONSOLIDATION SUMMARY")
    print("=" * 65)

    print(f"Total posts: {len(df)}")
    print(f"Flagged posts: {int(review_required.sum())}")
    print(
        "Completed reviews: "
        f"{int((review_required & categories.notna()).sum())}"
    )
    print(
        "Binary labels available: "
        f"{int(df['final_incident_target'].notna().sum())}"
    )
    print(
        "Binary labels missing: "
        f"{int(df['final_incident_target'].isna().sum())}"
    )

    print("\nSaved consolidated dataset:")
    print(OUTPUT_FILE)

    print("\nSaved quality-control report:")
    print(REPORT_FILE)

    print("\nSTAGE 6.2 COMPLETED")


if __name__ == "__main__":
    main()
