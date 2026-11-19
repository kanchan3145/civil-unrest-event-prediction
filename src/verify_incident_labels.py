
from pathlib import Path
import pandas as pd

# ============================================================
# STAGE 6.1 - HUMAN INCIDENT VERIFICATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

ANNOTATION_DIR = PROJECT_DIR / "data" / "annotation"

INPUT_FILE = ANNOTATION_DIR / "incident_verification.csv"

OUTPUT_FILE = (
    ANNOTATION_DIR / "incident_verification_reviewed.csv"
)

VALID_CATEGORIES = {
    "a": "actual_incident",
    "p": "planned_event",
    "r": "related_action",
    "n": "non_event",
    "u": "uncertain",
}

# Binary labels are derived only from the human-selected
# incident-specific category.
BINARY_LABELS = {
    "actual_incident": 1,
    "planned_event": 0,
    "related_action": 0,
    "non_event": 0,
    "uncertain": pd.NA,
}


def load_data():
    """Load previous review progress if available."""

    if OUTPUT_FILE.exists():
        print("\nPrevious verification file found.")
        print("Resuming saved verification progress.")

        df = pd.read_csv(
            OUTPUT_FILE,
            dtype={"post_id": str}
        )
    else:
        if not INPUT_FILE.exists():
            raise FileNotFoundError(
                f"Input file not found: {INPUT_FILE}"
            )

        df = pd.read_csv(
            INPUT_FILE,
            dtype={"post_id": str}
        )

    required = {
        "post_id",
        "text",
        "manual_label",
        "event_category",
        "verification_status",
        "requires_incident_verification",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if (
        df["post_id"].isna().any()
        or df["post_id"].duplicated().any()
    ):
        raise ValueError(
            "Missing or duplicate post_id values detected."
        )

    for column in [
        "verified_incident_label",
        "verified_event_category",
        "verification_notes",
    ]:
        if column not in df.columns:
            df[column] = pd.NA

    df["verified_incident_label"] = pd.to_numeric(
        df["verified_incident_label"],
        errors="coerce"
    ).astype("Int64")

    df["verified_event_category"] = (
        df["verified_event_category"]
        .astype("string")
        .fillna("")
    )

    df["verification_notes"] = (
        df["verification_notes"]
        .astype("string")
        .fillna("")
    )

    return df


def save_progress(df):
    """Save immediately after each verification decision."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Write to a temporary file first to reduce the
    # risk of damaging the saved review on interruption.
    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    df.to_csv(
        temporary_file,
        index=False,
        encoding="utf-8-sig"
    )

    temporary_file.replace(OUTPUT_FILE)


def is_true(value):
    """Safely interpret boolean values loaded from CSV."""

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def show_post(row, current, total):
    """Display a post and its existing evidence."""

    print("\n" + "=" * 75)
    print(f"POST {current} OF {total}")
    print("=" * 75)

    print("\nPOST ID:", row["post_id"])

    print("\nTEXT:")
    print(row["text"])

    print("\nORIGINAL MANUAL LABEL:")
    print(row["manual_label"])

    print("\nAUTOMATED EVENT CATEGORY:")
    print(row["event_category"])

    print("\nVERIFICATION STATUS:")
    print(row["verification_status"])

    for column in [
        "action_evidence",
        "incident_detail_evidence",
        "future_evidence",
        "historical_evidence",
        "event_evidence_reason",
    ]:
        if column in row.index:
            value = row[column]

            if pd.notna(value) and str(value).strip():
                print(
                    f"\n{column.upper()}:"
                )
                print(value)

    print("\n" + "-" * 75)
    print("SELECT THE INCIDENT-SPECIFIC CATEGORY")
    print("-" * 75)

    print("a = actual_incident")
    print("p = planned_event")
    print("r = related_action")
    print("n = non_event")
    print("u = uncertain")
    print("s = skip this post")
    print("q = save and quit")


def print_summary(df):
    """Display progress and verified category counts."""

    flagged = df[
        df["requires_incident_verification"].map(is_true)
    ]

    completed = flagged[
        flagged["verified_event_category"]
        .astype("string")
        .fillna("")
        .str.strip()
        .ne("")
    ]

    print("\n" + "=" * 65)
    print("INCIDENT VERIFICATION SUMMARY")
    print("=" * 65)

    print("Total flagged posts:", len(flagged))
    print("Completed verifications:", len(completed))
    print("Remaining:", len(flagged) - len(completed))

    if not completed.empty:
        print("\nVerified categories:")
        print(
            completed["verified_event_category"]
            .value_counts()
            .to_string()
        )

    print("\nSaved file:")
    print(OUTPUT_FILE)


def main():

    print("=" * 65)
    print("CIVIL UNREST PROJECT - STAGE 6.1")
    print("HUMAN INCIDENT VERIFICATION")
    print("=" * 65)

    df = load_data()

    flagged_indices = [
        index
        for index, value in
        df["requires_incident_verification"].items()
        if is_true(value)
    ]

    total = len(flagged_indices)

    print("\nTotal flagged posts:", total)

    for position, index in enumerate(
        flagged_indices,
        start=1
    ):
        existing = str(
            df.at[index, "verified_event_category"]
        ).strip()

        # Previously verified posts are skipped.
        if existing and existing != "<NA>":
            continue

        row = df.loc[index]

        show_post(
            row,
            current=position,
            total=total
        )

        while True:
            choice = input(
                "\nYour choice: "
            ).strip().lower()

            if choice == "q":
                save_progress(df)
                print_summary(df)
                print("\nProgress saved. Goodbye.")
                return

            if choice == "s":
                print("Post skipped.")
                break

            if choice not in VALID_CATEGORIES:
                print(
                    "Invalid choice. "
                    "Enter a, p, r, n, u, s, or q."
                )
                continue

            category = VALID_CATEGORIES[choice]

            print("\nSelected category:", category)

            notes = input(
                "Verification notes "
                "(optional, press Enter to skip): "
            ).strip()

            df.at[
                index,
                "verified_event_category"
            ] = category

            df.at[
                index,
                "verified_incident_label"
            ] = BINARY_LABELS[category]

            df.at[
                index,
                "verification_notes"
            ] = notes

            save_progress(df)

            print("\nVerification saved successfully.")
            break

    save_progress(df)
    print_summary(df)

    print("\nSTAGE 6.1 VERIFICATION SESSION COMPLETED")


if __name__ == "__main__":
    main()
