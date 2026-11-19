
from pathlib import Path
import pandas as pd


# ============================================================
# CIVIL UNREST PROJECT - STAGE 6.4
# PROVISIONAL INCIDENT LABEL VERIFICATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotation"

INPUT_FILE = ANNOTATION_DIR / "incident_priority_review.csv"

OUTPUT_FILE = (
    ANNOTATION_DIR / "incident_provisional_reviewed.csv"
)

VALID_CATEGORIES = {
    "a": "actual_incident",
    "p": "planned_event",
    "r": "related_action",
    "n": "non_event",
    "u": "uncertain",
}

BINARY_MAPPING = {
    "actual_incident": 1,
    "planned_event": 0,
    "related_action": 0,
    "non_event": 0,
    "uncertain": pd.NA,
}


def detect_text_column(df):

    candidates = [
        "text",
        "renderedContent",
        "rendered_content",
        "clean_text",
        "tweet_text",
        "content",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Text column not found. "
        f"Available columns: {list(df.columns)}"
    )


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n"
            "Run Stage 6.3 first."
        )

    # Preserve text and identifiers as strings.
    source_df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=True,
    )

    if "audit_record_id" not in source_df.columns:
        raise ValueError(
            "Missing audit_record_id column."
        )

    if source_df["audit_record_id"].isna().any():
        raise ValueError(
            "Some audit_record_id values are missing."
        )

    if source_df["audit_record_id"].duplicated().any():
        raise ValueError(
            "Duplicate audit_record_id values detected."
        )

    if OUTPUT_FILE.exists():

        print("\nPrevious review file found.")
        print("Resuming saved verification session.")

        saved_df = pd.read_csv(
            OUTPUT_FILE,
            dtype=str,
            keep_default_na=True,
        )

        required = [
            "audit_record_id",
            "provisional_verified_category",
            "provisional_verification_notes",
        ]

        missing = [
            col for col in required
            if col not in saved_df.columns
        ]

        if missing:
            raise ValueError(
                f"Saved review file missing columns: {missing}"
            )

        if saved_df["audit_record_id"].duplicated().any():
            raise ValueError(
                "Duplicate IDs in saved review file."
            )

        source_ids = set(source_df["audit_record_id"])
        saved_ids = set(saved_df["audit_record_id"])

        if source_ids != saved_ids:
            raise ValueError(
                "Review queue and saved review IDs differ. "
                "Do not overwrite the existing review file."
            )

        saved_labels = saved_df.set_index(
            "audit_record_id"
        )["provisional_verified_category"]

        saved_notes = saved_df.set_index(
            "audit_record_id"
        )["provisional_verification_notes"]

        source_df["provisional_verified_category"] = (
            source_df["audit_record_id"].map(saved_labels)
        )

        source_df["provisional_verification_notes"] = (
            source_df["audit_record_id"].map(saved_notes)
        )

    else:

        source_df["provisional_verified_category"] = pd.NA
        source_df["provisional_verification_notes"] = pd.NA

    return source_df


def save_progress(df):

    df = df.copy()

    df["provisional_verified_incident_target"] = (
        df["provisional_verified_category"]
        .map(BINARY_MAPPING)
        .astype("Int64")
    )

    # Write to a temporary file first.
    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    df.to_csv(temp_file, index=False)

    # Replace the output only after writing succeeds.
    temp_file.replace(OUTPUT_FILE)


def print_post(row, index, total, text_column):

    print("\n" + "=" * 70)
    print(f"POST {index} OF {total}")
    print("=" * 70)

    print("\nAUDIT ID:")
    print(row["audit_record_id"])

    print("\nAUDIT PRIORITY:")
    print(row.get("audit_priority", "Unknown"))

    print("\nTEXT:")
    print(str(row[text_column]))

    for column, heading in [
        ("manual_label", "ORIGINAL MANUAL LABEL"),
        ("event_category", "AUTOMATED EVENT CATEGORY"),
        ("audit_caution_flags", "AUDIT CAUTION FLAGS"),
        ("strong_evidence_phrases", "STRONG EVIDENCE"),
        ("medium_evidence_phrases", "MEDIUM EVIDENCE"),
    ]:

        if column in row.index:
            value = row[column]

            if pd.notna(value) and str(value).strip():
                print(f"\n{heading}:")
                print(value)

    print("\n" + "-" * 70)
    print("SELECT THE INCIDENT-SPECIFIC CATEGORY")
    print("-" * 70)

    print("a = actual_incident")
    print("p = planned_event")
    print("r = related_action")
    print("n = non_event")
    print("u = uncertain")
    print("s = skip this post")
    print("q = save and quit")


def print_summary(df):

    completed = int(
        df["provisional_verified_category"].notna().sum()
    )

    total = len(df)

    print("\n" + "=" * 65)
    print("STAGE 6.4 VERIFICATION SUMMARY")
    print("=" * 65)

    print(f"Total provisional posts: {total}")
    print(f"Completed verifications: {completed}")
    print(f"Remaining: {total - completed}")

    print("\nVerified categories:")

    counts = (
        df["provisional_verified_category"]
        .value_counts()
    )

    if counts.empty:
        print("No completed verifications yet.")
    else:
        print(counts.to_string())

    print("\nSaved file:")
    print(OUTPUT_FILE)


def main():

    print("=" * 65)
    print("CIVIL UNREST PROJECT - STAGE 6.4")
    print("PROVISIONAL INCIDENT LABEL VERIFICATION")
    print("=" * 65)

    df = load_data()

    text_column = detect_text_column(df)

    # Check that this is the provisional review queue.
    if "final_label_status" in df.columns:

        invalid = (
            df["final_label_status"]
            .astype("string")
            .str.strip()
            .ne("provisional_unverified")
            .fillna(True)
        )

        if invalid.any():
            raise ValueError(
                "Review queue contains non-provisional records."
            )

    if len(df) != 68:
        print(
            f"\nWARNING: Expected 68 posts, found {len(df)}."
        )

    # Validate existing saved labels.
    saved_categories = set(
        df["provisional_verified_category"]
        .dropna()
        .unique()
    )

    invalid_categories = (
        saved_categories - set(VALID_CATEGORIES.values())
    )

    if invalid_categories:
        raise ValueError(
            f"Invalid saved categories: {invalid_categories}"
        )

    completed = int(
        df["provisional_verified_category"].notna().sum()
    )

    print(f"\nTotal posts: {len(df)}")
    print(f"Already completed: {completed}")
    print(f"Remaining: {len(df) - completed}")

    for position in range(len(df)):

        row = df.iloc[position]

        # Do not repeat completed posts.
        if pd.notna(
            row["provisional_verified_category"]
        ):
            continue

        print_post(
            row,
            position + 1,
            len(df),
            text_column,
        )

        while True:

            choice = input(
                "\nYour choice: "
            ).strip().lower()

            if choice == "q":

                save_progress(df)
                print_summary(df)

                print("\nProgress saved. You can resume later.")
                return

            if choice == "s":

                print("Post skipped for now.")
                break

            if choice in VALID_CATEGORIES:

                category = VALID_CATEGORIES[choice]

                notes = input(
                    "Verification notes "
                    "(optional, press Enter to skip): "
                ).strip()

                df.at[
                    df.index[position],
                    "provisional_verified_category"
                ] = category

                df.at[
                    df.index[position],
                    "provisional_verification_notes"
                ] = notes

                save_progress(df)

                print(
                    f"\nSaved: {category}"
                )

                break

            print(
                "Invalid choice. "
                "Enter a, p, r, n, u, s, or q."
            )

    save_progress(df)
    print_summary(df)

    remaining = int(
        df["provisional_verified_category"].isna().sum()
    )

    if remaining == 0:
        print("\nSTAGE 6.4 VERIFICATION COMPLETED")
    else:
        print(
            "\nSome posts remain unverified. "
            "Run the script again to review them."
        )


if __name__ == "__main__":
    main()
