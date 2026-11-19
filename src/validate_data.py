
"""
Civil Unrest Project
Stage 1: Validate existing datasets

This script checks the processed tweet dataset and ACLED dataset.
It does not modify the original datasets.
"""

from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

TWEETS_FILE = PROJECT_DIR / "data" / "processed" / "tweets_processed.csv"
ACLED_FILE = PROJECT_DIR / "data" / "raw" / "acled_india.csv"

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_FILE = REPORT_DIR / "data_validation_report.txt"


# --------------------------------------------------
# 2. Required columns
# --------------------------------------------------

TWEET_COLUMNS = [
    "post_id",
    "date",
    "location",
    "text",
    "language",
    "user_hash"
]

ACLED_COLUMNS = [
    "week",
    "region",
    "country",
    "admin1",
    "event_type",
    "sub_event_type",
    "events",
    "fatalities",
    "population_exposure",
    "disorder_type",
    "centroid_latitude",
    "centroid_longitude"
]


# --------------------------------------------------
# 3. Validate file availability
# --------------------------------------------------

def check_file(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {file_path}"
        )

    print(f"Found: {file_path.name}")


# --------------------------------------------------
# 4. Validate processed tweet dataset
# --------------------------------------------------

def validate_tweets():

    print("\nVALIDATING TWEET DATASET")
    print("-" * 50)

    check_file(TWEETS_FILE)

    actual_columns = pd.read_csv(
        TWEETS_FILE,
        nrows=0
    ).columns.tolist()

    missing_columns = [
        col for col in TWEET_COLUMNS
        if col not in actual_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing tweet columns: {missing_columns}"
        )

    total_rows = 0
    missing_values = {col: 0 for col in TWEET_COLUMNS}

    earliest_date = None
    latest_date = None

    # Read in chunks to avoid loading the entire dataset at once.
    for chunk in pd.read_csv(
        TWEETS_FILE,
        usecols=TWEET_COLUMNS,
        chunksize=100000,
        dtype={"post_id": "string"}
    ):

        total_rows += len(chunk)

        for col in TWEET_COLUMNS:
            missing_values[col] += int(
                chunk[col].isna().sum()
            )

        dates = pd.to_datetime(
            chunk["date"],
            errors="coerce",
            utc=True
        ).dropna()

        if not dates.empty:

            chunk_min = dates.min()
            chunk_max = dates.max()

            if earliest_date is None or chunk_min < earliest_date:
                earliest_date = chunk_min

            if latest_date is None or chunk_max > latest_date:
                latest_date = chunk_max

    print("Total tweet records:", total_rows)
    print("Earliest tweet date:", earliest_date)
    print("Latest tweet date:", latest_date)

    print("\nMissing values:")

    for col, count in missing_values.items():
        print(f"{col}: {count}")

    return {
        "total_rows": total_rows,
        "earliest_date": str(earliest_date),
        "latest_date": str(latest_date),
        "missing_values": missing_values
    }


# --------------------------------------------------
# 5. Validate ACLED dataset
# --------------------------------------------------

def validate_acled():

    print("\nVALIDATING ACLED DATASET")
    print("-" * 50)

    check_file(ACLED_FILE)

    df = pd.read_csv(ACLED_FILE)

    missing_columns = [
        col for col in ACLED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing ACLED columns: {missing_columns}"
        )

    dates = pd.to_datetime(
        df["week"],
        errors="coerce"
    ).dropna()

    print("Total ACLED records:", len(df))

    if not dates.empty:
        print("Earliest ACLED week:", dates.min())
        print("Latest ACLED week:", dates.max())

    print("\nEvent types:")
    print(df["event_type"].value_counts().to_string())

    print("\nStates / regions:")
    print(df["admin1"].nunique())


# --------------------------------------------------
# 6. Main function
# --------------------------------------------------

def main():

    print("=" * 60)
    print("CIVIL UNREST PROJECT - DATA VALIDATION")
    print("=" * 60)

    tweet_results = validate_tweets()
    validate_acled()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as file:

        file.write("CIVIL UNREST PROJECT - DATA VALIDATION REPORT\n")
        file.write("=" * 50 + "\n\n")

        file.write(
            f"Total tweet records: {tweet_results['total_rows']}\n"
        )

        file.write(
            f"Earliest tweet date: {tweet_results['earliest_date']}\n"
        )

        file.write(
            f"Latest tweet date: {tweet_results['latest_date']}\n\n"
        )

        file.write("Missing tweet values:\n")

        for col, count in tweet_results["missing_values"].items():
            file.write(f"{col}: {count}\n")

        file.write(
            "\nNote: ACLED and tweet datasets must be checked "
            "for compatible dates before direct comparison.\n"
        )

    print("\n" + "=" * 60)
    print("DATA VALIDATION COMPLETED")
    print("Report saved at:", REPORT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
