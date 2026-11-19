
"""
Civil Unrest Event Extraction and Prediction from Social Media Data

STAGE 1: TWEET DATA PREPROCESSING

Input:
    data/raw/tweets.csv

Output:
    data/processed/tweets_processed_rebuilt.csv
    data/processed/tweets_processed_rebuilt.parquet

The original processed files are not overwritten.
"""

from pathlib import Path
import re
import html
import hashlib

import pandas as pd
import emoji
import pyarrow as pa
import pyarrow.parquet as pq

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

DetectorFactory.seed = 42

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_DIR / "data" / "raw" / "tweets.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

CSV_OUTPUT = OUTPUT_DIR / "tweets_processed_rebuilt.csv"
PARQUET_OUTPUT = OUTPUT_DIR / "tweets_processed_rebuilt.parquet"

CHUNK_SIZE = 25000


# ---------------------------------------------------------
# 2. TEXT CLEANING FUNCTION
# ---------------------------------------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities.
    text = html.unescape(text)

    # Replace URLs with a common marker.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "[URL]",
        text
    )

    # Remove Twitter user mentions.
    text = re.sub(
        r"(?<!\w)@\w+",
        "",
        text
    )

    # Remove hashtag symbol but preserve the words.
    text = re.sub(
        r"#(\w+)",
        r" \1 ",
        text
    )

    # Convert emojis into readable text.
    text = emoji.demojize(
        text,
        delimiters=(" ", " ")
    )

    # Remove unnecessary HTML tags.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ---------------------------------------------------------
# 3. EXTRACT TWEET ID
# ---------------------------------------------------------

def extract_post_id(tweet_url):

    if pd.isna(tweet_url):
        return None

    match = re.search(
        r"/status/(\d+)",
        str(tweet_url)
    )

    if match:
        return match.group(1)

    return None


# ---------------------------------------------------------
# 4. HASH USER ID FOR PRIVACY
# ---------------------------------------------------------

def hash_user_id(user_id):

    if pd.isna(user_id):
        return None

    value = str(user_id).strip()

    if not value:
        return None

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# 5. DETECT LANGUAGE
# ---------------------------------------------------------

def detect_language(text):

    if not text or len(text.split()) < 2:
        return "unknown"

    try:
        return detect(text)

    except LangDetectException:
        return "unknown"

    except Exception:
        return "unknown"


# ---------------------------------------------------------
# 6. CHECK REQUIRED RAW COLUMNS
# ---------------------------------------------------------

def check_input_file():

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw tweet dataset not found: {RAW_FILE}"
        )

    required_columns = [
        "tweetUrl",
        "date",
        "renderedContent",
        "userId",
        "retweetedTweet"
    ]

    actual_columns = pd.read_csv(
        RAW_FILE,
        nrows=0
    ).columns.tolist()

    missing_columns = [
        column
        for column in required_columns
        if column not in actual_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in raw dataset: {missing_columns}"
        )


# ---------------------------------------------------------
# 7. MAIN PREPROCESSING FUNCTION
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("CIVIL UNREST PROJECT - STAGE 1")
    print("TWEET DATA PREPROCESSING")
    print("=" * 65)

    check_input_file()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Do not accidentally overwrite an earlier rebuilt output.
    if CSV_OUTPUT.exists() or PARQUET_OUTPUT.exists():
        raise FileExistsError(
            "Rebuilt output already exists. Rename or move the "
            "previous rebuilt files before running again."
        )

    seen_post_ids = set()

    total_rows = 0
    duplicate_rows = 0
    retweet_rows = 0
    empty_rows = 0
    non_english_rows = 0
    saved_rows = 0

    csv_header_written = False
    parquet_writer = None

    try:

        # Read raw data in manageable chunks.
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                RAW_FILE,
                chunksize=CHUNK_SIZE,
                dtype={
                    "tweetUrl": "string",
                    "userId": "string"
                },
                low_memory=False
            ),
            start=1
        ):

            print(
                f"\nProcessing chunk {chunk_number}..."
            )

            total_rows += len(chunk)

            # ---------------------------------------------
            # Extract IDs and clean text
            # ---------------------------------------------

            chunk["post_id"] = chunk["tweetUrl"].apply(
                extract_post_id
            )

            chunk["text"] = chunk["renderedContent"].apply(
                clean_text
            )

            chunk["user_hash"] = chunk["userId"].apply(
                hash_user_id
            )

            chunk["date"] = pd.to_datetime(
                chunk["date"],
                errors="coerce",
                utc=True
            )

            # ---------------------------------------------
            # Remove invalid IDs and empty text
            # ---------------------------------------------

            before = len(chunk)

            chunk = chunk[
                chunk["post_id"].notna()
            ].copy()

            empty_rows += int(
                (chunk["text"].str.strip() == "").sum()
            )

            chunk = chunk[
                chunk["text"].str.strip() != ""
            ].copy()

            # ---------------------------------------------
            # Remove retweets
            # ---------------------------------------------

            is_retweet = (
                chunk["retweetedTweet"].notna()
                | chunk["text"].str.match(
                    r"^\s*RT\s+@",
                    case=False,
                    na=False
                )
            )

            retweet_rows += int(is_retweet.sum())

            chunk = chunk[
                ~is_retweet
            ].copy()

            # ---------------------------------------------
            # Remove duplicate tweet IDs
            # ---------------------------------------------

            duplicate_mask = (
                chunk["post_id"].duplicated()
                | chunk["post_id"].isin(seen_post_ids)
            )

            duplicate_rows += int(duplicate_mask.sum())

            chunk = chunk[
                ~duplicate_mask
            ].copy()

            seen_post_ids.update(
                chunk["post_id"].tolist()
            )

            # ---------------------------------------------
            # Detect language
            # ---------------------------------------------

            chunk["language"] = chunk["text"].apply(
                detect_language
            )

            non_english_rows += int(
                (chunk["language"] != "en").sum()
            )

            chunk = chunk[
                chunk["language"] == "en"
            ].copy()

            # ---------------------------------------------
            # Location is not present in the raw dataset.
            # It will be extracted in a later stage.
            # ---------------------------------------------

            chunk["location"] = "Unknown"

            # ---------------------------------------------
            # Keep only the required output columns
            # ---------------------------------------------

            processed = chunk[
                [
                    "post_id",
                    "date",
                    "location",
                    "text",
                    "language",
                    "user_hash"
                ]
            ].copy()

            processed["date"] = processed["date"].astype(
                "string"
            )

            processed = processed.reset_index(drop=True)

            if processed.empty:
                print("No usable English posts in this chunk.")
                continue

            # ---------------------------------------------
            # Save CSV incrementally
            # ---------------------------------------------

            processed.to_csv(
                CSV_OUTPUT,
                mode="a",
                header=not csv_header_written,
                index=False,
                encoding="utf-8-sig"
            )

            csv_header_written = True

            # ---------------------------------------------
            # Save Parquet incrementally
            # ---------------------------------------------

            table = pa.Table.from_pandas(
                processed,
                preserve_index=False
            )

            if parquet_writer is None:

                parquet_writer = pq.ParquetWriter(
                    PARQUET_OUTPUT,
                    table.schema,
                    compression="snappy"
                )

            else:

                table = table.cast(
                    parquet_writer.schema
                )

            parquet_writer.write_table(table)

            saved_rows += len(processed)

            print(
                f"Rows retained from this chunk: {len(processed)}"
            )

    finally:

        if parquet_writer is not None:
            parquet_writer.close()

    # -----------------------------------------------------
    # 8. FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("PREPROCESSING SUMMARY")
    print("=" * 65)

    print("Original rows:", total_rows)
    print("Duplicate rows removed:", duplicate_rows)
    print("Retweets removed:", retweet_rows)
    print("Empty text rows:", empty_rows)
    print("Non-English rows removed:", non_english_rows)
    print("Final processed rows:", saved_rows)

    print("\nCSV saved at:")
    print(CSV_OUTPUT)

    print("\nParquet saved at:")
    print(PARQUET_OUTPUT)

    print("\nSTAGE 1 COMPLETED")


if __name__ == "__main__":
    main()
