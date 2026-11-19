
from pathlib import Path
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/raw/tweets.csv"
PROCESSED = ROOT / "data/processed/tweets_processed.parquet"
CANDIDATES = ROOT / "data/structured/full_event_candidates_v1.csv"

OUT = ROOT / "outputs/stage35"
OUT.mkdir(parents=True, exist_ok=True)

URL_PATTERN = r"/status(?:es)?/([0-9]{1,20})(?:[/?#]|$)"


def recover_ids(urls):
    return (
        urls.astype("string")
        .str.extract(
            URL_PATTERN,
            flags=re.IGNORECASE,
            expand=False,
        )
    )


def main():
    print("\n=== STAGE 35: EXACT RAW-ID RECONCILIATION ===")

    processed = pd.read_parquet(
        PROCESSED,
        columns=["post_id"],
    )

    processed_ids = set(
        processed["post_id"].astype("string").dropna()
    )

    candidates = pd.read_csv(
        CANDIDATES,
        usecols=["tweetUrl"],
        dtype="string",
    )

    candidates["exact_id"] = recover_ids(
        candidates["tweetUrl"]
    )

    if candidates["exact_id"].isna().any():
        raise ValueError("Candidate URLs contain missing IDs")

    candidate_ids = set(candidates["exact_id"])
    unmatched_ids = candidate_ids - processed_ids

    print("Processed tweets:", len(processed_ids))
    print("Unique candidate tweets:", len(candidate_ids))
    print("Candidates outside processed:", len(unmatched_ids))

    print("\nScanning raw tweet URLs in chunks...")

    found_in_raw = set()
    total_rows = 0
    invalid_raw_urls = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            RAW,
            usecols=["tweetUrl"],
            dtype="string",
            chunksize=50000,
        ),
        start=1,
    ):
        total_rows += len(chunk)

        exact_ids = recover_ids(chunk["tweetUrl"])

        invalid_raw_urls += int(exact_ids.isna().sum())

        chunk_ids = set(exact_ids.dropna())

        found_in_raw.update(
            chunk_ids & unmatched_ids
        )

        print(
            f"Chunk {chunk_number}: "
            f"rows={total_rows:,}, "
            f"unmatched found={len(found_in_raw):,}",
            flush=True,
        )

    missing_from_raw = unmatched_ids - found_in_raw

    summary = {
        "raw_rows_scanned": total_rows,
        "raw_urls_without_recoverable_ids": invalid_raw_urls,
        "processed_unique_tweets": len(processed_ids),
        "unique_candidate_tweets": len(candidate_ids),
        "candidate_ids_outside_processed": len(unmatched_ids),
        "unmatched_ids_found_in_raw": len(found_in_raw),
        "unmatched_ids_missing_from_raw": len(missing_from_raw),
    }

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(summary, indent=2))

    pd.DataFrame({
        "tweetId": sorted(unmatched_ids),
        "found_in_raw": [
            tweet_id in found_in_raw
            for tweet_id in sorted(unmatched_ids)
        ],
    }).to_csv(
        OUT / "unmatched_id_reconciliation.csv",
        index=False,
    )

    (OUT / "raw_id_reconciliation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nReconciliation completed.")


if __name__ == "__main__":
    main()
