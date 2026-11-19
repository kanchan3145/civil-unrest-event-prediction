
from pathlib import Path
import json
import re

import pandas as pd


# ============================================================
# STAGE 35 — FINAL FULL-CORPUS COVERAGE MANIFEST
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data/processed/tweets_processed.parquet"
CANDIDATES = ROOT / "data/structured/full_event_candidates_v1.csv"
PREDICTIONS = ROOT / "data/structured/full_event_predictions_v1.csv"

RECONCILIATION = (
    ROOT / "outputs/stage35/raw_id_reconciliation_summary.json"
)

OUT = ROOT / "outputs/stage35"

URL_PATTERN = r"/status(?:es)?/([0-9]{1,20})(?:[/?#]|$)"

PROB = "minilm_positive_probability"
TARGET = "minilm_predicted_target"

PREDICTION_FIELDS = [
    "minilm_predicted_category",
    "minilm_review_required",
    "minilm_model_version",
    "minilm_human_verified",
]


def recover_ids(urls, name):
    ids = (
        urls.astype("string")
        .str.extract(
            URL_PATTERN,
            flags=re.IGNORECASE,
            expand=False,
        )
    )

    if ids.isna().any():
        raise ValueError(
            f"{name}: {int(ids.isna().sum())} invalid URLs"
        )

    return ids.astype("string")


def load_source():
    print("\n[1/8] Loading processed dataset...")

    source = pd.read_parquet(
        SOURCE,
        columns=["post_id"],
    ).rename(columns={"post_id": "tweetId"})

    source["tweetId"] = (
        source["tweetId"].astype("string").str.strip()
    )

    valid = source["tweetId"].str.fullmatch(
        r"[0-9]{1,20}"
    ).fillna(False)

    if not valid.all():
        raise ValueError("Invalid processed tweet IDs")

    if source["tweetId"].duplicated().any():
        raise ValueError("Duplicate processed tweet IDs")

    print("Processed tweets:", len(source))
    return source


def load_candidate_ids():
    print("\n[2/8] Loading candidate IDs...")

    df = pd.read_csv(
        CANDIDATES,
        usecols=["tweetUrl"],
        dtype="string",
    )

    df["tweetId"] = recover_ids(
        df["tweetUrl"], "Candidates"
    )

    print("Candidate rows:", len(df))
    print("Unique candidate tweets:", df["tweetId"].nunique())

    return df["tweetId"]


def load_predictions():
    print("\n[3/8] Loading MiniLM predictions...")

    columns = ["tweetUrl", PROB, TARGET] + PREDICTION_FIELDS

    df = pd.read_csv(
        PREDICTIONS,
        usecols=columns,
        dtype={"tweetUrl": "string"},
        low_memory=False,
    )

    df["tweetId"] = recover_ids(
        df["tweetUrl"], "Predictions"
    )

    df[PROB] = pd.to_numeric(df[PROB], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    if df[PROB].isna().any():
        raise ValueError("Invalid prediction probabilities")

    if not df[PROB].between(0, 1).all():
        raise ValueError("Probabilities outside [0, 1]")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Invalid binary predictions")

    print("Prediction rows:", len(df))
    print("Unique predicted tweets:", df["tweetId"].nunique())

    return df


def validate_alignment(candidate_ids, predictions):
    print("\n[4/8] Validating row alignment...")

    if len(candidate_ids) != len(predictions):
        raise ValueError("Candidate/prediction row counts differ")

    if not candidate_ids.reset_index(drop=True).equals(
        predictions["tweetId"].reset_index(drop=True)
    ):
        raise ValueError("Candidate/prediction order differs")

    print("Row alignment: PASS")


def validate_reconciliation(outside_count):
    print("\n[5/8] Validating raw-source reconciliation...")

    if outside_count == 0:
        return

    if not RECONCILIATION.exists():
        raise FileNotFoundError(RECONCILIATION)

    audit = json.loads(
        RECONCILIATION.read_text(encoding="utf-8")
    )

    if (
        audit.get("candidate_ids_outside_processed") != outside_count
        or audit.get("unmatched_ids_found_in_raw") != outside_count
        or audit.get("unmatched_ids_missing_from_raw") != 0
    ):
        raise ValueError(
            "Raw-source reconciliation does not match"
        )

    print("Raw-confirmed excluded candidates:", outside_count)


def aggregate_predictions(predictions):
    print("\n[6/8] Aggregating repeated tweet predictions...")

    grouped = predictions.groupby(
        "tweetId", sort=False
    )

    stats = grouped.agg(
        prediction_row_count=("tweetId", "size"),
        distinct_targets=(TARGET, "nunique"),
        probability_min=(PROB, "min"),
        probability_max=(PROB, "max"),
        probability_mean=(PROB, "mean"),
    ).reset_index()

    stats["probability_range"] = (
        stats["probability_max"] - stats["probability_min"]
    )

    label_conflicts = stats[
        stats["distinct_targets"] > 1
    ].copy()

    probability_variations = stats[
        stats["probability_range"] > 1e-6
    ].copy()

    print(
        "Repeated prediction rows:",
        len(predictions) - len(stats),
    )
    print("Binary-label conflicts:", len(label_conflicts))
    print(
        "Tweets with probability variation:",
        len(probability_variations),
    )

    if len(label_conflicts):
        label_conflicts.to_csv(
            OUT / "binary_label_conflicts.csv",
            index=False,
        )
        raise ValueError(
            "Binary-label conflicts detected. "
            "Manual investigation required."
        )

    # A tweet may have multiple extraction rows.
    # Preserve the unanimous original label.
    first = predictions.drop_duplicates(
        "tweetId", keep="first"
    ).copy()

    # Ensure categorical metadata is consistent.
    for field in PREDICTION_FIELDS:
        counts = grouped[field].nunique(dropna=False)

        if (counts > 1).any():
            raise ValueError(
                f"Conflicting duplicate metadata: {field}"
            )

    first = first[
        ["tweetId", TARGET] + PREDICTION_FIELDS
    ]

    unique = first.merge(
        stats[
            [
                "tweetId",
                "prediction_row_count",
                "probability_min",
                "probability_max",
                "probability_mean",
                "probability_range",
            ]
        ],
        on="tweetId",
        validate="one_to_one",
    )

    unique[PROB] = unique["probability_mean"]

    unique["probability_aggregation_method"] = "mean"

    unique["probability_variation_flag"] = (
        unique["probability_range"] > 1e-6
    )

    # Existing review flags are retained. Additionally,
    # flag materially varying duplicate predictions.
    unique["duplicate_probability_review"] = (
        unique["probability_variation_flag"]
    )

    return unique, probability_variations


def build_manifest(source, unique):
    print("\n[7/8] Building complete corpus manifest...")

    source_ids = set(source["tweetId"])

    included = unique[
        unique["tweetId"].isin(source_ids)
    ].copy()

    excluded = unique[
        ~unique["tweetId"].isin(source_ids)
    ].copy()

    manifest = source.merge(
        included,
        on="tweetId",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    manifest["candidate_selected"] = (
        manifest["_merge"] == "both"
    )

    manifest["classification_status"] = (
        "not_classified_prefilter_excluded"
    )

    manifest.loc[
        manifest["candidate_selected"],
        "classification_status",
    ] = "classified_minilm"

    manifest.drop(columns=["_merge"], inplace=True)

    classified = int(manifest["candidate_selected"].sum())
    positive = int((manifest[TARGET] == 1).sum())
    negative = int((manifest[TARGET] == 0).sum())
    unclassified = len(manifest) - classified

    if len(manifest) != len(source):
        raise ValueError("Manifest row count mismatch")

    if positive + negative != classified:
        raise ValueError("Classification accounting mismatch")

    return (
        manifest,
        included,
        excluded,
        classified,
        positive,
        negative,
        unclassified,
    )


def save_outputs(
    manifest,
    included,
    excluded,
    variations,
    candidate_ids,
    predictions,
    classified,
    positive,
    negative,
    unclassified,
):
    print("\n[8/8] Saving validated Stage 35 outputs...")

    summary = {
        "stage": 35,
        "processed_tweets": int(len(manifest)),
        "candidate_extraction_rows": int(len(candidate_ids)),
        "unique_candidate_tweets": int(candidate_ids.nunique()),
        "prediction_rows": int(len(predictions)),
        "unique_prediction_tweets": int(
            predictions["tweetId"].nunique()
        ),
        "repeated_prediction_rows": int(
            predictions["tweetId"].duplicated().sum()
        ),
        "unique_candidates_in_processed": int(len(included)),
        "unique_candidates_excluded_from_processed": int(
            len(excluded)
        ),
        "unique_classified_tweets": int(classified),
        "predicted_positive_tweets": int(positive),
        "predicted_negative_tweets": int(negative),
        "unclassified_tweets": int(unclassified),
        "classification_coverage_percent": round(
            100 * classified / len(manifest), 4
        ),
        "binary_label_conflicts": 0,
        "probability_variation_tweets": int(len(variations)),
        "identifier_source": "tweetUrl",
        "duplicate_probability_aggregation": "mean",
        "unclassified_policy": "unknown_not_non_event",
        "verified_incident_count": None,
        "limitations": [
            "Only prefiltered candidate tweets have predictions",
            "The MiniLM classifier was trained on 84 examples",
            "Probabilities are not calibrated",
            "Predicted positive tweets are not verified incidents",
            "Raw-source preprocessing exclusions need review",
        ],
    }

    files = [
        (manifest, OUT / "full_corpus_manifest.parquet"),
        (included, OUT / "unique_tweet_predictions.parquet"),
        (
            excluded,
            OUT / "raw_candidates_excluded_from_processed.parquet",
        ),
        (
            variations,
            OUT / "probability_variation_audit.parquet",
        ),
    ]

    temporary_files = []

    for df, destination in files:
        temp = destination.with_suffix(".tmp.parquet")
        df.to_parquet(temp, index=False)
        temporary_files.append((temp, destination))

    summary_temp = OUT / "coverage_summary.tmp.json"
    summary_final = OUT / "coverage_summary.json"

    summary_temp.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    for temp, destination in temporary_files:
        temp.replace(destination)

    summary_temp.replace(summary_final)

    print("\n" + "=" * 65)
    print("FINAL STAGE 35 SUMMARY")
    print("=" * 65)
    print(json.dumps(summary, indent=2))

    print("\nSTAGE 35 COMPLETED SUCCESSFULLY")


def main():
    print("\n" + "=" * 65)
    print("STAGE 35: FINAL FULL-CORPUS MANIFEST")
    print("=" * 65)

    OUT.mkdir(parents=True, exist_ok=True)

    for path in [SOURCE, CANDIDATES, PREDICTIONS]:
        if not path.exists():
            raise FileNotFoundError(path)

    source = load_source()
    candidate_ids = load_candidate_ids()
    predictions = load_predictions()

    validate_alignment(candidate_ids, predictions)

    source_ids = set(source["tweetId"])
    outside_ids = set(candidate_ids) - source_ids

    validate_reconciliation(len(outside_ids))

    unique, variations = aggregate_predictions(predictions)

    (
        manifest,
        included,
        excluded,
        classified,
        positive,
        negative,
        unclassified,
    ) = build_manifest(source, unique)

    save_outputs(
        manifest,
        included,
        excluded,
        variations,
        candidate_ids,
        predictions,
        classified,
        positive,
        negative,
        unclassified,
    )


if __name__ == "__main__":
    main()
