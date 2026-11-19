
from pathlib import Path
import hashlib
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer


# ============================================================
# STAGE 36 — FULL-CORPUS MINILM INFERENCE
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data/processed/tweets_processed.parquet"
STAGE35 = ROOT / "outputs/stage35/full_corpus_manifest.parquet"

MODEL_DIR = ROOT / "models/minilm_event_classifier_v1"
CLASSIFIER_PATH = MODEL_DIR / "classifier.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

OUT = ROOT / "outputs/stage36"
CHECKPOINT_DIR = OUT / "checkpoints"

RUN_CONFIG = OUT / "run_config.json"
FINAL_OUTPUT = OUT / "full_corpus_predictions.parquet"
FINAL_SUMMARY = OUT / "stage36_summary.json"

# CPU-friendly settings.
EMBEDDING_BATCH_SIZE = 16
CHECKPOINT_SIZE = 512
PARQUET_READ_BATCH_SIZE = 4096

ID_COLUMN = "tweetId"
PROB = "minilm_positive_probability"
TARGET = "minilm_predicted_target"

SEED = 42


# ============================================================
# UTILITIES
# ============================================================

def log(message):
    print(message, flush=True)


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_json(data, destination):
    temporary = destination.with_suffix(".tmp.json")

    temporary.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    os.replace(temporary, destination)


def atomic_parquet(frame, destination):
    temporary = destination.with_suffix(".tmp.parquet")

    frame.to_parquet(
        temporary,
        index=False,
    )

    os.replace(temporary, destination)


def validate_ids(series, name):
    values = series.astype("string").str.strip()

    valid = values.str.fullmatch(
        r"[0-9]{1,20}"
    ).fillna(False)

    if not valid.all():
        raise ValueError(
            f"{name}: invalid tweet IDs detected"
        )

    return values


# ============================================================
# LOAD AND VALIDATE EXISTING ARTIFACTS
# ============================================================

def load_existing_manifest():
    log("\n[1/8] Loading Stage 35 manifest...")

    manifest = pd.read_parquet(STAGE35)

    required = [
        ID_COLUMN,
        "classification_status",
        TARGET,
        PROB,
    ]

    missing = [
        column
        for column in required
        if column not in manifest.columns
    ]

    if missing:
        raise ValueError(
            f"Stage 35 manifest missing columns: {missing}"
        )

    manifest[ID_COLUMN] = validate_ids(
        manifest[ID_COLUMN],
        "Stage 35",
    )

    if manifest[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate IDs in Stage 35 manifest"
        )

    already_classified = (
        manifest["classification_status"]
        == "classified_minilm"
    )

    unclassified = (
        manifest["classification_status"]
        == "not_classified_prefilter_excluded"
    )

    if not (already_classified | unclassified).all():
        raise ValueError(
            "Unexpected Stage 35 classification statuses"
        )

    if manifest.loc[
        already_classified, TARGET
    ].isna().any():
        raise ValueError(
            "Previously classified tweets have missing labels"
        )

    if manifest.loc[
        unclassified, TARGET
    ].notna().any():
        raise ValueError(
            "Unclassified tweets unexpectedly have labels"
        )

    log(f"Total processed tweets: {len(manifest):,}")
    log(
        f"Already classified: "
        f"{int(already_classified.sum()):,}"
    )
    log(
        f"Remaining: "
        f"{int(unclassified.sum()):,}"
    )

    return manifest, unclassified


def load_model():
    log("\n[2/8] Loading existing MiniLM model...")

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(CLASSIFIER_PATH)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    encoder_name = metadata.get("encoder")
    threshold = float(
        metadata.get("decision_threshold", 0.5)
    )

    lower = float(
        metadata.get("review_lower_bound", 0.4)
    )

    upper = float(
        metadata.get("review_upper_bound", 0.6)
    )

    if not encoder_name:
        raise ValueError(
            "Encoder name missing from model metadata"
        )

    if not (0 <= lower <= threshold <= upper <= 1):
        raise ValueError(
            "Invalid model probability thresholds"
        )

    classifier = joblib.load(CLASSIFIER_PATH)

    if not hasattr(classifier, "predict_proba"):
        raise TypeError(
            "Saved classifier does not support predict_proba"
        )

    encoder = SentenceTransformer(
        encoder_name,
        device="cpu",
    )

    dimension = encoder.get_sentence_embedding_dimension()

    expected_dimension = metadata.get(
        "embedding_dimension"
    )

    if (
        expected_dimension is not None
        and dimension != int(expected_dimension)
    ):
        raise ValueError(
            "Encoder dimension differs from model metadata"
        )

    expected_input = getattr(
        classifier,
        "n_features_in_",
        None,
    )

    if (
        expected_input is not None
        and int(expected_input) != dimension
    ):
        raise ValueError(
            "Classifier input dimension mismatch"
        )

    classes = getattr(classifier, "classes_", None)

    if classes is None:
        raise ValueError(
            "Classifier classes_ attribute missing"
        )

    classes = list(classes)

    if set(classes) != {0, 1}:
        raise ValueError(
            f"Unexpected classifier classes: {classes}"
        )

    positive_index = classes.index(1)

    log(f"Encoder: {encoder_name}")
    log(f"Embedding dimension: {dimension}")
    log(f"Decision threshold: {threshold}")
    log(f"Model version: {metadata.get('model_version')}")

    return (
        encoder,
        classifier,
        metadata,
        positive_index,
        threshold,
        lower,
        upper,
    )


# ============================================================
# RESUME SAFETY
# ============================================================

def initialize_run(metadata, total_remaining):
    log("\n[3/8] Checking run configuration...")

    config = {
        "stage": 36,
        "processed_sha256": sha256_file(PROCESSED),
        "stage35_manifest_sha256": sha256_file(STAGE35),
        "classifier_sha256": sha256_file(CLASSIFIER_PATH),
        "metadata_sha256": sha256_file(METADATA_PATH),
        "model_version": metadata.get("model_version"),
        "encoder": metadata.get("encoder"),
        "normalize_embeddings": metadata.get(
            "normalize_embeddings", True
        ),
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "checkpoint_size": CHECKPOINT_SIZE,
        "remaining_tweets": total_remaining,
    }

    if RUN_CONFIG.exists():
        existing = json.loads(
            RUN_CONFIG.read_text(encoding="utf-8")
        )

        if existing != config:
            raise ValueError(
                "Stage 36 inputs or configuration changed. "
                "Existing checkpoints cannot safely be reused. "
                "Inspect outputs/stage36 before starting a new run."
            )

        log("Existing run configuration matches. Resume enabled.")

    else:
        if list(CHECKPOINT_DIR.glob("part_*.parquet")):
            raise ValueError(
                "Checkpoints exist without run_config.json. "
                "Refusing unsafe resume."
            )

        atomic_json(config, RUN_CONFIG)
        log("New run configuration saved.")

    return config


def checkpoint_path(index):
    return CHECKPOINT_DIR / f"part_{index:06d}.parquet"


# ============================================================
# INFERENCE
# ============================================================

def predict_batch(
    ids,
    texts,
    encoder,
    classifier,
    metadata,
    positive_index,
    threshold,
    lower,
    upper,
):
    embeddings = encoder.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=bool(
            metadata.get("normalize_embeddings", True)
        ),
        device="cpu",
    )

    probabilities = classifier.predict_proba(
        embeddings
    )[:, positive_index]

    labels = (
        probabilities >= threshold
    ).astype(np.int8)

    review = (
        (probabilities >= lower)
        & (probabilities <= upper)
    )

    return pd.DataFrame({
        ID_COLUMN: pd.Series(ids, dtype="string"),
        PROB: probabilities.astype(np.float32),
        TARGET: labels,
        "minilm_predicted_category": np.where(
            labels == 1,
            "actual_incident",
            "non_event",
        ),
        "minilm_review_required": review,
        "minilm_model_version": metadata.get(
            "model_version"
        ),
        "minilm_human_verified": False,
    })


def process_chunk(
    chunk_index,
    ids,
    texts,
    encoder,
    classifier,
    metadata,
    positive_index,
    threshold,
    lower,
    upper,
):
    destination = checkpoint_path(chunk_index)

    if destination.exists():
        existing = pd.read_parquet(destination)

        expected_ids = pd.Series(
            ids,
            dtype="string",
        )

        actual_ids = existing[ID_COLUMN].astype(
            "string"
        )

        if not actual_ids.reset_index(drop=True).equals(
            expected_ids.reset_index(drop=True)
        ):
            raise ValueError(
                f"Checkpoint {chunk_index} has incorrect IDs"
            )

        if (
            len(existing) != len(ids)
            or existing[ID_COLUMN].duplicated().any()
            or existing[PROB].isna().any()
            or not existing[PROB].between(0, 1).all()
            or not existing[TARGET].isin([0, 1]).all()
        ):
            raise ValueError(
                f"Checkpoint {chunk_index} failed validation"
            )

        log(
            f"Chunk {chunk_index}: "
            f"resumed {len(ids)} predictions"
        )

        return

    output = predict_batch(
        ids,
        texts,
        encoder,
        classifier,
        metadata,
        positive_index,
        threshold,
        lower,
        upper,
    )

    atomic_parquet(output, destination)

    log(
        f"Chunk {chunk_index}: "
        f"saved {len(output)} predictions"
    )


def run_inference(
    manifest,
    unclassified,
    encoder,
    classifier,
    metadata,
    positive_index,
    threshold,
    lower,
    upper,
):
    log("\n[4/8] Running CPU inference...")

    remaining_ids = set(
        manifest.loc[
            unclassified,
            ID_COLUMN,
        ].astype("string")
    )

    found_ids = set()

    buffer_ids = []
    buffer_texts = []

    chunk_index = 0
    scanned_rows = 0

    parquet_file = pq.ParquetFile(PROCESSED)

    for batch in parquet_file.iter_batches(
        batch_size=PARQUET_READ_BATCH_SIZE,
        columns=["post_id", "text"],
    ):
        frame = batch.to_pandas()

        frame["post_id"] = frame["post_id"].astype(
            "string"
        )

        scanned_rows += len(frame)

        selected = frame[
            frame["post_id"].isin(remaining_ids)
        ]

        for tweet_id, tweet_text in zip(
            selected["post_id"],
            selected["text"],
        ):
            if tweet_id in found_ids:
                raise ValueError(
                    f"Duplicate processed tweet ID: {tweet_id}"
                )

            found_ids.add(tweet_id)

            buffer_ids.append(tweet_id)

            if pd.isna(tweet_text):
                buffer_texts.append("")
            else:
                buffer_texts.append(str(tweet_text))

            if len(buffer_ids) == CHECKPOINT_SIZE:
                process_chunk(
                    chunk_index,
                    buffer_ids,
                    buffer_texts,
                    encoder,
                    classifier,
                    metadata,
                    positive_index,
                    threshold,
                    lower,
                    upper,
                )

                chunk_index += 1
                buffer_ids = []
                buffer_texts = []

        if scanned_rows % 100000 < PARQUET_READ_BATCH_SIZE:
            log(
                f"Scanned {scanned_rows:,} processed rows; "
                f"selected {len(found_ids):,} unclassified tweets"
            )

    if buffer_ids:
        process_chunk(
            chunk_index,
            buffer_ids,
            buffer_texts,
            encoder,
            classifier,
            metadata,
            positive_index,
            threshold,
            lower,
            upper,
        )

        chunk_index += 1

    if found_ids != remaining_ids:
        missing = remaining_ids - found_ids

        raise ValueError(
            f"{len(missing)} unclassified tweet IDs "
            "were not found in processed Parquet"
        )

    expected_files = {
        checkpoint_path(i)
        for i in range(chunk_index)
    }

    actual_files = set(
        CHECKPOINT_DIR.glob("part_*.parquet")
    )

    if actual_files != expected_files:
        raise ValueError(
            "Checkpoint file set does not match "
            "the expected inference chunks"
        )

    log(
        f"\nInference complete: "
        f"{len(found_ids):,} tweets, "
        f"{chunk_index:,} checkpoints"
    )

    return chunk_index


# ============================================================
# COMBINE AND VALIDATE
# ============================================================

def combine_results(
    manifest,
    unclassified,
    chunk_count,
):
    log("\n[5/8] Combining checkpoint predictions...")

    parts = []

    for index in range(chunk_count):
        part = pd.read_parquet(
            checkpoint_path(index)
        )

        parts.append(part)

    if parts:
        new_predictions = pd.concat(
            parts,
            ignore_index=True,
        )
    else:
        new_predictions = pd.DataFrame({
            ID_COLUMN: pd.Series(dtype="string"),
            PROB: pd.Series(dtype="float32"),
            TARGET: pd.Series(dtype="int8"),
        })

    if new_predictions[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate tweet IDs in new predictions"
        )

    expected_ids = set(
        manifest.loc[
            unclassified,
            ID_COLUMN,
        ]
    )

    actual_ids = set(
        new_predictions[ID_COLUMN].astype("string")
    )

    if actual_ids != expected_ids:
        raise ValueError(
            "New predictions do not cover exactly "
            "the previously unclassified tweets"
        )

    log(
        f"New predictions loaded: "
        f"{len(new_predictions):,}"
    )

    log("\n[6/8] Integrating previous classifications...")

    old = manifest.loc[
        ~unclassified
    ].copy()

    new = manifest.loc[
        unclassified,
        [ID_COLUMN],
    ].merge(
        new_predictions,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    old["classification_source"] = (
        "stage35_candidate_minilm"
    )

    new["classification_source"] = (
        "stage36_full_corpus_minilm"
    )

    # Keep a consistent schema for both old and new records.
    columns = [
        ID_COLUMN,
        PROB,
        TARGET,
        "minilm_predicted_category",
        "minilm_review_required",
        "minilm_model_version",
        "minilm_human_verified",
        "classification_source",
    ]

    final = pd.concat(
        [
            old[columns],
            new[columns],
        ],
        ignore_index=True,
    )

    log("\n[7/8] Validating full coverage...")

    if len(final) != len(manifest):
        raise ValueError(
            "Final row count differs from processed corpus"
        )

    if final[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate IDs in final classification"
        )

    if set(final[ID_COLUMN]) != set(
        manifest[ID_COLUMN]
    ):
        raise ValueError(
            "Final classification IDs do not match "
            "the Stage 35 processed corpus"
        )

    if final[TARGET].isna().any():
        raise ValueError(
            "Missing final classification labels"
        )

    if final[PROB].isna().any():
        raise ValueError(
            "Missing final classification probabilities"
        )

    if not final[TARGET].isin([0, 1]).all():
        raise ValueError(
            "Invalid final classification labels"
        )

    return final, len(new_predictions)


def save_final(final, new_count, metadata, elapsed):
    log("\n[8/8] Saving full-corpus predictions...")

    positives = int(
        (final[TARGET] == 1).sum()
    )

    negatives = int(
        (final[TARGET] == 0).sum()
    )

    summary = {
        "stage": 36,
        "total_processed_tweets": int(len(final)),
        "previously_classified_tweets": int(
            len(final) - new_count
        ),
        "newly_classified_tweets": int(new_count),
        "total_classified_tweets": int(len(final)),
        "classification_coverage_percent": 100.0,
        "predicted_positive_tweets": positives,
        "predicted_negative_tweets": negatives,
        "model_version": metadata.get("model_version"),
        "encoder": metadata.get("encoder"),
        "device": "cpu",
        "inference_seconds_this_run": round(elapsed, 2),
        "limitations": [
            "Classifier trained on only 84 verified examples",
            "Model probabilities are uncalibrated",
            "Predictions are not verified civil unrest incidents",
            "Existing candidate predictions are reused",
            "Raw-source tweets excluded from processed corpus "
            "are not part of this classification coverage",
            "Independent validation is required",
        ],
    }

    atomic_parquet(
        final,
        FINAL_OUTPUT,
    )

    atomic_json(
        summary,
        FINAL_SUMMARY,
    )

    log("\n" + "=" * 65)
    log("STAGE 36: FINAL SUMMARY")
    log("=" * 65)

    log(json.dumps(summary, indent=2))

    log(f"\nSaved: {FINAL_OUTPUT}")
    log(f"Saved: {FINAL_SUMMARY}")

    log("\nSTAGE 36 COMPLETED SUCCESSFULLY")


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(SEED)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("\n" + "=" * 65)
    log("STAGE 36: FULL-CORPUS MINILM CLASSIFICATION")
    log("=" * 65)

    for path in [
        PROCESSED,
        STAGE35,
        CLASSIFIER_PATH,
        METADATA_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    start = time.time()

    manifest, unclassified = load_existing_manifest()

    (
        encoder,
        classifier,
        metadata,
        positive_index,
        threshold,
        lower,
        upper,
    ) = load_model()

    initialize_run(
        metadata,
        int(unclassified.sum()),
    )

    chunk_count = run_inference(
        manifest,
        unclassified,
        encoder,
        classifier,
        metadata,
        positive_index,
        threshold,
        lower,
        upper,
    )

    final, new_count = combine_results(
        manifest,
        unclassified,
        chunk_count,
    )

    save_final(
        final,
        new_count,
        metadata,
        time.time() - start,
    )


if __name__ == "__main__":
    main()
