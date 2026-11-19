
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# STAGE 19 - FULL CANDIDATE CLASSIFICATION WITH MINILM
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "structured"
    / "full_event_candidates_v1.csv"
)

MODEL_DIR = (
    ROOT / "models" / "minilm_event_classifier_v1"
)

CLASSIFIER_FILE = MODEL_DIR / "classifier.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"

OUTPUT_DIR = ROOT / "data" / "structured"

OUTPUT_FILE = (
    OUTPUT_DIR / "full_event_predictions_v1.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "full_event_predictions_summary_v1.json"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR / "stage19_checkpoints_v1"
)

CHUNK_SIZE = 256
EMBEDDING_BATCH_SIZE = 16

EXTRA_COLUMNS = [
    "minilm_positive_probability",
    "minilm_predicted_target",
    "minilm_predicted_category",
    "minilm_review_required",
    "minilm_model_version",
    "minilm_human_verified"
]


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

def load_model():

    if not CLASSIFIER_FILE.exists():
        raise FileNotFoundError(CLASSIFIER_FILE)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(METADATA_FILE)

    metadata = json.loads(
        METADATA_FILE.read_text(encoding="utf-8")
    )

    required = {
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "label_0": "non_event",
        "label_1": "actual_incident"
    }

    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Unexpected metadata {key}: "
                f"{metadata.get(key)}"
            )

    print("Loading saved classifier...", flush=True)

    # Only load model artifacts produced by your own
    # trusted training script.
    classifier = joblib.load(CLASSIFIER_FILE)

    if not hasattr(classifier, "predict_proba"):
        raise ValueError(
            "Classifier does not support predict_proba."
        )

    if not hasattr(classifier, "classes_"):
        raise ValueError(
            "Classifier has no classes_ attribute."
        )

    classes = list(classifier.classes_)

    if set(classes) != {0, 1}:
        raise ValueError(
            f"Unexpected classifier classes: {classes}"
        )

    positive_index = classes.index(1)

    print("Loading MiniLM encoder...", flush=True)

    encoder = SentenceTransformer(
        metadata["encoder"],
        device="cpu"
    )

    return (
        encoder,
        classifier,
        metadata,
        positive_index
    )


# ============================================================
# 2. CLASSIFY A CHUNK
# ============================================================

def classify_chunk(
    chunk,
    encoder,
    classifier,
    metadata,
    positive_index
):

    texts = (
        chunk["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = encoder.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=metadata[
            "normalize_embeddings"
        ]
    )

    if embeddings.shape != (
        len(texts),
        metadata["embedding_dimension"]
    ):
        raise ValueError(
            "Unexpected embedding dimensions."
        )

    probabilities = classifier.predict_proba(
        embeddings
    )[:, positive_index]

    threshold = float(
        metadata["decision_threshold"]
    )

    lower = float(
        metadata["review_lower_bound"]
    )

    upper = float(
        metadata["review_upper_bound"]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    review_flags = (
        (probabilities >= lower)
        & (probabilities <= upper)
    )

    result = chunk.copy()

    result["minilm_positive_probability"] = (
        probabilities
    )

    result["minilm_predicted_target"] = (
        predictions
    )

    result["minilm_predicted_category"] = np.where(
        predictions == 1,
        "actual_incident",
        "non_event"
    )

    result["minilm_review_required"] = (
        review_flags
    )

    result["minilm_model_version"] = (
        metadata["model_version"]
    )

    result["minilm_human_verified"] = False

    return result


# ============================================================
# 3. CHECKPOINT HELPERS
# ============================================================

def part_path(chunk_number):

    return (
        CHECKPOINT_DIR
        / f"part_{chunk_number:06d}.csv"
    )


def file_fingerprint(path):

    stat = path.stat()

    return {
        "resolved_path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns
    }


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def prepare_checkpoint(metadata):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_file = (
        CHECKPOINT_DIR / "manifest.json"
    )

    manifest = {
        "input": file_fingerprint(INPUT_FILE),
        "classifier_sha256": sha256_file(
            CLASSIFIER_FILE
        ),
        "metadata_sha256": sha256_file(
            METADATA_FILE
        ),
        "chunk_size": CHUNK_SIZE,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "model_version": metadata["model_version"],
        "output_columns": EXTRA_COLUMNS
    }

    if manifest_file.exists():

        old_manifest = json.loads(
            manifest_file.read_text(
                encoding="utf-8"
            )
        )

        if old_manifest != manifest:
            raise ValueError(
                "Checkpoint settings or input/model "
                "files have changed. Existing checkpoints "
                "were preserved. Use a new versioned "
                "checkpoint directory."
            )

    else:

        if list(CHECKPOINT_DIR.glob("part_*.csv")):
            raise ValueError(
                "Checkpoint parts exist without a "
                "manifest. Refusing unsafe resume."
            )

        with manifest_file.open(
            "x",
            encoding="utf-8"
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2
            )


def save_part(result, chunk_number):

    final_path = part_path(chunk_number)

    if final_path.exists():
        raise FileExistsError(final_path)

    temporary_path = final_path.with_suffix(
        ".csv.tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    result.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8"
    )

    # A completed chunk becomes visible atomically.
    os.replace(
        temporary_path,
        final_path
    )


def validate_part(path, expected_rows, expected_columns):

    part = pd.read_csv(
        path,
        usecols=expected_columns,
        dtype=str,
        keep_default_na=False
    )

    if len(part) != expected_rows:
        raise ValueError(
            f"Invalid checkpoint row count: {path}"
        )

    return part


# ============================================================
# 4. COMBINE SAVED CHUNKS
# ============================================================

def combine_parts(
    total_chunks,
    input_columns,
    expected_rows
):

    print(
        "\nCombining checkpoint files...",
        flush=True
    )

    all_columns = input_columns + EXTRA_COLUMNS

    counts = Counter()

    probability_sum = 0.0
    probability_count = 0

    review_count = 0
    total_rows = 0

    # Exclusive creation prevents overwriting a
    # previous completed prediction file.
    with OUTPUT_FILE.open(
        "x",
        newline="",
        encoding="utf-8"
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=all_columns
        )

        writer.writeheader()

        for chunk_number in range(total_chunks):

            path = part_path(chunk_number)

            if not path.exists():
                raise FileNotFoundError(path)

            with path.open(
                "r",
                newline="",
                encoding="utf-8"
            ) as source:

                reader = csv.DictReader(source)

                if reader.fieldnames != all_columns:
                    raise ValueError(
                        f"Column mismatch: {path}"
                    )

                for row in reader:

                    writer.writerow(row)

                    total_rows += 1

                    label = row[
                        "minilm_predicted_category"
                    ]

                    counts[label] += 1

                    probability_sum += float(
                        row[
                            "minilm_positive_probability"
                        ]
                    )

                    probability_count += 1

                    if (
                        row["minilm_review_required"]
                        .lower() == "true"
                    ):
                        review_count += 1

    if total_rows != expected_rows:
        raise ValueError(
            "Combined output row count mismatch. "
            f"Expected {expected_rows}, got {total_rows}."
        )

    return {
        "total_candidates_classified": total_rows,
        "prediction_counts": dict(counts),
        "review_required": review_count,
        "average_positive_probability": round(
            probability_sum / max(
                probability_count, 1
            ),
            6
        )
    }


# ============================================================
# 5. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help=(
            "Limit new chunks processed in this run. "
            "0 means process all remaining chunks."
        )
    )

    args = parser.parse_args()

    print("=" * 65)
    print("STAGE 19 - FULL DATASET CANDIDATE CLASSIFICATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if OUTPUT_FILE.exists() or SUMMARY_FILE.exists():
        raise FileExistsError(
            "Stage 19 final output already exists. "
            "Existing outputs were not overwritten."
        )

    start_time = time.time()

    encoder, classifier, metadata, positive_index = (
        load_model()
    )

    prepare_checkpoint(metadata)

    input_columns = list(
        pd.read_csv(
            INPUT_FILE,
            nrows=0
        ).columns
    )

    if "text" not in input_columns:
        raise ValueError(
            "Input file has no text column."
        )

    total_processed = 0
    completed_chunks = 0
    new_chunks = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=False,
            low_memory=False
        )
    ):

        completed_chunks += 1

        checkpoint_file = part_path(
            chunk_number
        )

        if checkpoint_file.exists():

            validate_part(
                checkpoint_file,
                len(chunk),
                input_columns + EXTRA_COLUMNS
            )

        else:

            if (
                args.max_chunks > 0
                and new_chunks >= args.max_chunks
            ):
                print(
                    "\nRun limit reached. "
                    "Rerun the same command to resume.",
                    flush=True
                )
                return

            result = classify_chunk(
                chunk,
                encoder,
                classifier,
                metadata,
                positive_index
            )

            save_part(
                result,
                chunk_number
            )

            new_chunks += 1

        total_processed += len(chunk)

        if (
            completed_chunks % 20 == 0
            or new_chunks == 1
        ):
            print(
                f"Chunks completed: {completed_chunks} | "
                f"Candidates covered: {total_processed:,} | "
                f"New chunks this run: {new_chunks}",
                flush=True
            )

    # The output is created only after every
    # candidate has a completed checkpoint.
    summary = combine_parts(
        completed_chunks,
        input_columns,
        total_processed
    )

    summary.update({
        "stage": 19,
        "model_version": metadata[
            "model_version"
        ],
        "encoder": metadata["encoder"],
        "training_examples": metadata[
            "training_examples"
        ],
        "decision_threshold": metadata[
            "decision_threshold"
        ],
        "review_lower_bound": metadata[
            "review_lower_bound"
        ],
        "review_upper_bound": metadata[
            "review_upper_bound"
        ],
        "completed_chunks": completed_chunks,
        "new_chunks_this_run": new_chunks,
        "processing_seconds_this_run": round(
            time.time() - start_time,
            2
        ),
        "limitations": [
            "Classifier trained on only 84 verified examples",
            "Only 16 positive training examples",
            "Probabilities are not calibrated",
            "Predicted incidents are not verified incidents",
            "Keyword prefilter can miss relevant events",
            "Predictions do not establish event time or location",
            "No human review performed in this stage"
        ]
    })

    with SUMMARY_FILE.open(
        "x",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2
        )

    print("\n" + "=" * 65)
    print("STAGE 19 COMPLETE")
    print("=" * 65)

    print(json.dumps(
        summary,
        indent=2
    ))

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
