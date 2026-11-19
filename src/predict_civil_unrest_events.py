
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ============================================================
# STAGE 10 - REUSABLE CIVIL UNREST EVENT PREDICTION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    ROOT / "models" / "minilm_event_classifier_v1"
)

CLASSIFIER_FILE = MODEL_DIR / "classifier.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"

DEFAULT_INPUT = (
    ROOT / "data" / "annotation"
    / "additional_hybrid_labels_v2.csv"
)

DEFAULT_OUTPUT = (
    ROOT / "data" / "evaluation"
    / "minilm_candidate_predictions_v1.csv"
)

BATCH_SIZE = 16


# ============================================================
# 1. LOAD MODEL
# ============================================================

def load_model():

    if not CLASSIFIER_FILE.exists():
        raise FileNotFoundError(
            "Saved classifier not found. "
            "Run Stage 10 training first."
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing metadata: {METADATA_FILE}"
        )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    print("Loading saved classifier...")

    # Only load joblib artifacts created by your own
    # trusted training script.
    classifier = joblib.load(
        CLASSIFIER_FILE
    )

    print("Loading MiniLM encoder...")

    encoder = SentenceTransformer(
        metadata["encoder"],
        device="cpu"
    )

    return encoder, classifier, metadata


# ============================================================
# 2. PREDICT NEW TWEETS
# ============================================================

def predict_texts(
    texts,
    encoder,
    classifier,
    metadata
):

    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=metadata[
            "normalize_embeddings"
        ]
    )

    expected_dimension = metadata[
        "embedding_dimension"
    ]

    if embeddings.shape[1] != expected_dimension:
        raise ValueError(
            "Embedding dimension does not match "
            "the saved classifier."
        )

    probabilities = classifier.predict_proba(
        embeddings
    )[:, 1]

    threshold = metadata["decision_threshold"]

    lower = metadata["review_lower_bound"]
    upper = metadata["review_upper_bound"]

    predicted_target = (
        probabilities >= threshold
    ).astype(int)

    labels = np.where(
        predicted_target == 1,
        "actual_incident",
        "non_event"
    )

    # This review flag is a provisional triage heuristic,
    # not a calibrated confidence interval.
    review_required = (
        (probabilities >= lower)
        & (probabilities <= upper)
    )

    return (
        probabilities,
        predicted_target,
        labels,
        review_required
    )


# ============================================================
# 3. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict broad civil-unrest events "
            "using the saved MiniLM classifier."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT
    )

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    print("=" * 65)
    print("STAGE 10 - CIVIL UNREST EVENT PREDICTION")
    print("=" * 65)

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    if output_file.exists():
        raise FileExistsError(
            f"Output already exists: {output_file}"
        )

    df = pd.read_csv(
        input_file,
        dtype={"post_id": "string"}
    )

    if "text" not in df.columns:
        raise ValueError(
            "Input CSV must contain a text column."
        )

    if "post_id" in df.columns:
        if df["post_id"].isna().any():
            raise ValueError("Missing post IDs.")

        if df["post_id"].duplicated().any():
            raise ValueError("Duplicate post IDs.")

    if len(df) == 0:
        raise ValueError("Input CSV is empty.")

    print("\nInput tweets:", len(df))

    encoder, classifier, metadata = load_model()

    texts = (
        df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("\nGenerating predictions...")

    (
        probabilities,
        targets,
        labels,
        review_flags
    ) = predict_texts(
        texts,
        encoder,
        classifier,
        metadata
    )

    result = df.copy()

    result["minilm_positive_probability"] = (
        probabilities
    )

    result["minilm_predicted_target"] = (
        targets
    )

    result["minilm_predicted_category"] = (
        labels
    )

    result["minilm_review_required"] = (
        review_flags
    )

    result["minilm_model_version"] = (
        metadata["model_version"]
    )

    result["minilm_human_verified"] = False

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPREDICTION DISTRIBUTION:")
    print(
        result["minilm_predicted_category"]
        .value_counts()
        .to_string()
    )

    print("\nREVIEW FLAGS:")
    print(
        result["minilm_review_required"]
        .value_counts()
        .to_string()
    )

    print("\nPROBABILITY STATISTICS:")
    print(
        result["minilm_positive_probability"]
        .describe()
        .to_string()
    )

    print("\nOUTPUT SAVED:")
    print(output_file)

    print("\nExisting labels were NOT modified.")
    print("STAGE 10 PREDICTION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
