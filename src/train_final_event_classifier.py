
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import random

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# STAGE 10 - FINAL MINILM EVENT CLASSIFIER
# ============================================================

SEED = 42

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "annotation"
    / "incident_labels_final_v2.csv"
)

MODEL_DIR = (
    ROOT / "models" / "minilm_event_classifier_v1"
)

CLASSIFIER_FILE = MODEL_DIR / "classifier.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"
TRAINING_MANIFEST_FILE = MODEL_DIR / "training_manifest.csv"

# The encoder is loaded from its Hugging Face model ID.
# The fitted classification model is saved locally.

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1. LOAD HUMAN-VERIFIED TRAINING DATA
# ============================================================

def load_training_data():

    print("\nLoading verified labels...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    required = {
        "post_id",
        "text",
        "final_incident_target"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["post_id"].isna().any():
        raise ValueError("Missing post IDs.")

    if df["post_id"].duplicated().any():
        raise ValueError("Duplicate post IDs.")

    raw_target = df["final_incident_target"]

    target = pd.to_numeric(
        raw_target,
        errors="coerce"
    )

    invalid = (
        raw_target.notna()
        & ~target.isin([0, 1])
    )

    if invalid.any():
        raise ValueError(
            "Unexpected target values found."
        )

    train = df[
        target.isin([0, 1])
    ].copy()

    train["target"] = (
        target.loc[train.index].astype(int)
    )

    if train["text"].isna().any():
        raise ValueError(
            "Missing text in verified training examples."
        )

    train["text"] = (
        train["text"].astype(str)
    )

    if train["text"].str.strip().eq("").any():
        raise ValueError(
            "Empty text in verified training examples."
        )

    train = train.reset_index(drop=True)

    if train["target"].nunique() != 2:
        raise ValueError(
            "Both positive and negative classes required."
        )

    print("Total reviewed:", len(df))
    print("Definite training examples:", len(train))
    print("Excluded uncertain:", len(df) - len(train))

    print("\nClass distribution:")
    print(
        train["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return train


# ============================================================
# 2. CREATE TRAINING FINGERPRINT
# ============================================================

def dataset_fingerprint(df):

    digest = hashlib.sha256()

    records = df[
        ["post_id", "text", "target"]
    ].sort_values("post_id")

    for row in records.itertuples(index=False):
        record = json.dumps(
            [
                str(row.post_id),
                row.text,
                int(row.target)
            ],
            ensure_ascii=False
        )

        digest.update(record.encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()


# ============================================================
# 3. TRAIN MODEL
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 10 - FINAL MINILM EVENT CLASSIFIER")
    print("=" * 65)

    # Never silently overwrite a trained model.
    for path in [
        CLASSIFIER_FILE,
        METADATA_FILE,
        TRAINING_MANIFEST_FILE
    ]:
        if path.exists():
            raise FileExistsError(
                f"Existing output must be preserved: {path}"
            )

    train = load_training_data()

    print("\nLoading MiniLM encoder...")

    encoder = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    texts = train["text"].tolist()
    labels = train["target"].to_numpy()

    print("\nGenerating sentence embeddings...")

    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Embedding shape:", embeddings.shape)

    # Match the semantic baseline's general approach:
    # standardized MiniLM embeddings + balanced logistic regression.
    classifier = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "logistic_regression",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=SEED
            )
        )
    ])

    print("\nTraining Logistic Regression...")

    classifier.fit(
        embeddings,
        labels
    )

    print("Training completed.")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        classifier,
        CLASSIFIER_FILE
    )

    fingerprint = dataset_fingerprint(train)

    manifest = train[
        ["post_id", "target"]
    ].copy()

    manifest.to_csv(
        TRAINING_MANIFEST_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    metadata = {
        "stage": "10",
        "model_version": "minilm_event_classifier_v1",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "encoder": MODEL_NAME,
        "encoder_device": "cpu",
        "normalize_embeddings": True,
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "classifier": (
            "StandardScaler + balanced LogisticRegression"
        ),
        "random_seed": SEED,
        "training_examples": int(len(train)),
        "positive_examples": int(
            (labels == 1).sum()
        ),
        "negative_examples": int(
            (labels == 0).sum()
        ),
        "training_fingerprint_sha256": fingerprint,
        "decision_threshold": 0.50,
        "review_lower_bound": 0.40,
        "review_upper_bound": 0.60,
        "label_0": "non_event",
        "label_1": "actual_incident",
        "policy": "broad_civil_unrest_event_detection",
        "human_verified_training_only": True,
        "pseudo_labels_used": False,
        "note": (
            "Final fit on all definite verified examples. "
            "Training-set predictions are not validation results. "
            "Probability thresholds are provisional and uncalibrated."
        )
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nMODEL SAVED:")
    print(CLASSIFIER_FILE)

    print("\nMETADATA SAVED:")
    print(METADATA_FILE)

    print("\nTRAINING MANIFEST SAVED:")
    print(TRAINING_MANIFEST_FILE)

    print("\nOriginal verified labels were NOT modified.")
    print("STAGE 10 TRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
