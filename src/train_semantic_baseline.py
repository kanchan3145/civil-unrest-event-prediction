

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "annotation"
    / "incident_labels_final_v2.csv"
)

OUTPUT_DIR = ROOT / "data" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = (
    OUTPUT_DIR / "semantic_baseline_predictions.csv"
)

METRICS_FILE = (
    OUTPUT_DIR / "semantic_baseline_metrics.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42


def main():
    print("=" * 65)
    print("STAGE 8.4 - SEMANTIC BASELINE")
    print("=" * 65)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    df["final_incident_target"] = pd.to_numeric(
        df["final_incident_target"],
        errors="coerce"
    )

    df = df[
        df["final_incident_target"].isin([0, 1])
    ].copy().reset_index(drop=True)

    df["text"] = df["text"].fillna("").astype(str)

    X_text = df["text"].tolist()
    y = df["final_incident_target"].astype(int).to_numpy()

    print("\nVerified examples:", len(df))
    print("Positive:", int(sum(y == 1)))
    print("Negative:", int(sum(y == 0)))

    print("\nLoading sentence embedding model...")

    encoder = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    print("\nGenerating embeddings...")

    embeddings = encoder.encode(
        X_text,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Embedding shape:", embeddings.shape)

    cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=SEED
    )

    predictions = np.zeros(len(df), dtype=int)
    probabilities = np.zeros(len(df), dtype=float)
    fold_numbers = np.zeros(len(df), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(embeddings, y),
        start=1
    ):
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED
        )

        classifier.fit(
            embeddings[train_idx],
            y[train_idx]
        )

        predictions[test_idx] = classifier.predict(
            embeddings[test_idx]
        )

        probabilities[test_idx] = (
            classifier.predict_proba(
                embeddings[test_idx]
            )[:, 1]
        )

        fold_numbers[test_idx] = fold

        print(f"Fold {fold} completed")

    print("\nCLASSIFICATION REPORT:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Non-event", "Actual incident"],
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    print("\nCONFUSION MATRIX:")
    print(matrix)

    metrics = {
        "model": MODEL_NAME,
        "sample_count": len(df),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "confusion_matrix": matrix.tolist(),
        "cv_folds": 4,
        "seed": SEED
    }

    output = df.copy()
    output["cv_fold"] = fold_numbers
    output["predicted_target"] = predictions
    output["positive_probability"] = probabilities

    if PREDICTIONS_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {PREDICTIONS_FILE}"
        )

    if METRICS_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {METRICS_FILE}"
        )

    output.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nSAVED:", PREDICTIONS_FILE)
    print("SAVED:", METRICS_FILE)
    print("STAGE 8.4 COMPLETED")


if __name__ == "__main__":
    main()
