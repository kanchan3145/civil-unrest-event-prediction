
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "annotation"
    / "incident_labels_final_v2.csv"
)

OUTPUT_DIR = ROOT / "data" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = (
    OUTPUT_DIR / "verified_baseline_predictions.csv"
)

METRICS_FILE = (
    OUTPUT_DIR / "verified_baseline_metrics.json"
)

SEED = 42


# --------------------------------------------------
# 2. TEXT NORMALIZATION
# --------------------------------------------------

def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# 3. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_data():
    print("\nLoading verified dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    required = [
        "post_id",
        "text",
        "final_incident_target"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Total reviewed tweets:", len(df))

    df["final_incident_target"] = pd.to_numeric(
        df["final_incident_target"],
        errors="coerce"
    )

    invalid = (
        df["final_incident_target"].notna()
        & ~df["final_incident_target"].isin([0, 1])
    )

    if invalid.any():
        raise ValueError(
            "Found binary targets other than 0 or 1"
        )

    uncertain_count = (
        df["final_incident_target"].isna().sum()
    )

    print("Uncertain/unlabeled:", uncertain_count)

    verified = df[
        df["final_incident_target"].isin([0, 1])
    ].copy()

    verified["clean_text"] = (
        verified["text"].apply(normalize_text)
    )

    empty_count = (
        verified["clean_text"] == ""
    ).sum()

    print("Empty verified texts:", empty_count)

    verified = verified[
        verified["clean_text"] != ""
    ].copy()

    duplicate_ids = (
        verified["post_id"].duplicated().sum()
    )

    print("Duplicate post IDs:", duplicate_ids)

    if duplicate_ids:
        raise ValueError(
            "Duplicate post IDs found. Resolve before training."
        )

    duplicate_texts = (
        verified["clean_text"].duplicated().sum()
    )

    print("Exact duplicate texts:", duplicate_texts)

    if duplicate_texts:
        raise ValueError(
            "Duplicate normalized texts found. "
            "Resolve before cross-validation."
        )

    verified["target"] = (
        verified["final_incident_target"].astype(int)
    )

    print("\nVerified class distribution:")
    print(
        verified["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if verified["target"].nunique() != 2:
        raise ValueError(
            "Both binary classes are required."
        )

    if verified["target"].value_counts().min() < 4:
        raise ValueError(
            "At least four examples per class "
            "are needed for four-fold CV."
        )

    return verified.reset_index(drop=True)


# --------------------------------------------------
# 4. BUILD MODEL
# --------------------------------------------------

def build_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=SEED
            )
        )
    ])


# --------------------------------------------------
# 5. CROSS-VALIDATION
# --------------------------------------------------

def evaluate_model(df):
    X = df["clean_text"]
    y = df["target"]

    cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=SEED
    )

    predictions = np.zeros(
        len(df),
        dtype=int
    )

    probabilities = np.zeros(
        len(df),
        dtype=float
    )

    fold_numbers = np.zeros(
        len(df),
        dtype=int
    )

    print("\nRunning four-fold cross-validation...")

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):
        model = build_model()

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        predictions[test_idx] = model.predict(
            X.iloc[test_idx]
        )

        probabilities[test_idx] = (
            model.predict_proba(
                X.iloc[test_idx]
            )[:, 1]
        )

        fold_numbers[test_idx] = fold

        print(
            f"Fold {fold}: "
            f"train={len(train_idx)}, "
            f"test={len(test_idx)}"
        )

    return predictions, probabilities, fold_numbers


# --------------------------------------------------
# 6. METRICS AND OUTPUT
# --------------------------------------------------

def save_results(
    df,
    predictions,
    probabilities,
    fold_numbers
):
    y_true = df["target"].to_numpy()

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1]
            ).tolist()
        ),
        "sample_count": int(len(df)),
        "positive_count": int(sum(y_true == 1)),
        "negative_count": int(sum(y_true == 0)),
        "cv_folds": 4,
        "seed": SEED
    }

    print("\n" + "=" * 65)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 65)

    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=[
                "Non-event",
                "Actual incident"
            ],
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        )
    )

    print("\nAccuracy:", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1-score:", round(metrics["f1"], 4))

    output = df.copy()

    output["cv_fold"] = fold_numbers
    output["predicted_target"] = predictions
    output["positive_probability"] = probabilities

    output["correct_prediction"] = (
        output["target"] == output["predicted_target"]
    )

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

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nSaved predictions:", PREDICTIONS_FILE)
    print("Saved metrics:", METRICS_FILE)


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():
    print("=" * 65)
    print("STAGE 8.1 - VERIFIED MACHINE LEARNING BASELINE")
    print("=" * 65)

    np.random.seed(SEED)

    df = load_data()

    predictions, probabilities, folds = (
        evaluate_model(df)
    )

    save_results(
        df,
        predictions,
        probabilities,
        folds
    )

    print("\nSTAGE 8.1 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
