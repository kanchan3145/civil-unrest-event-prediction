
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
)

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "evaluation"
    / "verified_baseline_predictions.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "evaluation"
    / "baseline_threshold_analysis.csv"
)

THRESHOLDS = [
    0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75,
    0.80
]


def main():
    print("=" * 65)
    print("STAGE 8.2 - BASELINE THRESHOLD ANALYSIS")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)

    y_true = df["target"].astype(int).to_numpy()

    probabilities = (
        df["positive_probability"]
        .astype(float)
        .to_numpy()
    )

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Invalid probability values")

    results = []

    for threshold in THRESHOLDS:
        y_pred = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_true, y_pred
            ),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        })

    result = pd.DataFrame(results)

    print("\nTHRESHOLD COMPARISON:")
    print(
        result.round(3).to_string(index=False)
    )

    # Diagnostic only: selecting the best threshold
    # on these same predictions is optimistic.
    best = result.sort_values(
        ["f1", "recall", "threshold"],
        ascending=[False, False, False]
    ).iloc[0]

    print("\nBEST OBSERVED F1 (EXPLORATORY):")
    print("Threshold:", best["threshold"])
    print("Precision:", round(best["precision"], 4))
    print("Recall:", round(best["recall"], 4))
    print("F1:", round(best["f1"], 4))

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSAVED:", OUTPUT_FILE)
    print("STAGE 8.2 COMPLETED")


if __name__ == "__main__":
    main()
