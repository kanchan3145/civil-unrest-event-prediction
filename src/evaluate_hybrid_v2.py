
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from hybrid_auto_annotate_v2 import classify_v2

BASE = Path(__file__).resolve().parent.parent / "data" / "annotation"

INPUT_FILE = BASE / "incident_labels_final_v2.csv"
OUTPUT_FILE = BASE / "hybrid_v2_evaluation.csv"

CATEGORIES = [
    "actual_incident",
    "planned_event",
    "related_action",
    "non_event",
    "uncertain"
]


def main():
    print("=" * 65)
    print("STAGE 7.6 - HYBRID V2 EVALUATION")
    print("=" * 65)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    if not df["post_id"].is_unique:
        raise ValueError("Duplicate post IDs")

    predictions = df.apply(classify_v2, axis=1)

    result = pd.concat(
        [
            df.reset_index(drop=True),
            predictions.reset_index(drop=True)
        ],
        axis=1
    )

    true_labels = result["final_event_category"]
    predicted_labels = result["hybrid_v2_category"]

    print("\nFIVE-CATEGORY CLASSIFICATION REPORT:")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=CATEGORIES,
            zero_division=0
        )
    )

    print("\nCONFUSION MATRIX:")
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=CATEGORIES
    )

    print(
        pd.DataFrame(
            matrix,
            index=CATEGORIES,
            columns=CATEGORIES
        ).to_string()
    )

    true_binary = pd.to_numeric(
        result["final_incident_target"],
        errors="coerce"
    )

    predicted_binary = pd.to_numeric(
        result["hybrid_v2_target"],
        errors="coerce"
    )

    mask = true_binary.notna() & predicted_binary.notna()

    print("\nBINARY EVALUATION:")
    print("Evaluated:", int(mask.sum()))
    print("Excluded:", int((~mask).sum()))

    if mask.any():
        print(
            classification_report(
                true_binary[mask].astype(int),
                predicted_binary[mask].astype(int),
                labels=[0, 1],
                zero_division=0
            )
        )

    result["category_match"] = (
        true_labels == predicted_labels
    )

    print("\nCATEGORY AGREEMENT:")
    print(
        result["category_match"]
        .value_counts()
        .to_string()
    )

    print("\nFALSE NEGATIVE INCIDENTS:")
    missed = result[
        (result["final_event_category"] == "actual_incident")
        & (result["hybrid_v2_category"] != "actual_incident")
    ]

    for _, row in missed.iterrows():
        print("\nPOST:", row["post_id"])
        print("PREDICTED:", row["hybrid_v2_category"])
        print("TEXT:", row["text"])

    print("\nFALSE POSITIVE INCIDENTS:")
    false_positives = result[
        (result["final_event_category"] != "actual_incident")
        & (result["hybrid_v2_category"] == "actual_incident")
    ]

    for _, row in false_positives.iterrows():
        print("\nPOST:", row["post_id"])
        print("VERIFIED:", row["final_event_category"])
        print("TEXT:", row["text"])

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Evaluation file already exists: {OUTPUT_FILE}"
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSAVED:", OUTPUT_FILE)
    print("STAGE 7.6 COMPLETED")


if __name__ == "__main__":
    main()
