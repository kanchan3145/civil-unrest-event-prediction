
from pathlib import Path
import pandas as pd

from hybrid_auto_annotate_v2 import classify_v2
from select_reliable_pseudo_labels import select_label

BASE = Path(__file__).resolve().parent.parent / "data" / "annotation"

INPUT_FILE = BASE / "incident_labels_final_v2.csv"
OUTPUT_FILE = BASE / "pseudo_selection_evaluation.csv"


def main():
    print("=" * 65)
    print("STAGE 7.11 - PSEUDO-LABEL SELECTION EVALUATION")
    print("=" * 65)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    predictions = df.apply(classify_v2, axis=1)

    result = pd.concat(
        [
            df.reset_index(drop=True),
            predictions.reset_index(drop=True)
        ],
        axis=1
    )

    selections = result.apply(
        select_label,
        axis=1,
        result_type="expand"
    )

    selections.columns = [
        "selection_status",
        "selection_reason"
    ]

    result = pd.concat(
        [
            result.reset_index(drop=True),
            selections.reset_index(drop=True)
        ],
        axis=1
    )

    selected = result[
        result["selection_status"] == "selected_positive"
    ].copy()

    print("\nSELECTION DISTRIBUTION:")
    print(
        result["selection_status"]
        .value_counts()
        .to_string()
    )

    print("\nVERIFIED CATEGORIES OF SELECTED POSITIVES:")

    if len(selected) > 0:
        print(
            selected["final_event_category"]
            .value_counts(dropna=False)
            .to_string()
        )

        correct = (
            selected["final_event_category"]
            == "actual_incident"
        ).sum()

        print("\nSELECTED POSITIVE AGREEMENT:")
        print(f"{correct}/{len(selected)}")
        print(f"Rate: {correct / len(selected):.2%}")

    else:
        print("No positives selected")

    verified_positives = result[
        result["final_event_category"] == "actual_incident"
    ]

    captured = (
        verified_positives["selection_status"]
        == "selected_positive"
    ).sum()

    print("\nVERIFIED INCIDENT COVERAGE:")
    print(f"{captured}/{len(verified_positives)}")

    print("\nSELECTED POSITIVE DETAILS:")

    for _, row in selected.iterrows():
        print("\nPOST:", row["post_id"])
        print("VERIFIED:", row["final_event_category"])
        print("TEXT:", row["text"])
        print("REASON:", row["selection_reason"])

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSAVED:", OUTPUT_FILE)
    print("STAGE 7.11 COMPLETED")


if __name__ == "__main__":
    main()
