
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "evaluation"
    / "verified_baseline_predictions.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "evaluation"
    / "baseline_error_analysis.csv"
)

THRESHOLD = 0.45


def main():
    print("=" * 65)
    print("STAGE 8.3 - BASELINE ERROR ANALYSIS")
    print("=" * 65)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    df["predicted_at_045"] = (
        df["positive_probability"] >= THRESHOLD
    ).astype(int)

    df["error_type"] = "correct"

    df.loc[
        (df["target"] == 1)
        & (df["predicted_at_045"] == 0),
        "error_type"
    ] = "false_negative"

    df.loc[
        (df["target"] == 0)
        & (df["predicted_at_045"] == 1),
        "error_type"
    ] = "false_positive"

    print("\nERROR DISTRIBUTION:")
    print(
        df["error_type"]
        .value_counts()
        .to_string()
    )

    for error_type in [
        "false_negative",
        "false_positive"
    ]:
        subset = df[
            df["error_type"] == error_type
        ]

        print("\n" + "=" * 65)
        print(error_type.upper())
        print("=" * 65)

        for _, row in subset.iterrows():
            print("\nPOST ID:", row["post_id"])
            print(
                "PROBABILITY:",
                round(row["positive_probability"], 4)
            )
            print("TEXT:", row["text"])
            print("-" * 65)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSAVED:", OUTPUT_FILE)
    print("STAGE 8.3 COMPLETED")


if __name__ == "__main__":
    main()
