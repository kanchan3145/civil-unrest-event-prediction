
from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

EVAL_DIR = ROOT / "data" / "evaluation"

TFIDF_FILE = (
    EVAL_DIR / "verified_baseline_predictions.csv"
)

SEMANTIC_FILE = (
    EVAL_DIR / "semantic_baseline_predictions.csv"
)

OUTPUT_FILE = (
    EVAL_DIR / "baseline_model_comparison.csv"
)


# ============================================================
# 2. STANDARDIZE DATASET COLUMNS
# ============================================================

def standardize_columns(df, model_name):

    df = df.copy()

    # Different scripts may save the verified target
    # under different column names.
    if "target" not in df.columns:

        if "final_incident_target" in df.columns:

            df = df.rename(
                columns={
                    "final_incident_target": "target"
                }
            )

        else:
            raise ValueError(
                f"{model_name}: No target column found. "
                f"Available columns: {df.columns.tolist()}"
            )

    required_columns = {
        "post_id",
        "text",
        "target",
        "predicted_target",
        "positive_probability"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name} missing columns: {missing}"
        )

    # Ensure post IDs are unique.
    if df["post_id"].isna().any():
        raise ValueError(
            f"{model_name} contains missing post IDs"
        )

    if df["post_id"].duplicated().any():
        raise ValueError(
            f"{model_name} contains duplicate post IDs"
        )

    # Validate numerical columns.
    numeric_columns = [
        "target",
        "predicted_target",
        "positive_probability"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if df[column].isna().any():
            raise ValueError(
                f"{model_name}: Invalid values in {column}"
            )

    for column in ["target", "predicted_target"]:

        if not df[column].isin([0, 1]).all():
            raise ValueError(
                f"{model_name}: {column} must contain 0 or 1"
            )

        df[column] = df[column].astype(int)

    if not df["positive_probability"].between(
        0, 1
    ).all():
        raise ValueError(
            f"{model_name}: Probabilities must be between 0 and 1"
        )

    return df


# ============================================================
# 3. DISPLAY EXAMPLES
# ============================================================

def display_examples(df, category):

    subset = df[
        df["comparison"] == category
    ]

    print("\n" + "=" * 65)
    print(category.upper())
    print("=" * 65)

    print("Total examples:", len(subset))

    if subset.empty:
        print("No examples found.")
        return

    for _, row in subset.iterrows():

        print("\nPOST ID:", row["post_id"])

        print(
            "Actual label:",
            row["target"]
        )

        print(
            "TF-IDF prediction:",
            row["tfidf_prediction"]
        )

        print(
            "TF-IDF probability:",
            round(row["tfidf_probability"], 4)
        )

        print(
            "Semantic prediction:",
            row["semantic_prediction"]
        )

        print(
            "Semantic probability:",
            round(row["semantic_probability"], 4)
        )

        print("TEXT:", row["text"])

        print("-" * 65)


# ============================================================
# 4. MAIN FUNCTION
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 8.5 - BASELINE MODEL COMPARISON")
    print("=" * 65)

    # --------------------------------------------------------
    # STEP 1: CHECK INPUT FILES
    # --------------------------------------------------------

    if not TFIDF_FILE.exists():
        raise FileNotFoundError(
            f"TF-IDF file not found: {TFIDF_FILE}"
        )

    if not SEMANTIC_FILE.exists():
        raise FileNotFoundError(
            f"Semantic file not found: {SEMANTIC_FILE}"
        )

    # --------------------------------------------------------
    # STEP 2: LOAD PREDICTIONS
    # --------------------------------------------------------

    print("\nLoading prediction files...")

    tfidf = pd.read_csv(
        TFIDF_FILE,
        dtype={"post_id": "string"}
    )

    semantic = pd.read_csv(
        SEMANTIC_FILE,
        dtype={"post_id": "string"}
    )

    print("TF-IDF rows:", len(tfidf))
    print("Semantic rows:", len(semantic))

    # --------------------------------------------------------
    # STEP 3: STANDARDIZE COLUMNS
    # --------------------------------------------------------

    tfidf = standardize_columns(
        tfidf,
        "TF-IDF"
    )

    semantic = standardize_columns(
        semantic,
        "Semantic"
    )

    print("\nColumn validation successful.")

    # --------------------------------------------------------
    # STEP 4: PREPARE TF-IDF DATA
    # --------------------------------------------------------

    tfidf_data = tfidf[
        [
            "post_id",
            "text",
            "target",
            "predicted_target",
            "positive_probability"
        ]
    ].rename(
        columns={
            "predicted_target": "tfidf_prediction",
            "positive_probability": "tfidf_probability"
        }
    )

    # --------------------------------------------------------
    # STEP 5: PREPARE SEMANTIC DATA
    # --------------------------------------------------------

    semantic_data = semantic[
        [
            "post_id",
            "text",
            "target",
            "predicted_target",
            "positive_probability"
        ]
    ].rename(
        columns={
            "text": "semantic_text",
            "target": "semantic_target",
            "predicted_target": "semantic_prediction",
            "positive_probability": "semantic_probability"
        }
    )

    # --------------------------------------------------------
    # STEP 6: MERGE USING POST ID
    # --------------------------------------------------------

    merged = tfidf_data.merge(
        semantic_data,
        on="post_id",
        how="inner",
        validate="one_to_one"
    )

    print("\nMatched posts:", len(merged))

    if (
        len(merged) != len(tfidf)
        or len(merged) != len(semantic)
    ):
        raise ValueError(
            "Both models were not evaluated on identical post sets."
        )

    # Verify human labels match.
    if not (
        merged["target"] == merged["semantic_target"]
    ).all():
        raise ValueError(
            "Human-verified target labels do not match."
        )

    # Verify text consistency.
    if not (
        merged["text"].fillna("").astype(str).str.strip()
        ==
        merged["semantic_text"].fillna("").astype(str).str.strip()
    ).all():

        print(
            "\nWARNING: Some tweet texts differ between files."
        )

        print(
            "Post IDs and verified labels match. "
            "Continuing comparison."
        )

    merged = merged.drop(
        columns=[
            "semantic_target",
            "semantic_text"
        ]
    )

    # --------------------------------------------------------
    # STEP 7: CLASSIFY MODEL AGREEMENT
    # --------------------------------------------------------

    merged["comparison"] = "unclassified"

    actual = merged["target"]

    t = merged["tfidf_prediction"]

    s = merged["semantic_prediction"]

    # Both models correctly detected an incident.
    merged.loc[
        (actual == 1) & (t == 1) & (s == 1),
        "comparison"
    ] = "both_detected_incident"

    # Only semantic model detected an incident.
    merged.loc[
        (actual == 1) & (t == 0) & (s == 1),
        "comparison"
    ] = "semantic_only_detected_incident"

    # Only TF-IDF detected an incident.
    merged.loc[
        (actual == 1) & (t == 1) & (s == 0),
        "comparison"
    ] = "tfidf_only_detected_incident"

    # Both models missed an actual incident.
    merged.loc[
        (actual == 1) & (t == 0) & (s == 0),
        "comparison"
    ] = "both_missed_incident"

    # Both models incorrectly predicted an incident.
    merged.loc[
        (actual == 0) & (t == 1) & (s == 1),
        "comparison"
    ] = "both_false_positive"

    # Only semantic model produced a false positive.
    merged.loc[
        (actual == 0) & (t == 0) & (s == 1),
        "comparison"
    ] = "semantic_only_false_positive"

    # Only TF-IDF produced a false positive.
    merged.loc[
        (actual == 0) & (t == 1) & (s == 0),
        "comparison"
    ] = "tfidf_only_false_positive"

    # Both models correctly predicted non-event.
    merged.loc[
        (actual == 0) & (t == 0) & (s == 0),
        "comparison"
    ] = "both_correct_non_event"

    if (merged["comparison"] == "unclassified").any():
        raise ValueError(
            "Some rows could not be classified."
        )

    # --------------------------------------------------------
    # STEP 8: PRINT COMPARISON SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 65)

    print(
        merged["comparison"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # STEP 9: DISPLAY IMPORTANT CASES
    # --------------------------------------------------------

    important_categories = [
        "both_missed_incident",
        "semantic_only_detected_incident",
        "tfidf_only_detected_incident",
        "both_false_positive",
        "semantic_only_false_positive",
        "tfidf_only_false_positive"
    ]

    for category in important_categories:

        display_examples(
            merged,
            category
        )

    # --------------------------------------------------------
    # STEP 10: SAVE OUTPUT
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 65)
    print("OUTPUT SAVED")
    print("=" * 65)

    print(OUTPUT_FILE)

    print("\nSTAGE 8.5 COMPLETED SUCCESSFULLY")


# ============================================================
# 5. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
