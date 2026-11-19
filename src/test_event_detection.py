
"""
Civil Unrest Event Extraction and Prediction from Social Media Data

STAGE 3: ZERO-SHOT EVENT DETECTION

Input:
    data/annotation/balanced_candidate_pool.csv

Output:
    data/annotation/event_detection_test.csv

Model:
    typeform/distilbert-base-uncased-mnli
"""

from pathlib import Path
import pandas as pd
import torch

from transformers import pipeline


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "annotation"
    / "balanced_candidate_pool.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "annotation"
    / "event_detection_test.csv"
)

MODEL_NAME = "typeform/distilbert-base-uncased-mnli"

SAMPLE_SIZE = 50
RANDOM_SEED = 42

EVENT_LABEL = (
    "a social media post reporting or describing "
    "a specific civil unrest event that happened"
)

NON_EVENT_LABEL = (
    "a social media post discussing a topic, expressing "
    "an opinion or support, or mentioning a movement "
    "without reporting a specific event"
)

CANDIDATE_LABELS = [
    EVENT_LABEL,
    NON_EVENT_LABEL
]


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_data():

    print("\nLoading candidate dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Candidate dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    required_columns = [
        "post_id",
        "date",
        "location",
        "text",
        "language",
        "user_hash",
        "sampling_group"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=["text", "sampling_group"]
    ).copy()

    df["text"] = df["text"].astype(str)

    df = df[
        df["text"].str.strip() != ""
    ].copy()

    print("Total candidate records:", len(df))

    return df


# --------------------------------------------------
# 3. SELECT TEST SAMPLES
# --------------------------------------------------

def select_samples(df):

    print("\nSelecting test samples...")

    keyword_df = df[
        df["sampling_group"] == "keyword_candidate"
    ]

    control_df = df[
        df["sampling_group"] == "random_control"
    ]

    if len(keyword_df) < SAMPLE_SIZE:
        raise ValueError(
            "Not enough keyword candidates for sampling."
        )

    if len(control_df) < SAMPLE_SIZE:
        raise ValueError(
            "Not enough random controls for sampling."
        )

    keyword_sample = keyword_df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED
    ).copy()

    control_sample = control_df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED
    ).copy()

    keyword_sample["sampling_group"] = "keyword_candidate"
    control_sample["sampling_group"] = "random_control"

    test_df = pd.concat(
        [keyword_sample, control_sample],
        ignore_index=True
    )

    test_df = test_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print("Keyword candidates:", len(keyword_sample))
    print("Random controls:", len(control_sample))
    print("Total test samples:", len(test_df))

    return test_df


# --------------------------------------------------
# 4. LOAD ZERO-SHOT CLASSIFICATION MODEL
# --------------------------------------------------

def load_model():

    print("\nLoading zero-shot classification model...")
    print("Model:", MODEL_NAME)

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=-1
    )

    print("Model loaded successfully.")

    return classifier


# --------------------------------------------------
# 5. CLASSIFY POSTS
# --------------------------------------------------

def classify_posts(classifier, df):

    print("\nStarting event detection...")
    print("This may take some time on CPU.")

    predictions = []

    texts = df["text"].tolist()

    results = classifier(
        texts,
        candidate_labels=CANDIDATE_LABELS,
        multi_label=False,
        batch_size=8,
        truncation=True
    )

    for index, prediction in enumerate(results):

        # The model returns labels sorted by score.
        # Therefore, map scores using their label names.
        score_by_label = dict(
            zip(
                prediction["labels"],
                prediction["scores"]
            )
        )

        event_score = score_by_label[EVENT_LABEL]
        non_event_score = score_by_label[NON_EVENT_LABEL]

        predicted_label = prediction["labels"][0]

        if predicted_label == EVENT_LABEL:
            predicted_class = "event_reporting"
        else:
            predicted_class = "topic_or_opinion"

        predictions.append({
            "event_score": float(event_score),
            "non_event_score": float(non_event_score),
            "predicted_label": predicted_class
        })

        if (index + 1) % 10 == 0:
            print(
                f"Processed {index + 1}/{len(texts)} posts"
            )

    prediction_df = pd.DataFrame(predictions)

    result_df = pd.concat(
        [df.reset_index(drop=True), prediction_df],
        axis=1
    )

    return result_df


# --------------------------------------------------
# 6. SAVE RESULTS AND DISPLAY SUMMARY
# --------------------------------------------------

def save_results(df):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("EVENT DETECTION SUMMARY")
    print("=" * 60)

    print("\nPredicted classes:")
    print(
        df["predicted_label"]
        .value_counts()
        .to_string()
    )

    print("\nPredictions by sampling group:")
    print(
        pd.crosstab(
            df["sampling_group"],
            df["predicted_label"]
        ).to_string()
    )

    print("\nAverage event score by sampling group:")
    print(
        df.groupby("sampling_group")["event_score"]
        .mean()
        .round(4)
        .to_string()
    )

    print("\nSaved results:")
    print(OUTPUT_FILE)


# --------------------------------------------------
# 7. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 60)
    print("CIVIL UNREST PROJECT - STAGE 3")
    print("ZERO-SHOT EVENT DETECTION")
    print("=" * 60)

    df = load_data()

    test_df = select_samples(df)

    classifier = load_model()

    result_df = classify_posts(
        classifier,
        test_df
    )

    save_results(result_df)

    print("\nSTAGE 3 COMPLETED")


if __name__ == "__main__":
    main()
