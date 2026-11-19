
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SEED = 42
BASE = Path("data/annotation")

TRAIN_FILE = BASE / "incident_labels_final_v2.csv"
NEW_FILE = BASE / "additional_annotation_sample.csv"
OUTPUT_FILE = BASE / "additional_auto_labels.csv"

print("=" * 60)
print("STAGE 7.2 - AUTOMATIC PSEUDO-ANNOTATION")
print("=" * 60)

train = pd.read_csv(
    TRAIN_FILE,
    dtype={"post_id": "string"}
)

new = pd.read_csv(
    NEW_FILE,
    dtype={"post_id": "string"}
)

# Only train on verified binary labels.
train["target"] = pd.to_numeric(
    train["final_incident_target"],
    errors="coerce"
)

train = train[
    train["target"].isin([0, 1])
    & train["text"].notna()
].copy()

assert train["post_id"].is_unique
assert new["post_id"].is_unique
assert not new["post_id"].isin(train["post_id"]).any()

print("\nTraining records:", len(train))
print("New records:", len(new))

print("\nTraining class distribution:")
print(train["target"].value_counts().to_string())

# TF-IDF extracts informative words and phrases.
# Logistic Regression predicts the probability
# that a tweet describes an actual incident.
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=10000,
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

model.fit(
    train["text"].fillna(""),
    train["target"].astype(int)
)

# Probability of actual incident.
probability = model.predict_proba(
    new["text"].fillna("")
)[:, 1]

new["auto_incident_probability"] = probability.round(4)

# Conservative thresholds.
# These are initial heuristic values, not calibrated
# confidence guarantees.
new["auto_incident_target"] = pd.NA

new.loc[
    new["auto_incident_probability"] >= 0.80,
    "auto_incident_target"
] = 1

new.loc[
    new["auto_incident_probability"] <= 0.20,
    "auto_incident_target"
] = 0

new["auto_label_status"] = "uncertain"

new.loc[
    new["auto_incident_target"].eq(1),
    "auto_label_status"
] = "pseudo_positive"

new.loc[
    new["auto_incident_target"].eq(0),
    "auto_label_status"
] = "pseudo_negative"

new["auto_label_source"] = "tfidf_logistic_regression"
new["human_verified"] = False

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"{OUTPUT_FILE} already exists. "
        "Existing output will not be overwritten."
    )

new.to_csv(OUTPUT_FILE, index=False)

print("\nAUTOMATIC LABEL DISTRIBUTION:")
print(
    new["auto_label_status"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nPROBABILITY STATISTICS:")
print(
    new["auto_incident_probability"]
    .describe()
    .round(4)
    .to_string()
)

print("\nSAVED:", OUTPUT_FILE)
print("\nSTAGE 7.2 COMPLETED")
