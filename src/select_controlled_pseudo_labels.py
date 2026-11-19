
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sentence_transformers import SentenceTransformer


# ============================================================
# STAGE 8.7 - CONTROLLED PSEUDO-LABEL SELECTION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent

VERIFIED_FILE = (
    ROOT / "data" / "annotation"
    / "incident_labels_final_v2.csv"
)

CANDIDATE_FILE = (
    ROOT / "data" / "annotation"
    / "additional_hybrid_labels_v2.csv"
)

OUTPUT_DIR = ROOT / "data" / "annotation"

ALL_PREDICTIONS_FILE = (
    OUTPUT_DIR / "controlled_pseudo_label_predictions_v1.csv"
)

SELECTED_FILE = (
    OUTPUT_DIR / "controlled_pseudo_labels_v1.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "controlled_pseudo_label_summary_v1.json"
)

# Conservative agreement thresholds.
# These are experimental and are not calibrated.
POSITIVE_THRESHOLD = 0.70
NEGATIVE_THRESHOLD = 0.30

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    print("\nLoading datasets...")

    verified = pd.read_csv(
        VERIFIED_FILE,
        dtype={"post_id": "string"}
    )

    candidates = pd.read_csv(
        CANDIDATE_FILE,
        dtype={"post_id": "string"}
    )

    verified_required = {
        "post_id",
        "text",
        "final_incident_target"
    }

    candidate_required = {
        "post_id",
        "text",
        "hybrid_v2_category",
        "hybrid_v2_target",
        "hybrid_v2_human_verified"
    }

    for name, df, required in [
        ("Verified", verified, verified_required),
        ("Candidates", candidates, candidate_required)
    ]:

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {missing}"
            )

        if df["post_id"].isna().any():
            raise ValueError(
                f"{name}: missing post IDs"
            )

        if df["post_id"].duplicated().any():
            raise ValueError(
                f"{name}: duplicate post IDs"
            )

    overlap = set(verified["post_id"]) & set(
        candidates["post_id"]
    )

    if overlap:
        raise ValueError(
            f"Found {len(overlap)} candidate IDs "
            "already in the verified dataset."
        )

    verified["final_incident_target"] = pd.to_numeric(
        verified["final_incident_target"],
        errors="coerce"
    )

    train = verified[
        verified["final_incident_target"].isin([0, 1])
    ].copy()

    train["target"] = (
        train["final_incident_target"].astype(int)
    )

    if train["target"].nunique() != 2:
        raise ValueError(
            "Training data must contain both classes."
        )

    print("Verified total:", len(verified))
    print("Definite training examples:", len(train))
    print("Candidate tweets:", len(candidates))

    print("\nTraining class distribution:")
    print(train["target"].value_counts().to_string())

    return train, candidates


# ============================================================
# 2. TRAIN TF-IDF BASELINE
# ============================================================

def train_tfidf(train, candidates):

    print("\nTraining TF-IDF baseline...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
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
        train["text"].fillna("").astype(str),
        train["target"]
    )

    probabilities = model.predict_proba(
        candidates["text"].fillna("").astype(str)
    )[:, 1]

    print("TF-IDF predictions completed.")

    return probabilities


# ============================================================
# 3. TRAIN SEMANTIC BASELINE
# ============================================================

def train_semantic(train, candidates):

    print("\nLoading MiniLM embedding model...")

    encoder = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )

    print("Encoding verified training tweets...")

    train_embeddings = encoder.encode(
        train["text"].fillna("").astype(str).tolist(),
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Encoding candidate tweets...")

    candidate_embeddings = encoder.encode(
        candidates["text"].fillna("").astype(str).tolist(),
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
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
        train_embeddings,
        train["target"]
    )

    probabilities = model.predict_proba(
        candidate_embeddings
    )[:, 1]

    print("Semantic predictions completed.")

    return probabilities


# ============================================================
# 4. SELECT CONTROLLED PSEUDO-LABELS
# ============================================================

def select_labels(candidates, tfidf_probs, semantic_probs):

    result = candidates.copy()

    result["tfidf_probability"] = tfidf_probs
    result["semantic_probability"] = semantic_probs

    result["hybrid_v2_target"] = pd.to_numeric(
        result["hybrid_v2_target"],
        errors="coerce"
    )

    result["controlled_pseudo_target"] = pd.NA

    result["controlled_pseudo_status"] = "abstain"

    result["controlled_pseudo_reason"] = (
        "insufficient_model_agreement"
    )

    # Never pseudo-label a record already marked human verified.
    human_verified = (
        result["hybrid_v2_human_verified"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    result.loc[
        human_verified,
        "controlled_pseudo_reason"
    ] = "already_human_verified"

    category = (
        result["hybrid_v2_category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Both models must agree with the hybrid label.
    positive_mask = (
        ~human_verified
        & category.eq("actual_incident")
        & result["hybrid_v2_target"].eq(1)
        & result["tfidf_probability"].ge(
            POSITIVE_THRESHOLD
        )
        & result["semantic_probability"].ge(
            POSITIVE_THRESHOLD
        )
    )

    negative_mask = (
        ~human_verified
        & category.eq("non_event")
        & result["hybrid_v2_target"].eq(0)
        & result["tfidf_probability"].le(
            NEGATIVE_THRESHOLD
        )
        & result["semantic_probability"].le(
            NEGATIVE_THRESHOLD
        )
    )

    result.loc[
        positive_mask,
        "controlled_pseudo_target"
    ] = 1

    result.loc[
        positive_mask,
        "controlled_pseudo_status"
    ] = "selected_positive"

    result.loc[
        positive_mask,
        "controlled_pseudo_reason"
    ] = "tfidf_semantic_hybrid_agreement"

    result.loc[
        negative_mask,
        "controlled_pseudo_target"
    ] = 0

    result.loc[
        negative_mask,
        "controlled_pseudo_status"
    ] = "selected_negative"

    result.loc[
        negative_mask,
        "controlled_pseudo_reason"
    ] = "tfidf_semantic_hybrid_agreement"

    result["controlled_pseudo_source"] = (
        "tfidf_minilm_hybrid_v2"
    )

    result["controlled_pseudo_human_verified"] = False

    return result


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 8.7 - CONTROLLED PSEUDO-LABEL SELECTION")
    print("=" * 65)

    # Refuse to overwrite existing experiment outputs.
    for path in [
        ALL_PREDICTIONS_FILE,
        SELECTED_FILE,
        SUMMARY_FILE
    ]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    train, candidates = load_data()

    tfidf_probs = train_tfidf(
        train,
        candidates
    )

    semantic_probs = train_semantic(
        train,
        candidates
    )

    result = select_labels(
        candidates,
        tfidf_probs,
        semantic_probs
    )

    selected = result[
        result["controlled_pseudo_status"].isin([
            "selected_positive",
            "selected_negative"
        ])
    ].copy()

    print("\nSELECTION DISTRIBUTION:")
    print(
        result["controlled_pseudo_status"]
        .value_counts()
        .to_string()
    )

    print("\nSelected examples:", len(selected))

    summary = {
        "stage": "8.7",
        "policy": "broad_civil_unrest_event_detection",
        "verified_training_examples": len(train),
        "candidate_examples": len(candidates),
        "selected_examples": len(selected),
        "selected_positive": int(
            (selected["controlled_pseudo_target"] == 1).sum()
        ),
        "selected_negative": int(
            (selected["controlled_pseudo_target"] == 0).sum()
        ),
        "abstained": int(
            (result["controlled_pseudo_status"] == "abstain").sum()
        ),
        "positive_threshold": POSITIVE_THRESHOLD,
        "negative_threshold": NEGATIVE_THRESHOLD,
        "human_verified_labels_modified": False,
        "pseudo_labels_are_human_verified": False,
        "selection_method": "tfidf_minilm_hybrid_v2_agreement"
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        ALL_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    selected.to_csv(
        SELECTED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    print("\nOUTPUT FILES:")
    print(ALL_PREDICTIONS_FILE)
    print(SELECTED_FILE)
    print(SUMMARY_FILE)

    print("\nOriginal verified labels were NOT modified.")
    print("STAGE 8.7 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
