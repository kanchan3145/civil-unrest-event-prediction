
from pathlib import Path
import argparse
import json
import random

import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)


# ============================================================
# STAGE 11.4 - HUMAN-REVIEWED EVENT EXTRACTION EVALUATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DIR = ROOT / "data" / "extracted"

SOURCE = DIR / "structured_event_candidates_v3.csv"
MENTIONS = DIR / "event_mentions_v3.csv"
AUDIT = DIR / "event_candidate_audit_v3.csv"

SAMPLE = DIR / "human_extraction_audit_v1.csv"

REPORT = DIR / "human_extraction_evaluation_v1.json"

KNOWN = DIR / "human_extraction_evaluation_rows_v1.csv"

SEED = 42

VALID = {
    "event",
    "not_event",
    "uncertain"
}


# ============================================================
# 1. PREPARE HUMAN ANNOTATION DATASET
# ============================================================

def prepare():

    if SAMPLE.exists():
        raise FileExistsError(
            f"Preserving existing audit: {SAMPLE}"
        )

    tweets = pd.read_csv(
        SOURCE,
        dtype={"post_id": "string"},
        keep_default_na=False
    )

    mentions = pd.read_csv(
        MENTIONS,
        dtype={"post_id": "string"},
        keep_default_na=False
    )

    audit = pd.read_csv(
        AUDIT,
        dtype={"post_id": "string"},
        keep_default_na=False
    )

    if (
        tweets.post_id.isna().any()
        or tweets.post_id.duplicated().any()
    ):
        raise ValueError(
            "Invalid or duplicate tweet post_id"
        )

    if len(tweets) != 300:
        print(
            f"Note: source contains {len(tweets)} "
            "tweets (not 300)."
        )

    required = {
        "post_id",
        "text",
        "extraction_status",
        "classifier_predicted_target",
        "event_types"
    }

    if not required.issubset(tweets.columns):
        raise ValueError(
            "Unexpected V3 tweet-level schema"
        )

    known_ids = set(audit.post_id)

    if len(known_ids) != 19:
        print(
            "Warning: known audit contains "
            f"{len(known_ids)} records, expected 19"
        )

    rng = random.Random(SEED)

    selected = set(known_ids)

    # Select examples from different extraction groups.
    # Include difficult cases and classifier negatives.

    targets = {
        "event_candidate": 15,
        "needs_review": 15,
        "classifier_negative_review": 15,
        "insufficient_evidence": 12,
        "not_selected_by_classifier": 12,
        "exclude_unrelated_conflict": 2
    }

    for group, n in targets.items():

        pool = sorted(
            tweets.loc[
                (
                    tweets.extraction_status == group
                )
                &
                (
                    ~tweets.post_id.isin(selected)
                ),
                "post_id"
            ].tolist()
        )

        selected.update(
            rng.sample(
                pool,
                min(n, len(pool))
            )
        )

    output = tweets[
        tweets.post_id.isin(selected)
    ].copy()

    order = {
        value: i
        for i, value in enumerate(
            sorted(selected)
        )
    }

    output["_sort"] = output.post_id.map(order)

    output = output.sort_values(
        "_sort"
    ).drop(
        columns="_sort"
    )

    # Collect evidence extracted by Stage 11.3.

    evidence = (
        mentions.groupby(
            "post_id",
            sort=False
        )["event_evidence"]
        .apply(
            lambda x: " || ".join(
                dict.fromkeys(
                    str(v)
                    for v in x
                    if str(v).strip()
                )
            )[:1500]
        )
    )

    types = (
        mentions.groupby(
            "post_id",
            sort=False
        )["event_type"]
        .apply(
            lambda x: ", ".join(
                dict.fromkeys(
                    str(v)
                    for v in x
                    if str(v).strip()
                )
            )
        )
    )

    output["extracted_evidence"] = (
        output.post_id.map(
            evidence
        ).fillna("")
    )

    output["extracted_event_types"] = (
        output.post_id.map(
            types
        ).fillna("")
    )

    output["in_previous_19_audit"] = (
        output.post_id.isin(
            known_ids
        )
    )

    # Human annotation fields.
    # Leave them empty until manually reviewed.

    output["manual_decision"] = ""

    output["manual_event_type"] = ""

    output["manual_event_location"] = ""

    output["manual_event_date"] = ""

    output["manual_notes"] = ""

    output["reviewer_id"] = ""

    output["reviewed_at"] = ""

    output.to_csv(
        SAMPLE,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 65)
    print("STAGE 11.4 - AUDIT PREPARATION")
    print("=" * 65)

    print("\nAudit rows:", len(output))

    print(
        "Known cases included:",
        int(
            output.in_previous_19_audit.sum()
        )
    )

    print("\nBY EXTRACTION STATUS:")

    print(
        output.extraction_status
        .value_counts()
        .to_string()
    )

    print("\nCreated:", SAMPLE)

    print(
        "\nFill manual_decision with:"
        "\nevent / not_event / uncertain"
    )

    print(
        "\nDo not modify the model prediction columns."
    )


# ============================================================
# 2. EVALUATE HUMAN ANNOTATIONS
# ============================================================

def evaluate():

    if not SAMPLE.exists():
        raise FileNotFoundError(
            "Run --prepare first"
        )

    if REPORT.exists() or KNOWN.exists():
        raise FileExistsError(
            "Evaluation output already exists; "
            "preserving prior evaluation"
        )

    df = pd.read_csv(
        SAMPLE,
        dtype={"post_id": "string"},
        keep_default_na=False
    )

    if df.post_id.duplicated().any():
        raise ValueError(
            "Duplicate post_id in audit"
        )

    decision = (
        df.manual_decision
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid = sorted(
        set(decision) - VALID - {""}
    )

    if invalid:
        raise ValueError(
            "Invalid manual_decision values: "
            f"{invalid}. "
            "Use event, not_event, uncertain or blank."
        )

    completed = decision.isin(
        VALID
    )

    definite = decision.isin({
        "event",
        "not_event"
    })

    if not definite.any():

        print(
            "No definite human-reviewed rows yet."
        )

        print(
            "Fill manual_decision first."
        )

        return

    reviewed = df[
        definite
    ].copy()

    reviewed["manual_binary"] = (
        decision[definite] == "event"
    ).astype(int)

    # Strict extraction prediction:
    # Only event_candidate is positive.

    reviewed["extraction_binary"] = (
        reviewed.extraction_status
        .eq("event_candidate")
        .astype(int)
    )

    reviewed["classifier_binary"] = (
        pd.to_numeric(
            reviewed.classifier_predicted_target,
            errors="raise"
        ).astype(int)
    )

    if not reviewed.classifier_binary.isin(
        [0, 1]
    ).all():
        raise ValueError(
            "Invalid classifier targets"
        )

    truth = reviewed.manual_binary

    metrics = {}

    for label, column in [
        ("extraction", "extraction_binary"),
        ("classifier", "classifier_binary")
    ]:

        pred = reviewed[column]

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                truth,
                pred,
                average="binary",
                zero_division=0
            )
        )

        cm = confusion_matrix(
            truth,
            pred,
            labels=[0, 1]
        ).tolist()

        metrics[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(
                accuracy_score(
                    truth,
                    pred
                )
            ),
            "confusion_matrix_labels_0_1": cm
        }

    report = {
        "scope": (
            "Purposively stratified manual audit; "
            "not an unbiased population estimate"
        ),
        "total_audit_rows": int(
            len(df)
        ),
        "completed_rows": int(
            completed.sum()
        ),
        "definite_rows_used": int(
            definite.sum()
        ),
        "uncertain_rows_excluded": int(
            (decision == "uncertain").sum()
        ),
        "blank_rows_excluded": int(
            (decision == "").sum()
        ),
        "human_positive": int(
            truth.sum()
        ),
        "human_negative": int(
            (truth == 0).sum()
        ),
        "metrics": metrics,
        "warning": (
            "Human decisions must be independently "
            "annotated. Do not treat prior assistant "
            "assessments as ground truth. "
            "No independent external event "
            "verification performed."
        )
    }

    reviewed.to_csv(
        KNOWN,
        index=False,
        encoding="utf-8-sig"
    )

    REPORT.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8"
    )

    print("=" * 65)
    print("STAGE 11.4 - HUMAN REVIEW EVALUATION")
    print("=" * 65)

    print(
        f"\nCompleted: "
        f"{report['completed_rows']}/"
        f"{report['total_audit_rows']}"
    )

    print(
        "Definite labels used:",
        report["definite_rows_used"]
    )

    for name, metric in metrics.items():

        print(
            f"\n{name.upper()} METRICS:"
        )

        print(
            f"Precision: {metric['precision']:.3f}"
        )

        print(
            f"Recall: {metric['recall']:.3f}"
        )

        print(
            f"F1-score: {metric['f1']:.3f}"
        )

        print(
            f"Accuracy: {metric['accuracy']:.3f}"
        )

        print(
            "Confusion matrix [[TN, FP], [FN, TP]]:"
        )

        print(
            metric["confusion_matrix_labels_0_1"]
        )

    print("\nSaved:", REPORT)

    print("Saved:", KNOWN)


# ============================================================
# 3. MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Stage 11.4 human audit"
    )

    action = parser.add_mutually_exclusive_group(
        required=True
    )

    action.add_argument(
        "--prepare",
        action="store_true"
    )

    action.add_argument(
        "--evaluate",
        action="store_true"
    )

    args = parser.parse_args()

    if args.prepare:
        prepare()
    else:
        evaluate()
