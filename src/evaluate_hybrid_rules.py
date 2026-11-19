
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hybrid_auto_annotate import classify_post

BASE = PROJECT_DIR / "data" / "annotation"

INPUT_FILE = BASE / "incident_labels_final_v2.csv"
OUTPUT_FILE = BASE / "hybrid_rule_evaluation.csv"

print("=" * 65)
print("STAGE 7.4 - HYBRID RULE EVALUATION")
print("=" * 65)

df = pd.read_csv(
    INPUT_FILE,
    dtype={"post_id": "string"}
)

assert df["post_id"].is_unique

# The hybrid function accepts the model probability
# as a diagnostic field. It does not affect the rule decision.
df["auto_incident_probability"] = 0.5

predictions = df.apply(classify_post, axis=1)

result = pd.concat(
    [df.reset_index(drop=True), predictions],
    axis=1
)

print("\nFIVE-CATEGORY CLASSIFICATION REPORT:")

categories = [
    "actual_incident",
    "planned_event",
    "related_action",
    "non_event",
    "uncertain"
]

print(
    classification_report(
        result["final_event_category"],
        result["hybrid_category"],
        labels=categories,
        zero_division=0
    )
)

print("\nFIVE-CATEGORY CONFUSION MATRIX:")

matrix = confusion_matrix(
    result["final_event_category"],
    result["hybrid_category"],
    labels=categories
)

print(
    pd.DataFrame(
        matrix,
        index=categories,
        columns=categories
    ).to_string()
)

# Binary evaluation only where both sides have
# definite incident/non-incident labels.
true_target = pd.to_numeric(
    result["final_incident_target"],
    errors="coerce"
)

pred_target = pd.to_numeric(
    result["hybrid_incident_target"],
    errors="coerce"
)

mask = true_target.notna() & pred_target.notna()

print("\nBINARY CLASSIFICATION REPORT:")
print("Evaluated records:", mask.sum())
print("Excluded due to uncertainty:", (~mask).sum())

if mask.sum() > 0:
    print(
        classification_report(
            true_target[mask].astype(int),
            pred_target[mask].astype(int),
            labels=[0, 1],
            zero_division=0
        )
    )

result["category_match"] = (
    result["final_event_category"]
    == result["hybrid_category"]
)

print("\nCATEGORY AGREEMENT:")
print(
    result["category_match"]
    .value_counts()
    .to_string()
)

print("\nFIRST 15 CATEGORY DISAGREEMENTS:")

columns = [
    "post_id",
    "text",
    "final_event_category",
    "hybrid_category",
    "hybrid_reason"
]

disagreements = result[
    ~result["category_match"]
].copy()

disagreements["text"] = (
    disagreements["text"]
    .fillna("")
    .str.slice(0, 180)
)

print(
    disagreements[columns]
    .head(15)
    .to_string(index=False)
)

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"{OUTPUT_FILE} already exists. "
        "Existing evaluation was not overwritten."
    )

result.to_csv(OUTPUT_FILE, index=False)

print("\nSAVED:", OUTPUT_FILE)
print("\nSTAGE 7.4 COMPLETED")
