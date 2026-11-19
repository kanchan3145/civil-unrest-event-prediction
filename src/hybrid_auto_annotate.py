
from pathlib import Path
import re

import numpy as np
import pandas as pd

from extract_event_evidence_v3 import analyze_post, clean_text

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE = PROJECT_DIR / "data" / "annotation"

INPUT_FILE = BASE / "additional_auto_labels.csv"
OUTPUT_FILE = BASE / "additional_hybrid_labels.csv"

# Additional expressions not fully covered by V3.
INCIDENT_PATTERNS = {
    "injury": [
        r"\b\d+\s+(?:people|personnel|farmers|protesters|police|officers?)\s+(?:were\s+)?injured\b",
        r"\b(?:police|farmers|protesters|officers?)\s+(?:were\s+)?injured\b",
    ],
    "arrest": [
        r"\barrested\s+(?:at|in|near|outside|while|during)\b",
        r"\b(?:were|was|have been|has been)\s+(?:arrested|detained)\b",
    ],
    "clash": [
        r"\bclashes?\s+with\b",
        r"\bclashes?\s+between\b",
        r"\bclashes?\s+during\b",
    ],
    "police_action": [
        r"\bpolice\s+(?:fire|fired|used)\s+tear\s*gas\b",
        r"\bpolice\s+(?:lathi[- ]charged|detained|arrested)\b",
    ],
    "protest_activity": [
        r"\bthousands?\s+(?:of\s+people\s+)?(?:attended|joined)\s+.{0,40}\b(?:protest|rally|mahapanchayat|march)\b",
        r"\b(?:farmers|protesters)\s+climb(?:ed|s)?\s+on(?:to)?\b",
    ],
}

FUTURE_PATTERNS = [
    r"\b(?:will|shall)\s+(?:march|protest|hold|organize|organise|stage)\b",
    r"\b(?:planned|scheduled|upcoming)\s+(?:protest|rally|march|strike)\b",
    r"\b(?:plans?|planning)\s+to\s+(?:march|protest|hold|organize)\b",
]

NEGATION_PATTERNS = [
    r"\b(?:did not|never|no evidence of|false claim|fake news)\b",
]

UNCERTAINTY_PATTERNS = [
    r"\b(?:allegedly|unconfirmed|rumou?r|not sure)\b",
]

RELATED_PATTERNS = [
    r"\b(?:government|centre|center|pm)\s+.{0,40}\b(?:repealed|announced|decided)\b",
    r"\bfarm laws?\s+(?:were|have been)\s+repealed\b",
]

def find_matches(text, patterns):
    hits = []

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            hits.append(match.group(0))

    return list(dict.fromkeys(hits))


def classify_post(row):
    text = clean_text(row.get("text", ""))

    # Existing V3 evidence analysis.
    evidence = analyze_post(row)

    incident_hits = []
    for category, patterns in INCIDENT_PATTERNS.items():
        incident_hits.extend(find_matches(text, patterns))

    future_hits = find_matches(text, FUTURE_PATTERNS)
    negation_hits = find_matches(text, NEGATION_PATTERNS)
    uncertainty_hits = find_matches(text, UNCERTAINTY_PATTERNS)
    related_hits = find_matches(text, RELATED_PATTERNS)

    v3_decision = evidence["event_evidence_decision"]

    probability = float(
        row.get("auto_incident_probability", 0.5)
    )

    reasons = []

    if incident_hits:
        reasons.append("Additional incident evidence detected")

    if evidence["action_evidence"]:
        reasons.append("V3 event expression detected")

    if future_hits:
        reasons.append("Future event expression detected")

    if uncertainty_hits:
        reasons.append("Uncertainty expression detected")

    if negation_hits:
        reasons.append("Negation expression detected")

    # Conservative decision logic.
    if negation_hits or uncertainty_hits:
        category = "uncertain"

    elif incident_hits:
        category = "actual_incident"

    elif v3_decision == "event_candidate":
        category = "actual_incident"

    elif future_hits or v3_decision == "planned_or_future":
        category = "planned_event"

    elif related_hits or v3_decision == "related_action":
        category = "related_action"

    elif v3_decision == "non_event":
        category = "non_event"

    else:
        category = "uncertain"

    target_map = {
        "actual_incident": 1,
        "planned_event": 0,
        "related_action": 0,
        "non_event": 0,
        "uncertain": np.nan,
    }

    # This is a heuristic score, not a calibrated probability.
    if category == "actual_incident":
        evidence_strength = "strong" if incident_hits else "moderate"
    elif category == "uncertain":
        evidence_strength = "insufficient"
    else:
        evidence_strength = "rule_based"

    return pd.Series({
        "hybrid_category": category,
        "hybrid_incident_target": target_map[category],
        "hybrid_evidence_strength": evidence_strength,
        "hybrid_evidence": "; ".join(incident_hits),
        "hybrid_reason": "; ".join(reasons) or evidence["event_evidence_reason"],
        "hybrid_v3_decision": v3_decision,
        "hybrid_model_probability": probability,
        "hybrid_label_source": "automatic_hybrid_rules",
        "human_verified": False,
    })


def main():
    print("=" * 65)
    print("STAGE 7.3 - HYBRID AUTOMATIC ANNOTATION")
    print("=" * 65)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    assert df["post_id"].notna().all()
    assert df["post_id"].is_unique
    assert len(df) == 300

    results = df.apply(classify_post, axis=1)

    output = pd.concat(
        [df.reset_index(drop=True), results],
        axis=1
    )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. "
            "Existing output will not be overwritten."
        )

    output.to_csv(OUTPUT_FILE, index=False)

    print("\nHYBRID CATEGORY DISTRIBUTION:")
    print(output["hybrid_category"].value_counts().to_string())

    print("\nBINARY TARGET DISTRIBUTION:")
    print(
        output["hybrid_incident_target"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nSAVED:", OUTPUT_FILE)
    print("\nSTAGE 7.3 COMPLETED")


if __name__ == "__main__":
    main()
