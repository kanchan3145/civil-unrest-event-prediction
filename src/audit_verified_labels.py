
from pathlib import Path
import json
import re
import html

import pandas as pd


# ============================================================
# STAGE 8.6 - VERIFIED LABEL POLICY AUDIT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "annotation"
    / "incident_labels_final_v2.csv"
)

OUTPUT_DIR = ROOT / "data" / "annotation"

AUDIT_FILE = (
    OUTPUT_DIR / "verified_label_audit_v1.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "verified_label_audit_summary_v1.json"
)

# This file is read-only in this stage.
# No changes will be written to INPUT_FILE.


# ============================================================
# 1. BROAD CIVIL-UNREST EVENT INDICATORS
# ============================================================

# Patterns are deliberately diagnostic, not definitive.
# They help identify tweets that deserve closer inspection.

STRONG_EVENT_PATTERNS = {
    "clash_or_violence": [
        r"\bclash(?:ed|es|ing)?\b",
        r"\battacked?\b",
        r"\bbeat(?:en|ing)?\b",
        r"\bpelting stones\b",
        r"\bstone[- ]pelting\b",
        r"\binjur(?:ed|ies|y)\b",
        r"\bviolent clashes\b",
    ],

    "police_action": [
        r"\btear[- ]gas\b",
        r"\blathi[- ]charg(?:e|ed|ing)\b",
        r"\bdetain(?:ed|ing)?\b",
        r"\barrest(?:ed|s|ing)?\b",
        r"\bpolice fired\b",
        r"\bpolice used force\b",
    ],

    "march_or_rally": [
        r"\bmarch(?:ed|ing|es)?\b",
        r"\btractor (?:march|rally|parade)\b",
        r"\brall(?:y|ies)\b",
        r"\bprocession\b",
    ],

    "blockade_or_disruption": [
        r"\bblock(?:ed|ing|ade|ades)\b",
        r"\broads? (?:closed|blocked)\b",
        r"\brail roko\b",
        r"\bgherao\b",
    ],

    "ongoing_protest": [
        r"\bprotest(?:ed|ing|ers|ors|s)?\b",
        r"\bdemonstrat(?:ion|ions|ing|ors)\b",
        r"\bsit[- ]in\b",
        r"\bagitation\b",
    ],
}


# ============================================================
# 2. TEMPORAL AND CONTEXTUAL WARNING INDICATORS
# ============================================================

FUTURE_PATTERNS = [
    r"\bwill (?:march|protest|hold|organize|block|join)\b",
    r"\bto (?:observe|hold|organize|stage) "
    r"(?:a |an )?(?:protest|march|rally|strike)\b",
    r"\bplanned\b",
    r"\bplanning\b",
    r"\bproposed\b",
    r"\bpostponed\b",
    r"\bscheduled\b",
    r"\blikely to\b",
    r"\bexpected to\b",
    r"\bmay (?:march|protest|clash|detain)\b",
    r"\bprepared to\b",
    r"\bwould (?:march|protest|block)\b",
]

UNRELATED_CONTEXT_PATTERNS = [
    r"\bfact[- ]check\b",
    r"\bpassed off as\b",
    r"\bmisleading video\b",
    r"\bfalse claim\b",
    r"\bpanchayat elections\b",
]

UNCERTAINTY_PATTERNS = [
    r"\ballegedly\b",
    r"\brumou?r\b",
    r"\bunconfirmed\b",
    r"\buncertain\b",
    r"\bnot verified\b",
]

NEGATION_PATTERNS = [
    r"\bdid not (?:happen|occur|take place)\b",
    r"\bno (?:clash|violence|protest) occurred\b",
    r"\bnever happened\b",
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_text(value):
    """Normalize text for diagnostic pattern matching."""

    if pd.isna(value):
        return ""

    text = html.unescape(str(value))
    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_patterns(text, patterns):
    """Return matched pattern strings."""

    matches = []

    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(pattern)

    return matches


def extract_evidence(text):
    """Extract diagnostic evidence categories."""

    evidence = {}

    for category, patterns in STRONG_EVENT_PATTERNS.items():

        matches = find_patterns(text, patterns)

        if matches:
            evidence[category] = matches

    return evidence


def audit_row(row):
    """
    Analyze one verified tweet.

    This function never changes the original human label.
    """

    text = normalize_text(row["text"])

    category = str(
        row["final_event_category"]
    ).strip().lower()

    target = row["final_incident_target"]

    evidence = extract_evidence(text)

    future_matches = find_patterns(
        text,
        FUTURE_PATTERNS
    )

    unrelated_matches = find_patterns(
        text,
        UNRELATED_CONTEXT_PATTERNS
    )

    uncertainty_matches = find_patterns(
        text,
        UNCERTAINTY_PATTERNS
    )

    negation_matches = find_patterns(
        text,
        NEGATION_PATTERNS
    )

    flags = []

    # --------------------------------------------------------
    # A. LABEL CONSISTENCY
    # --------------------------------------------------------

    if category == "actual_incident":

        if pd.isna(target) or target != 1:
            flags.append("incident_target_mismatch")

    elif category == "uncertain":

        if pd.notna(target):
            flags.append("uncertain_target_mismatch")

    elif category in [
        "non_event",
        "planned_event",
        "related_action"
    ]:

        if pd.isna(target) or target != 0:
            flags.append("nonincident_target_mismatch")

    else:
        flags.append("unknown_human_category")

    # --------------------------------------------------------
    # B. POSSIBLE MISSING EVENT EVIDENCE
    # --------------------------------------------------------

    if target == 1:

        if not evidence:
            flags.append("positive_without_keyword_evidence")

        if future_matches:
            flags.append("positive_contains_future_language")

        if unrelated_matches:
            flags.append("positive_contains_unrelated_context")

        if negation_matches:
            flags.append("positive_contains_negation")

    # --------------------------------------------------------
    # C. POSSIBLE MISSED BROAD EVENTS
    # --------------------------------------------------------

    if target == 0:

        strong_categories = {
            "clash_or_violence",
            "police_action",
            "march_or_rally",
            "blockade_or_disruption"
        }

        detected_categories = set(evidence.keys())

        if detected_categories & strong_categories:
            flags.append("negative_contains_event_keywords")

        if "ongoing_protest" in detected_categories:
            flags.append("negative_contains_protest_language")

    # --------------------------------------------------------
    # D. TEMPORAL / CONTEXTUAL WARNINGS
    # --------------------------------------------------------

    if future_matches:
        flags.append("future_language_present")

    if unrelated_matches:
        flags.append("unrelated_context_present")

    if uncertainty_matches:
        flags.append("uncertainty_language_present")

    if negation_matches:
        flags.append("negation_present")

    # --------------------------------------------------------
    # E. REVIEW PRIORITY
    # --------------------------------------------------------

    high_priority_flags = {
        "incident_target_mismatch",
        "uncertain_target_mismatch",
        "nonincident_target_mismatch",
        "unknown_human_category",
    }

    if set(flags) & high_priority_flags:
        priority = "high"

    elif target == 1 and (
        "positive_without_keyword_evidence" in flags
        or "positive_contains_future_language" in flags
        or "positive_contains_unrelated_context" in flags
    ):
        priority = "high"

    elif target == 0 and (
        "negative_contains_event_keywords" in flags
    ):
        priority = "medium"

    elif pd.isna(target):
        priority = "medium"

    elif flags:
        priority = "low"

    else:
        priority = "none"

    return pd.Series({
        "audit_evidence_categories": "|".join(
            evidence.keys()
        ),
        "audit_future_matches": "|".join(
            future_matches
        ),
        "audit_unrelated_matches": "|".join(
            unrelated_matches
        ),
        "audit_uncertainty_matches": "|".join(
            uncertainty_matches
        ),
        "audit_negation_matches": "|".join(
            negation_matches
        ),
        "audit_flags": "|".join(
            dict.fromkeys(flags)
        ),
        "audit_flag_count": len(set(flags)),
        "audit_priority": priority,
        "audit_requires_review": priority != "none",
        "audit_label_changed": False,
    })


# ============================================================
# 4. MAIN FUNCTION
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 8.6 - VERIFIED LABEL POLICY AUDIT")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Verified dataset not found: {INPUT_FILE}"
        )

    print("\nLoading verified dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": "string"}
    )

    required_columns = {
        "post_id",
        "text",
        "final_event_category",
        "final_incident_target",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Total reviewed tweets:", len(df))

    # --------------------------------------------------------
    # STEP 1: DATA VALIDATION
    # --------------------------------------------------------

    if df["post_id"].isna().any():
        raise ValueError(
            "Missing post IDs found."
        )

    if df["post_id"].duplicated().any():
        raise ValueError(
            "Duplicate post IDs found."
        )

    df["final_incident_target"] = pd.to_numeric(
        df["final_incident_target"],
        errors="coerce"
    )

    invalid_targets = (
        df["final_incident_target"].notna()
        & ~df["final_incident_target"].isin([0, 1])
    )

    if invalid_targets.any():
        raise ValueError(
            "Invalid binary targets found."
        )

    print("\nHuman category distribution:")
    print(
        df["final_event_category"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nHuman binary target distribution:")
    print(
        df["final_incident_target"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # STEP 2: RUN DIAGNOSTIC AUDIT
    # --------------------------------------------------------

    print("\nRunning broad-event label audit...")

    audit_results = df.apply(
        audit_row,
        axis=1
    )

    audit_df = pd.concat(
        [
            df.reset_index(drop=True),
            audit_results.reset_index(drop=True)
        ],
        axis=1
    )

    # --------------------------------------------------------
    # STEP 3: SORT REVIEW PRIORITIES
    # --------------------------------------------------------

    priority_order = {
        "high": 0,
        "medium": 1,
        "low": 2,
        "none": 3
    }

    audit_df["_priority_order"] = (
        audit_df["audit_priority"]
        .map(priority_order)
    )

    audit_df = audit_df.sort_values(
        by=[
            "_priority_order",
            "audit_flag_count"
        ],
        ascending=[True, False],
        kind="stable"
    ).drop(
        columns=["_priority_order"]
    )

    # --------------------------------------------------------
    # STEP 4: PRINT AUDIT SUMMARY
    # --------------------------------------------------------

    print("\nAUDIT PRIORITY DISTRIBUTION:")

    print(
        audit_df["audit_priority"]
        .value_counts()
        .to_string()
    )

    print("\nREVIEW REQUIRED:")

    print(
        audit_df["audit_requires_review"]
        .value_counts()
        .to_string()
    )

    print("\nHIGH PRIORITY EXAMPLES:")

    high_priority = audit_df[
        audit_df["audit_priority"] == "high"
    ]

    for _, row in high_priority.iterrows():

        print("\nPOST ID:", row["post_id"])

        print(
            "Human category:",
            row["final_event_category"]
        )

        print(
            "Human target:",
            row["final_incident_target"]
        )

        print(
            "Flags:",
            row["audit_flags"]
        )

        print("TEXT:", row["text"])

        print("-" * 65)

    # --------------------------------------------------------
    # STEP 5: CREATE SUMMARY
    # --------------------------------------------------------

    summary = {
        "stage": "8.6",
        "policy": "broad_civil_unrest_event_detection",
        "total_reviewed": int(len(audit_df)),
        "verified_positive": int(
            (audit_df["final_incident_target"] == 1).sum()
        ),
        "verified_negative": int(
            (audit_df["final_incident_target"] == 0).sum()
        ),
        "verified_uncertain": int(
            audit_df["final_incident_target"].isna().sum()
        ),
        "priority_distribution": {
            str(k): int(v)
            for k, v in audit_df[
                "audit_priority"
            ].value_counts().items()
        },
        "review_required_count": int(
            audit_df["audit_requires_review"].sum()
        ),
        "human_labels_modified": False,
        "audit_is_ground_truth": False,
    }

    # --------------------------------------------------------
    # STEP 6: SAVE OUTPUTS
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for path in [AUDIT_FILE, SUMMARY_FILE]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    audit_df.to_csv(
        AUDIT_FILE,
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

    print("\n" + "=" * 65)
    print("OUTPUT FILES SAVED")
    print("=" * 65)

    print(AUDIT_FILE)
    print(SUMMARY_FILE)

    print("\nOriginal verified labels were NOT modified.")
    print("STAGE 8.6 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
