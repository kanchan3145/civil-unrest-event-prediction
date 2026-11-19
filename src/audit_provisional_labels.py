
from pathlib import Path
import re
import pandas as pd


# ============================================================
# CIVIL UNREST PROJECT - STAGE 6.3
# PROVISIONAL LABEL VALIDATION AND INCIDENT AUDIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotation"

INPUT_FILE = ANNOTATION_DIR / "incident_labels_consolidated.csv"

AUDIT_FILE = ANNOTATION_DIR / "incident_provisional_audit.csv"
REVIEW_FILE = ANNOTATION_DIR / "incident_priority_review.csv"
SUMMARY_FILE = ANNOTATION_DIR / "incident_audit_summary.csv"


# ------------------------------------------------------------
# 1. INCIDENT EVIDENCE PATTERNS
# ------------------------------------------------------------

# Stronger expressions describing concrete incidents.
STRONG_PATTERNS = {
    "violent_clash": [
        r"\bclashes?\s+(?:with|between|at|during)\b",
        r"\bviolent\s+clashes?\b",
        r"\bclashed\s+with\b",
    ],
    "police_force": [
        r"\btear[\s-]?gass?(?:ed|ing)?\b",
        r"\blathi[\s-]?charg(?:e|ed|ing)\b",
        r"\bbaton[\s-]?charg(?:e|ed|ing)\b",
        r"\bpolice\s+(?:fired|attacked|beat|beaten)\b",
    ],
    "arrest_or_detention": [
        r"\b(?:arrested|detained|taken into custody)\b",
        r"\bpolice\s+arrests?\b",
    ],
    "injury_or_violence": [
        r"\b(?:injured|wounded|killed)\s+(?:during|in|after)\b",
        r"\bstone[\s-]?pelt(?:ing|ed)\b",
        r"\b(?:attacked|assaulted|beaten)\s+(?:by|during|at)\b",
    ],
    "blockade_or_disruption": [
        r"\b(?:blocked|blockaded)\s+(?:the\s+)?(?:road|highway|railway|tracks)\b",
        r"\brail[\s-]?roko\s+(?:protest|held|began)\b",
    ],
}

# Expressions that may indicate protest activity.
MEDIUM_PATTERNS = {
    "protest_activity": [
        r"\b(?:farmers|protesters)\s+(?:marched|gathered|protested)\b",
        r"\b(?:rally|march|demonstration)\s+(?:held|began|started)\b",
        r"\btractor\s+(?:rally|march|parade)\b",
        r"\b(?:sit[\s-]?in|dharna)\b",
    ],
    "police_presence": [
        r"\bpolice\s+(?:deployed|surrounded|stopped)\b",
        r"\bbarricad(?:e|es|ed|ing)\b",
        r"\bfencing\b",
    ],
    "general_unrest": [
        r"\b(?:riot|riots|violence|unrest)\b",
        r"\b(?:protest|agitation)\s+(?:continued|continues)\b",
    ],
}

# Future or hypothetical expressions.
FUTURE_PATTERNS = [
    r"\b(?:will|would|may|might|could)\b",
    r"\b(?:plan|planned|planning|scheduled|proposed)\b",
    r"\b(?:tomorrow|next week|next month)\b",
    r"\b(?:likely to|expected to|set to)\b",
]

# Explicit negation or denial cues.
NEGATION_PATTERNS = [
    r"\b(?:no|not|never)\s+(?:clash|clashes|violence|riot|riots)\b",
    r"\b(?:false|fake|misleading|debunked)\b",
    r"\bdid not\b",
    r"\b(?:rumour|rumor)\b",
]


def find_matches(text, pattern_groups):
    """Return evidence categories with matched text snippets."""

    found_categories = []
    found_phrases = []

    for category, patterns in pattern_groups.items():
        for pattern in patterns:
            matches = re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            for match in matches:
                found_categories.append(category)
                found_phrases.append(match.group(0))

    return (
        sorted(set(found_categories)),
        sorted(set(found_phrases))
    )


def find_simple_matches(text, patterns):
    """Find matching future or negation expressions."""

    found = []

    for pattern in patterns:
        matches = re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:
            found.append(match.group(0))

    return sorted(set(found))


def audit_text(text):
    """
    Assign an audit priority, not an event label.

    These rules only identify potentially important posts.
    Human review remains necessary.
    """

    if pd.isna(text):
        text = ""

    text = str(text)

    strong_categories, strong_phrases = find_matches(
        text,
        STRONG_PATTERNS
    )

    medium_categories, medium_phrases = find_matches(
        text,
        MEDIUM_PATTERNS
    )

    future_phrases = find_simple_matches(
        text,
        FUTURE_PATTERNS
    )

    negation_phrases = find_simple_matches(
        text,
        NEGATION_PATTERNS
    )

    strong_count = len(strong_categories)
    medium_count = len(medium_categories)

    # Score is for audit ordering only.
    score = (strong_count * 3) + medium_count

    if strong_count > 0:
        priority = "high_priority"
    elif medium_count > 0:
        priority = "medium_priority"
    else:
        priority = "low_priority"

    # Future/negation cues are flags rather than automatic
    # exclusions, because a post can describe both a past
    # incident and a future plan.
    caution_flags = []

    if future_phrases:
        caution_flags.append("future_or_hypothetical")

    if negation_phrases:
        caution_flags.append("negation_or_debunking")

    return {
        "audit_priority": priority,
        "audit_score": score,
        "strong_evidence_categories": "; ".join(strong_categories),
        "medium_evidence_categories": "; ".join(medium_categories),
        "strong_evidence_phrases": "; ".join(strong_phrases),
        "medium_evidence_phrases": "; ".join(medium_phrases),
        "future_cues": "; ".join(future_phrases),
        "negation_cues": "; ".join(negation_phrases),
        "audit_caution_flags": "; ".join(caution_flags),
    }


def detect_text_column(df):
    """Find the tweet text column without selecting a URL column."""

    candidates = [
        "text",
        "renderedContent",
        "rendered_content",
        "clean_text",
        "tweet_text",
        "content",
        "raw_text",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Tweet text column not found.\n"
        f"Available columns: {list(df.columns)}"
    )


def main():

    print("=" * 65)
    print("CIVIL UNREST PROJECT - STAGE 6.3")
    print("PROVISIONAL LABEL VALIDATION AND INCIDENT AUDIT")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "final_label_status",
        "final_event_category",
        "final_incident_target",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    text_column = detect_text_column(df)

    print(f"\nLoaded records: {len(df)}")
    print(f"Text column: {text_column}")

    provisional_mask = (
        df["final_label_status"]
        .astype("string")
        .str.strip()
        .eq("provisional_unverified")
        .fillna(False)
    )

    print(
        f"Provisional records: {int(provisional_mask.sum())}"
    )

    # Audit every record for transparency, but only
    # provisional records enter the review queue.
    audit_results = df[text_column].apply(audit_text)

    audit_df = pd.DataFrame(
        audit_results.tolist(),
        index=df.index
    )

    df = pd.concat([df, audit_df], axis=1)

    # Preserve verified labels.
    df.loc[
        ~provisional_mask,
        "audit_priority"
    ] = "already_verified"

    # Assign deterministic review IDs using original row
    # positions, preserving the existing data order.
    df["audit_record_id"] = [
        f"AUDIT_{i + 1:04d}"
        for i in range(len(df))
    ]

    # Only provisional records are candidates for review.
    review_df = df.loc[provisional_mask].copy()

    priority_order = {
        "high_priority": 0,
        "medium_priority": 1,
        "low_priority": 2,
    }

    review_df["priority_rank"] = (
        review_df["audit_priority"]
        .map(priority_order)
    )

    review_df = review_df.sort_values(
        by=["priority_rank", "audit_score", "audit_record_id"],
        ascending=[True, False, True],
        kind="stable"
    )

    review_df = review_df.drop(
        columns=["priority_rank"]
    )

    # Audit files must not overwrite manually verified labels.
    df.to_csv(AUDIT_FILE, index=False)
    review_df.to_csv(REVIEW_FILE, index=False)

    counts = (
        review_df["audit_priority"]
        .value_counts()
        .reindex(
            [
                "high_priority",
                "medium_priority",
                "low_priority",
            ],
            fill_value=0
        )
    )

    summary = pd.DataFrame({
        "audit_priority": counts.index,
        "post_count": counts.values,
    })

    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nPROVISIONAL AUDIT RESULTS:")
    print(counts.to_string())

    print("\nREVIEW QUEUE:")
    print(f"Total posts: {len(review_df)}")

    print("\nTOP PRIORITY POSTS:")

    preview_columns = [
        "audit_record_id",
        "audit_priority",
        "audit_score",
        text_column,
    ]

    for _, row in review_df.head(10).iterrows():
        print("-" * 65)
        print("ID:", row["audit_record_id"])
        print("Priority:", row["audit_priority"])
        print("Score:", row["audit_score"])
        print("Text:", str(row[text_column])[:350])

    print("\nSaved audit dataset:")
    print(AUDIT_FILE)

    print("\nSaved priority review queue:")
    print(REVIEW_FILE)

    print("\nSaved audit summary:")
    print(SUMMARY_FILE)

    print("\nSTAGE 6.3 AUDIT COMPLETED")


if __name__ == "__main__":
    main()
