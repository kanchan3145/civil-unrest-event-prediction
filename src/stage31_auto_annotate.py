
"""
Stage 31: Automatic pre-annotation.

Uses existing Stage 30 V2 records.
Does not retrain models or overwrite manual reviews.

Run:
    python src/stage31_auto_annotate.py
"""

from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs/stage30_v2/temporal_event_records.csv"
OUT = ROOT / "outputs/stage31"
MANUAL = OUT / "validated_event_annotations.csv"

ACTOR_PATTERNS = {
    "Police": r"\bpolice\b|\bcops\b|\bSHO\b",
    "Farmers": r"\bfarmers?\b|\bkisan\b",
    "Government": r"\bgovernment\b|\bBJP\b|\bCM Khattar\b",
    "Local residents": r"\blocals\b|\bresidents\b",
}

VIOLENCE_PATTERN = (
    r"\bclash(?:es|ed)?\b|"
    r"\blathi[\s-]?charge\b|"
    r"\bviolence\b|\bviolent\b|"
    r"\btear gas\b|\binjur(?:y|ed|ies)\b|"
    r"\bbrutality\b"
)

FUTURE_PATTERN = (
    r"\bwill\b|\bplan(?:ned|ning)?\b|"
    r"\bexpected to\b|\bgoing to\b|"
    r"\bdecided to organise\b"
)

RETROSPECTIVE_PATTERN = (
    r"\bfresh videos\b|\bremember\b|"
    r"\banniversary\b|\bpreviously\b"
)

REQUIRED = {
    "cluster_id",
    "event_type",
    "state",
    "resolved_event_date",
    "date_confidence",
    "temporal_status",
    "representative_text",
    "duplicate_text_review",
}


def value(x):
    return "" if pd.isna(x) else str(x).strip()


def matches(pattern, text):
    return bool(re.search(pattern, text, flags=re.I))


def is_true(x):
    return value(x).lower() in {"true", "1", "yes"}


def infer_actors(text):
    found = [
        name
        for name, pattern in ACTOR_PATTERNS.items()
        if matches(pattern, text)
    ]
    return "; ".join(found)


def infer_target(text, event_type):
    if (
        event_type == "police_action"
        and matches(r"\bfarmers?\b|\bprotesters?\b", text)
    ):
        return "Farmers/protesters"

    if (
        event_type == "clash"
        and matches(r"\bpolice\b|\bcops\b", text)
        and matches(r"\bfarmers?\b", text)
    ):
        return "Police and farmers (clash participants)"

    return ""


def infer_scale(text):
    lower = text.lower()

    if matches(r"\btens of thousands\b|\bthousands\b", lower):
        return "large"

    if matches(r"\bhundreds\b", lower):
        return "medium"

    return "unknown"


def infer_occurrence(text, event_type):
    if matches(RETROSPECTIVE_PATTERN, text):
        return "historical_reference"

    if matches(
        r"\bdecided to organise\b|\bwill hold\b|"
        r"\bwill head\b|\bplans to hold\b",
        text,
    ):
        # A post may describe both an occurred event
        # and a future plan. Review the distinction.
        return "unclear"

    if matches(
        r"\bclashed\b|\binjured\b|\borganized\b|"
        r"\bparticipated\b|\bwere protesting\b|"
        r"\blathi[\s-]?charge\b|\bfiring\b|"
        r"\bknocked down\b",
        text,
    ):
        return "reported_occurred"

    if matches(
        r"\bare protesting\b|\bprotesting\b|"
        r"\bclash with\b|\bclashes with\b",
        text,
    ):
        return "ongoing"

    return "unclear"


def annotate(row):
    text = value(row["representative_text"])
    event_type = value(row["event_type"])
    state = value(row["state"])
    date = value(row["resolved_event_date"])
    date_conf = value(row["date_confidence"])
    temporal_status = value(row["temporal_status"])

    duplicate = is_true(row["duplicate_text_review"])
    occurrence = infer_occurrence(text, event_type)

    escalation = (
        "yes"
        if matches(VIOLENCE_PATTERN, text)
        else "unknown"
    )

    flags = []

    if not date:
        flags.append("missing_occurrence_date")

    if date_conf in {"low", "none", ""}:
        flags.append("weak_temporal_evidence")

    if temporal_status == "requires_review":
        flags.append("date_assumption_requires_review")

    if duplicate:
        flags.append("duplicate_representative_text")

    if occurrence == "unclear":
        flags.append("occurrence_uncertain")

    if matches(FUTURE_PATTERN, text):
        flags.append("future_reference_present")

    if matches(RETROSPECTIVE_PATTERN, text):
        flags.append("retrospective_reference_present")

    # These flags are hints, not calibrated
    # statistical confidence probabilities.
    if (
        not flags
        and date_conf == "high"
        and occurrence in {"reported_occurred", "ongoing"}
    ):
        review_priority = "low"
    elif (
        duplicate
        or occurrence == "unclear"
        or temporal_status == "requires_review"
    ):
        review_priority = "high"
    else:
        review_priority = "medium"

    return {
        "cluster_id": value(row["cluster_id"]),
        "auto_event_label": (
            "event_candidate"
            if occurrence != "unclear"
            else "uncertain"
        ),
        "auto_event_type": event_type,
        "auto_state": state,
        "auto_event_date": date,
        "auto_occurrence_status": occurrence,
        "auto_actor": infer_actors(text),
        "auto_target": infer_target(text, event_type),
        "auto_scale": infer_scale(text),
        "auto_escalation": escalation,
        "auto_incident_group": "",
        "auto_evidence_level": "text_only_unverified",
        "review_priority": review_priority,
        "review_reasons": "; ".join(flags),
        "annotation_source": "automatic_rule_based",
    }


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(
        SOURCE,
        dtype={"cluster_id": "string"},
        keep_default_na=False,
        low_memory=False,
    )

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["cluster_id"].duplicated().any():
        raise ValueError("Duplicate cluster IDs in source")

    predictions = pd.DataFrame(
        [annotate(row) for _, row in df.iterrows()]
    )

    # Preserve the user's existing manual work.
    if MANUAL.exists():
        manual = pd.read_csv(
            MANUAL,
            dtype={"cluster_id": "string"},
            keep_default_na=False,
        )

        if "review_status" in manual.columns:
            completed = set(
                manual.loc[
                    manual["review_status"] == "reviewed",
                    "cluster_id",
                ].astype(str)
            )
        else:
            completed = set()
    else:
        completed = set()

    predictions["already_manually_reviewed"] = (
        predictions["cluster_id"]
        .astype(str)
        .isin(completed)
    )

    combined = df.merge(
        predictions,
        on="cluster_id",
        how="left",
        validate="one_to_one",
    )

    queue = combined[
        ~combined["already_manually_reviewed"]
    ].copy()

    priority_order = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }

    queue["_priority"] = (
        queue["review_priority"].map(priority_order)
    )

    queue = (
        queue.sort_values("_priority")
        .drop(columns="_priority")
    )

    OUT.mkdir(parents=True, exist_ok=True)

    combined.to_csv(
        OUT / "auto_event_annotations.csv",
        index=False,
    )

    queue.to_csv(
        OUT / "auto_review_queue.csv",
        index=False,
    )

    summary = {
        "total_clusters": len(combined),
        "automatically_preannotated": len(predictions),
        "previously_manually_reviewed": len(completed),
        "remaining_review_queue": len(queue),
        "priority_distribution": (
            queue["review_priority"]
            .value_counts()
            .to_dict()
        ),
        "verified_incidents": 0,
        "notes": [
            "Automatic predictions are not ground truth.",
            "Dates remain unknown unless supported.",
            "Incident IDs are not automatically merged.",
            "Human-reviewed records are preserved.",
            "Rule confidence is not model accuracy.",
        ],
    }

    (
        OUT / "auto_annotation_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nSaved automatic annotations and review queue.")


if __name__ == "__main__":
    main()
