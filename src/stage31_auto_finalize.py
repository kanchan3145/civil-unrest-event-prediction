
"""
STAGE 31 - AUTOMATIC EVENT FINALIZATION

Run:
    python src/stage31_auto_finalize.py

Inputs:
    outputs/stage30_v2/temporal_event_records.csv
    outputs/stage31/auto_event_annotations.csv

Outputs:
    outputs/stage31_final/structured_events.csv
    outputs/stage31_final/incident_candidates.csv
    outputs/stage31_final/duplicate_candidates.csv
    outputs/stage31_final/state_week_features.csv
    outputs/stage31_final/quality_audit.json

No manual input.
No model retraining.
No external APIs.
No modification of earlier outputs.
"""

from pathlib import Path
from collections import Counter
import hashlib
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

TEMPORAL = (
    ROOT / "outputs/stage30_v2"
    / "temporal_event_records.csv"
)

AUTO = (
    ROOT / "outputs/stage31"
    / "auto_event_annotations.csv"
)

OUT = ROOT / "outputs/stage31_final"


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(value):
    text = clean(value).lower()
    text = re.sub(r"\[url\]|\[user\]", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def valid_date(value):
    text = clean(value)

    if not text:
        return ""

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return ""

    parsed = pd.to_datetime(text, errors="coerce")

    if pd.isna(parsed):
        return ""

    return parsed.strftime("%Y-%m-%d")


def contains(text, pattern):
    return bool(re.search(pattern, text, flags=re.I))


# Corrections based on the 21 source texts inspected.
# These remain automated, text-based annotations.
# They are NOT independent verification.

CORRECTIONS = {
    "V5-CTX-0002": {
        "event_type": "protest",
        "occurrence": "ongoing",
        "escalation": "unknown",
        "target": "Police action / government",
        "reason": "Protest against earlier lathi charge; no new police action established",
    },
    "V5-CTX-0003": {
        "event_type": "rally",
        "occurrence": "reported_occurred",
        "state": "Punjab",
        "reason": "Occurred Kartarpur rally and separate future Delhi plan",
    },
    "V5-CTX-0004": {
        "event_type": "rally",
        "occurrence": "reported_occurred",
        "state": "Punjab",
        "reason": "Same source text as 0003; no separate Delhi rally established",
    },
    "V5-CTX-0006": {
        "event_type": "protest",
        "occurrence": "unclear",
        "escalation": "unknown",
        "reason": "Slogan about protest against lathi charge; fresh police action not established",
    },
    "V5-CTX-0008": {
        "event_type": "protest",
        "occurrence": "ongoing",
        "reason": "Ongoing farmers protest mention",
    },
    "V5-CTX-0010": {
        "event_type": "protest_action",
        "occurrence": "reported_occurred",
        "escalation": "unknown",
        "reason": "Farmers on police water cannon vehicle; police force not established",
    },
    "V5-CTX-0011": {
        "event_type": "clash",
        "occurrence": "historical_reference",
        "reason": "Later report of Republic Day clashes, not a new February incident",
    },
    "V5-CTX-0012": {
        "event_type": "clash",
        "occurrence": "reported_occurred",
        "reason": "Reported clash at ITO; weekday-based date remains unconfirmed",
    },
    "V5-CTX-0014": {
        "event_type": "clash",
        "occurrence": "reported_occurred",
        "reason": "Reported Red Fort clash; occurrence date not explicit",
    },
    "V5-CTX-0016": {
        "event_type": "protest",
        "occurrence": "ongoing",
        "reason": "Ongoing protest and planned meetings; not a new meeting occurrence",
    },
}


def infer_occurrence(text, existing):
    """
    Preserve existing predictions unless a clearer
    textual rule applies.
    """

    if contains(
        text,
        r"\bfresh videos\b|\banniversary\b",
    ):
        return "historical_reference"

    if contains(
        text,
        r"\bwere protesting\b|\bclashed\b|"
        r"\binjured\b|\bparticipated\b|"
        r"\borganized\b|\bknocked down\b|"
        r"\bpolice fire\b|\bpolice use\b",
    ):
        return "reported_occurred"

    if contains(
        text,
        r"\bare protesting\b|\bthey are holding\b|"
        r"\bprotesting against\b",
    ):
        return "ongoing"

    return existing or "unclear"


def process_record(row):
    cid = clean(row["cluster_id"])
    text = normalize_text(row["representative_text"])

    event_type = clean(row["auto_event_type"])
    state = clean(row["auto_state"])
    occurrence = clean(row["auto_occurrence_status"])
    actor = clean(row["auto_actor"])
    target = clean(row["auto_target"])
    scale = clean(row["auto_scale"]) or "unknown"
    escalation = clean(row["auto_escalation"]) or "unknown"

    occurrence = infer_occurrence(text, occurrence)

    correction = CORRECTIONS.get(cid, {})

    event_type = correction.get("event_type", event_type)
    state = correction.get("state", state)
    occurrence = correction.get("occurrence", occurrence)
    target = correction.get("target", target)
    escalation = correction.get("escalation", escalation)

    # Preserve date evidence from Stage 30 V2.
    date = valid_date(row["resolved_event_date"])
    date_basis = clean(row["resolved_date_basis"])
    date_confidence = clean(row["date_confidence"])

    # Weekday matching is not enough to establish
    # an event date.
    if date_basis == "same_weekday_assumption":
        date = ""
        date_confidence = "unconfirmed"

    # Do not assign post dates to undated events.
    # Do not treat a future plan as an occurred event.

    if occurrence in {"planned_future", "unclear"}:
        eligible_occurrence = False
    else:
        eligible_occurrence = occurrence in {
            "reported_occurred",
            "ongoing",
            "historical_reference",
        }

    # Historical reports may describe real past events,
    # but need deduplication before counting.
    eligible = (
        bool(date)
        and bool(state)
        and eligible_occurrence
    )

    # All date-resolved records in this dataset are
    # provisional and require verification.
    return {
        "cluster_id": cid,
        "source_post_date": clean(
            row["reference_post_date"]
        ),
        "event_type": event_type,
        "state": state,
        "location_candidate": clean(
            row["location_candidate"]
        ),
        "event_date": date,
        "date_basis": date_basis,
        "date_confidence": date_confidence,
        "occurrence_status": occurrence,
        "actors": actor,
        "targets": target,
        "scale": scale,
        "escalation": escalation,
        "representative_text": clean(
            row["representative_text"]
        ),
        "supporting_post_ids": clean(
            row.get("supporting_post_ids", "")
        ),
        "normalized_text": text,
        "text_fingerprint": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        "correction_reason": correction.get(
            "reason", ""
        ),
        "date_resolved": bool(date),
        "provisional_aggregation_eligible": eligible,
        "annotation_method": "automatic_rules_with_text_reviewed_corrections",
        "verification_status": "unverified_text_candidate",
    }


def main():
    if not TEMPORAL.exists():
        raise FileNotFoundError(TEMPORAL)

    if not AUTO.exists():
        raise FileNotFoundError(AUTO)

    temporal = pd.read_csv(
        TEMPORAL,
        dtype={"cluster_id": "string"},
        keep_default_na=False,
        low_memory=False,
    )

    auto = pd.read_csv(
        AUTO,
        dtype={"cluster_id": "string"},
        keep_default_na=False,
        low_memory=False,
    )

    required_temporal = {
        "cluster_id",
        "representative_text",
        "reference_post_date",
        "resolved_event_date",
        "resolved_date_basis",
        "date_confidence",
        "location_candidate",
    }

    required_auto = {
        "cluster_id",
        "auto_event_type",
        "auto_state",
        "auto_occurrence_status",
        "auto_actor",
        "auto_target",
        "auto_scale",
        "auto_escalation",
    }

    if required_temporal - set(temporal.columns):
        raise ValueError("Missing Stage 30 V2 columns")

    if required_auto - set(auto.columns):
        raise ValueError("Missing automatic annotation columns")

    if (
        temporal["cluster_id"].duplicated().any()
        or auto["cluster_id"].duplicated().any()
    ):
        raise ValueError("Duplicate cluster IDs in input")

    # Keep only automatic prediction columns from AUTO.
    # Stage 30 V2 remains authoritative for source text
    # and temporal evidence.
    auto_cols = ["cluster_id"] + sorted(
        required_auto - {"cluster_id"}
    )

    merged = temporal.merge(
        auto[auto_cols],
        on="cluster_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not merged["_merge"].eq("both").all():
        raise ValueError(
            "Some Stage 30 clusters have no automatic annotations"
        )

    records = pd.DataFrame(
        [
            process_record(row)
            for _, row in merged.iterrows()
        ]
    )

    # Exact repeated text indicates shared evidence,
    # not necessarily independent incident confirmation.
    records["same_text_count"] = (
        records.groupby("text_fingerprint")[
            "cluster_id"
        ].transform("size")
    )

    records["duplicate_text_candidate"] = (
        records["same_text_count"] > 1
    )

    duplicate_rows = records[
        records["duplicate_text_candidate"]
    ].copy()

    # Conservative candidate consolidation:
    # count at most one record per identical text.
    #
    # Different texts can still describe the same
    # real incident. We cannot automatically establish
    # uniqueness from this small dataset.
    incident_candidates = (
        records.sort_values("cluster_id")
        .drop_duplicates(
            subset=["text_fingerprint"],
            keep="first",
        )
        .copy()
    )

    incident_candidates["candidate_incident_id"] = [
        f"AUTO-INC-{i:04d}"
        for i in range(1, len(incident_candidates) + 1)
    ]

    eligible = incident_candidates[
        incident_candidates[
            "provisional_aggregation_eligible"
        ]
    ].copy()

    if not eligible.empty:
        eligible["week"] = (
            pd.to_datetime(
                eligible["event_date"],
                errors="coerce",
            )
            .dt.to_period("W-SUN")
            .dt.start_time
            .dt.strftime("%Y-%m-%d")
        )

        weekly = (
            eligible.groupby(
                ["state", "week"],
                as_index=False,
            )
            .agg(
                provisional_candidate_count=(
                    "candidate_incident_id",
                    "nunique",
                ),
                candidate_event_types=(
                    "event_type",
                    lambda x: "; ".join(
                        sorted(set(x))
                    ),
                ),
            )
        )
    else:
        weekly = pd.DataFrame(
            columns=[
                "state",
                "week",
                "provisional_candidate_count",
                "candidate_event_types",
            ]
        )

    weekly["feature_status"] = (
        "provisional_unverified_not_training_ground_truth"
    )

    OUT.mkdir(parents=True, exist_ok=True)

    records.to_csv(
        OUT / "structured_events.csv",
        index=False,
    )

    incident_candidates.to_csv(
        OUT / "incident_candidates.csv",
        index=False,
    )

    duplicate_rows.to_csv(
        OUT / "duplicate_candidates.csv",
        index=False,
    )

    weekly.to_csv(
        OUT / "state_week_features.csv",
        index=False,
    )

    audit = {
        "input_clusters": int(len(records)),
        "auto_annotated_clusters": int(len(records)),
        "clusters_with_dates": int(
            records["date_resolved"].sum()
        ),
        "clusters_without_dates": int(
            (~records["date_resolved"]).sum()
        ),
        "duplicate_text_rows": int(
            len(duplicate_rows)
        ),
        "unique_text_candidates": int(
            len(incident_candidates)
        ),
        "provisional_aggregation_candidates": int(
            len(eligible)
        ),
        "provisional_state_weeks": int(
            len(weekly)
        ),
        "verified_incidents": 0,
        "event_type_distribution": dict(
            Counter(records["event_type"])
        ),
        "occurrence_distribution": dict(
            Counter(records["occurrence_status"])
        ),
        "limitations": [
            "All records are automatic, text-based candidates.",
            "Exact-text deduplication does not prove incident uniqueness.",
            "Different posts may report the same incident.",
            "Unknown occurrence dates remain missing.",
            "No externally verified incidents are established.",
            "No extraction accuracy metric is claimed.",
            "State-week counts are not approved forecasting ground truth.",
            "A held-out human-labeled evaluation set is still required.",
        ],
    }

    (
        OUT / "quality_audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSTAGE 31 AUTOMATIC FINALIZATION")
    print("=" * 55)
    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nOutput directory:", OUT)
    print("\nCompleted without interactive annotation.")


if __name__ == "__main__":
    main()
