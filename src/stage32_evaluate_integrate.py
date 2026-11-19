
"""
STAGE 32 - AUTOMATED EVALUATION AND INTEGRATION

Run:
    python src/stage32_evaluate_integrate.py

Input:
    outputs/stage31_final/structured_events.csv

Outputs:
    outputs/stage32/
        extraction_coverage.csv
        event_type_distribution.csv
        occurrence_distribution.csv
        quality_issues.csv
        duplicate_groups.csv
        event_dashboard_data.csv
        exploratory_state_week_features.csv
        evaluation_summary.json
        evaluation_report.md
        event_type_distribution.png
        extraction_coverage.png

No manual labeling.
No model training.
No external API calls.
No changes to earlier stages.
"""

from pathlib import Path
from collections import Counter
import hashlib
import json
import re

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT / "outputs"
    / "stage31_final"
    / "structured_events.csv"
)

OUT = ROOT / "outputs" / "stage32"


REQUIRED_COLUMNS = {
    "cluster_id",
    "event_type",
    "state",
    "location_candidate",
    "event_date",
    "date_basis",
    "date_confidence",
    "occurrence_status",
    "actors",
    "targets",
    "scale",
    "escalation",
    "representative_text",
    "source_post_date",
    "verification_status",
}

UNKNOWN_VALUES = {
    "",
    "unknown",
    "unclear",
    "none",
    "null",
    "nan",
    "not_available",
    "not available",
    "unresolved",
    "n/a",
}

# Only these occurrence types are eligible for
# exploratory event-week analysis.
# Ongoing protests are excluded because a mention
# does not establish a new incident in that week.
ELIGIBLE_OCCURRENCES = {
    "reported_occurred",
}

# Date evidence that may support a provisional
# occurrence date. This is not external verification.
ACCEPTABLE_DATE_BASES = {
    "relative_today",
    "republic_day_context",
    "explicit_event_date",
    "explicit_date",
    "explicit_full_date",
}

# No automatic record is approved as forecasting
# ground truth in this stage.


def value(row, column):
    result = row.get(column, "")
    if pd.isna(result):
        return ""
    return str(result).strip()


def known(text):
    return str(text).strip().lower() not in UNKNOWN_VALUES


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\[user\]|\[url\]", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_date(text):
    text = str(text).strip()

    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", text
    ):
        return None

    dt = pd.to_datetime(
        text,
        format="%Y-%m-%d",
        errors="coerce",
    )

    if pd.isna(dt):
        return None

    return dt


def date_available_at_publication(event_date, post_date):
    """
    An event cannot be known from a post before the
    post was published.

    This check only catches obvious contradictions.
    It does not establish when the event was first
    observable in the full social-media stream.
    """
    if event_date is None or post_date is None:
        return False

    return event_date <= post_date


def save_chart(labels, values, title, filename, ylabel):
    plt.figure(figsize=(9, 5))

    plt.bar(labels, values)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing Stage 31 input: {INPUT}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT,
        keep_default_na=False,
        dtype={"cluster_id": "string"},
        low_memory=False,
    )

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df["cluster_id"].duplicated().any():
        raise ValueError(
            "Duplicate cluster_id values in Stage 31 input"
        )

    if df.empty:
        raise ValueError("Stage 31 input is empty")

    # --------------------------------------------
    # A. FIELD COVERAGE
    # --------------------------------------------

    fields = {
        "Event type": "event_type",
        "State": "state",
        "Location": "location_candidate",
        "Occurrence date": "event_date",
        "Occurrence status": "occurrence_status",
        "Actors": "actors",
        "Targets": "targets",
        "Scale": "scale",
        "Escalation": "escalation",
    }

    coverage_rows = []

    for display_name, column in fields.items():
        count = int(
            df[column]
            .map(known)
            .sum()
        )

        coverage_rows.append({
            "field": display_name,
            "available": count,
            "missing_or_unknown": len(df) - count,
            "total": len(df),
            "coverage_percent": round(
                100 * count / len(df), 2
            ),
        })

    coverage = pd.DataFrame(coverage_rows)

    coverage.to_csv(
        OUT / "extraction_coverage.csv",
        index=False,
    )

    # --------------------------------------------
    # B. DISTRIBUTIONS
    # --------------------------------------------

    event_dist = (
        df["event_type"]
        .value_counts()
        .rename_axis("event_type")
        .reset_index(name="count")
    )

    occurrence_dist = (
        df["occurrence_status"]
        .value_counts()
        .rename_axis("occurrence_status")
        .reset_index(name="count")
    )

    event_dist.to_csv(
        OUT / "event_type_distribution.csv",
        index=False,
    )

    occurrence_dist.to_csv(
        OUT / "occurrence_distribution.csv",
        index=False,
    )

    # --------------------------------------------
    # C. DUPLICATE GROUPS
    # --------------------------------------------

    df["normalized_text_stage32"] = (
        df["representative_text"]
        .map(normalize_text)
    )

    df["text_hash_stage32"] = (
        df["normalized_text_stage32"]
        .map(
            lambda x: hashlib.sha256(
                x.encode("utf-8")
            ).hexdigest()
        )
    )

    duplicate_size = (
        df.groupby("text_hash_stage32")[
            "cluster_id"
        ].transform("size")
    )

    df["exact_text_duplicate"] = (
        duplicate_size > 1
    )

    duplicate_groups = df[
        df["exact_text_duplicate"]
    ][
        [
            "cluster_id",
            "event_type",
            "state",
            "event_date",
            "text_hash_stage32",
            "representative_text",
        ]
    ].copy()

    duplicate_groups.to_csv(
        OUT / "duplicate_groups.csv",
        index=False,
    )

    unique_text_count = int(
        df["text_hash_stage32"].nunique()
    )

    # --------------------------------------------
    # D. TEMPORAL + QUALITY AUDIT
    # --------------------------------------------

    quality_rows = []
    dashboard_rows = []

    for _, row in df.iterrows():
        cid = value(row, "cluster_id")

        event_date = parse_date(
            value(row, "event_date")
        )

        post_date = parse_date(
            value(row, "source_post_date")
        )

        occurrence = value(
            row, "occurrence_status"
        )

        date_basis = value(
            row, "date_basis"
        )

        issues = []

        if event_date is None:
            issues.append("missing_event_date")

        if not known(value(row, "state")):
            issues.append("missing_state")

        if not known(value(row, "event_type")):
            issues.append("missing_event_type")

        if occurrence in {
            "unclear",
            "planned_future",
            "planned_or_future",
        }:
            issues.append(
                "uncertain_or_future_occurrence"
            )

        if occurrence in {
            "ongoing",
            "historical_reference",
        }:
            issues.append(
                "not_a_new_incident_without_more_evidence"
            )

        if bool(row["exact_text_duplicate"]):
            issues.append(
                "duplicate_representative_text"
            )

        if (
            event_date is not None
            and date_basis not in ACCEPTABLE_DATE_BASES
        ):
            issues.append(
                "date_basis_not_approved"
            )

        if (
            event_date is not None
            and post_date is not None
            and event_date > post_date
        ):
            issues.append(
                "event_date_after_source_post"
            )

        if post_date is None:
            issues.append(
                "missing_source_post_date"
            )

        eligible = (
            event_date is not None
            and post_date is not None
            and date_available_at_publication(
                event_date, post_date
            )
            and known(value(row, "state"))
            and known(value(row, "event_type"))
            and occurrence in ELIGIBLE_OCCURRENCES
            and date_basis in ACCEPTABLE_DATE_BASES
        )

        # Do not interpret a candidate as verified.
        dashboard_rows.append({
            "cluster_id": cid,
            "event_type": value(
                row, "event_type"
            ),
            "state": value(row, "state"),
            "location": value(
                row, "location_candidate"
            ),
            "event_date": (
                event_date.strftime("%Y-%m-%d")
                if event_date is not None
                else ""
            ),
            "source_post_date": value(
                row, "source_post_date"
            ),
            "occurrence_status": occurrence,
            "actors": value(row, "actors"),
            "targets": value(row, "targets"),
            "scale": value(row, "scale"),
            "escalation": value(
                row, "escalation"
            ),
            "representative_text": value(
                row, "representative_text"
            ),
            "date_basis": date_basis,
            "date_confidence": value(
                row, "date_confidence"
            ),
            "verification_status": (
                "unverified_text_candidate"
            ),
            "exact_text_duplicate": bool(
                row["exact_text_duplicate"]
            ),
            "exploratory_week_eligible": eligible,
            "approved_for_forecast_training": False,
            "quality_flags": "; ".join(issues),
        })

        for issue in issues:
            quality_rows.append({
                "cluster_id": cid,
                "issue": issue,
            })

    quality = pd.DataFrame(
        quality_rows,
        columns=["cluster_id", "issue"],
    )

    quality.to_csv(
        OUT / "quality_issues.csv",
        index=False,
    )

    dashboard = pd.DataFrame(
        dashboard_rows
    )

    dashboard.to_csv(
        OUT / "event_dashboard_data.csv",
        index=False,
    )

    # --------------------------------------------
    # E. CONSERVATIVE EXPLORATORY AGGREGATION
    # --------------------------------------------

    eligible = dashboard[
        dashboard["exploratory_week_eligible"]
    ].copy()

    # Exact-text deduplication only.
    # This does NOT eliminate distinct-text reports
    # of the same real-world incident.

    if not eligible.empty:
        hashes = df.set_index(
            "cluster_id"
        )["text_hash_stage32"]

        eligible["text_hash_stage32"] = (
            eligible["cluster_id"].map(hashes)
        )

        eligible = eligible.drop_duplicates(
            subset=["text_hash_stage32"]
        )

        eligible["week"] = (
            pd.to_datetime(
                eligible["event_date"]
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
                provisional_report_count=(
                    "cluster_id", "nunique"
                ),
                event_types=(
                    "event_type",
                    lambda s: "; ".join(
                        sorted(set(s))
                    ),
                ),
            )
        )

    else:
        weekly = pd.DataFrame(
            columns=[
                "state",
                "week",
                "provisional_report_count",
                "event_types",
            ]
        )

    weekly["verification_status"] = (
        "unverified_report_counts"
    )

    weekly["approved_for_forecast_training"] = False

    weekly.to_csv(
        OUT / "exploratory_state_week_features.csv",
        index=False,
    )

    # --------------------------------------------
    # F. VISUALIZATIONS
    # --------------------------------------------

    save_chart(
        event_dist["event_type"].tolist(),
        event_dist["count"].tolist(),
        "Automatically Assigned Event Types",
        "event_type_distribution.png",
        "Number of clusters",
    )

    save_chart(
        coverage["field"].tolist(),
        coverage["coverage_percent"].tolist(),
        "Structured Extraction Field Coverage",
        "extraction_coverage.png",
        "Coverage (%)",
    )

    # --------------------------------------------
    # G. MACHINE-READABLE SUMMARY
    # --------------------------------------------

    issue_counts = dict(
        Counter(quality["issue"])
    )

    summary = {
        "stage": 32,
        "status": "completed",
        "input_clusters": int(len(df)),
        "unique_text_candidates": unique_text_count,
        "exact_duplicate_rows": int(
            df["exact_text_duplicate"].sum()
        ),
        "resolved_event_dates": int(
            df["event_date"]
            .map(
                lambda x: parse_date(x)
                is not None
            )
            .sum()
        ),
        "exploratory_eligible_reports_after_exact_dedup": int(
            len(eligible)
        ),
        "exploratory_state_weeks": int(
            len(weekly)
        ),
        "independently_verified_incidents": 0,
        "approved_forecasting_records": 0,
        "coverage_percent": {
            row["field"]: float(
                row["coverage_percent"]
            )
            for _, row in coverage.iterrows()
        },
        "event_type_distribution": {
            str(row["event_type"]): int(
                row["count"]
            )
            for _, row in event_dist.iterrows()
        },
        "occurrence_distribution": {
            str(row["occurrence_status"]): int(
                row["count"]
            )
            for _, row in occurrence_dist.iterrows()
        },
        "quality_issue_counts": issue_counts,
        "evaluation": {
            "extraction_coverage_computed": True,
            "consistency_audit_computed": True,
            "precision": None,
            "recall": None,
            "f1": None,
            "reason": (
                "Independent ground-truth event labels "
                "are not available for these 21 clusters."
            ),
        },
        "forecasting_integration": {
            "existing_forecasting_results_unchanged": True,
            "new_event_features_used_for_training": False,
            "reason": (
                "Event candidates are unverified, "
                "may describe the same incidents, "
                "and lack reliable temporal coverage."
            ),
        },
    }

    (
        OUT / "evaluation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------
    # H. REPORT
    # --------------------------------------------

    report = [
        "# Stage 32 — Automated Extraction Evaluation",
        "",
        "## Dataset",
        f"- Input clusters: {len(df)}",
        f"- Unique representative texts: {unique_text_count}",
        f"- Exact duplicate rows: {int(df['exact_text_duplicate'].sum())}",
        f"- Resolved occurrence dates: {summary['resolved_event_dates']}",
        "",
        "## Extraction Coverage",
        "",
        "| Field | Available | Total | Coverage |",
        "|---|---:|---:|---:|",
    ]

    for _, r in coverage.iterrows():
        report.append(
            f"| {r['field']} | "
            f"{r['available']} | "
            f"{r['total']} | "
            f"{r['coverage_percent']:.2f}% |"
        )

    report.extend([
        "",
        "## Event Type Distribution",
        "",
        "| Type | Clusters |",
        "|---|---:|",
    ])

    for _, r in event_dist.iterrows():
        report.append(
            f"| {r['event_type']} | {r['count']} |"
        )

    report.extend([
        "",
        "## Temporal and Incident Quality",
        "",
        f"- Exploratory eligible reports after exact-text deduplication: {len(eligible)}",
        f"- Exploratory state-weeks: {len(weekly)}",
        "- Verified incidents: 0",
        "- Approved forecasting records: 0",
        "",
        "## Evaluation Interpretation",
        "",
        "Coverage measures how often a field has a usable value.",
        "It does not measure whether the extracted value is correct.",
        "",
        "Precision, recall, and F1 are not reported because",
        "an independent ground-truth annotation set is unavailable.",
        "",
        "## Forecasting Integration",
        "",
        "The existing chronological forecasting experiments",
        "remain unchanged. These unverified candidate records",
        "are not introduced as training targets or predictors.",
        "",
        "Exploratory weekly counts are report counts, not",
        "independently established unique incident counts.",
        "",
        "## Limitations",
        "",
        "- Exact-text matching misses different reports of the same incident.",
        "- Most occurrence dates are unresolved.",
        "- Date context is not equivalent to external verification.",
        "- Ongoing protest mentions do not necessarily indicate new incidents.",
        "- Publication-time availability must be respected in forecasting.",
        "- Independent annotation is required for precision, recall, and F1.",
    ])

    (
        OUT / "evaluation_report.md"
    ).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\nSTAGE 32 EVALUATION")
    print("=" * 55)

    print("Input clusters:", len(df))
    print(
        "Unique text candidates:",
        unique_text_count,
    )
    print(
        "Resolved occurrence dates:",
        summary["resolved_event_dates"],
    )
    print(
        "Exploratory eligible reports:",
        len(eligible),
    )
    print(
        "Exploratory state-weeks:",
        len(weekly),
    )
    print("Verified incidents:", 0)
    print("Approved forecasting records:", 0)

    print("\nFIELD COVERAGE")
    print(coverage.to_string(index=False))

    print("\nEVENT TYPE DISTRIBUTION")
    print(event_dist.to_string(index=False))

    print("\nQUALITY ISSUE COUNTS")
    print(json.dumps(issue_counts, indent=2))

    print("\nOutput directory:", OUT)
    print("\nStage 32 completed successfully.")


if __name__ == "__main__":
    main()
