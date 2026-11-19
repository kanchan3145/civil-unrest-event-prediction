
"""
Stage 30 V2: Evidence-based temporal extraction.

Input:
 data/consolidated/incident_clusters_v5_v2.csv

Outputs:
 outputs/stage30_v2/temporal_event_records.csv
 outputs/stage30_v2/provisional_state_week_features.csv
 outputs/stage30_v2/temporal_audit.json
 outputs/stage30_v2/manual_review_queue.csv

No training, no external APIs, no raw tweet processing.
"""

from pathlib import Path
from datetime import timedelta
import calendar
import html
import json
import re

import pandas as pd

from stage30_integrate import resolve_state


ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT / "data/consolidated"
    / "incident_clusters_v5_v2.csv"
)

OUT = ROOT / "outputs/stage30_v2"

WEEKDAYS = {
    day.lower(): i
    for i, day in enumerate(calendar.day_name)
}

MONTHS = {
    month.lower(): i
    for i, month in enumerate(calendar.month_name)
    if month
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    return html.unescape(str(value))


def date_string(value):
    if pd.isna(value):
        return ""

    parsed = pd.to_datetime(
        value, errors="coerce", utc=True
    )

    if pd.isna(parsed):
        return ""

    return parsed.strftime("%Y-%m-%d")


def extract_temporal_evidence(text, post_date):
    """
    Returns:
      event_date
      date_basis
      temporal_evidence
      confidence
      temporal_status

    Important:
    A publication date is not automatically
    an event occurrence date.
    """

    text = normalize_text(text)
    lower = text.lower()

    anchor = pd.to_datetime(
        post_date, errors="coerce"
    )

    if pd.isna(anchor):
        anchor = None

    # -----------------------------------------
    # 1. Explicit ISO calendar date
    # -----------------------------------------

    iso = re.search(
        r"\b(20\d{2})-(\d{2})-(\d{2})\b",
        text,
    )

    if iso:
        date = date_string(iso.group(0))

        if date:
            return (
                date,
                "explicit_iso_date",
                iso.group(0),
                "high",
                "date_resolved",
            )

    # -----------------------------------------
    # 2. Named calendar date, e.g. 26 January
    # -----------------------------------------

    named = re.search(
        r"\b(\d{1,2})\s+"
        r"(January|February|March|April|May|June|"
        r"July|August|September|October|November|December)"
        r"(?:\s+(20\d{2}))?\b",
        text,
        flags=re.I,
    )

    if named:
        day = int(named.group(1))
        month = MONTHS[named.group(2).lower()]
        year = (
            int(named.group(3))
            if named.group(3)
            else anchor.year if anchor is not None
            else None
        )

        if year is not None:
            candidate = date_string(
                f"{year:04d}-{month:02d}-{day:02d}"
            )

            if candidate:
                return (
                    candidate,
                    "explicit_calendar_date",
                    named.group(0),
                    "high" if named.group(3)
                    else "medium",
                    "date_resolved",
                )

    # -----------------------------------------
    # 3. Relative expressions
    # -----------------------------------------

    if anchor is not None:

        relative_patterns = [
            (
                r"\byesterday\b",
                -1,
                "relative_yesterday",
            ),
            (
                r"\bthis morning\b|\btoday\b"
                r"|\btonight\b",
                0,
                "relative_same_day",
            ),
        ]

        for pattern, offset, basis in relative_patterns:
            match = re.search(pattern, lower)

            if match:
                resolved = (
                    anchor + timedelta(days=offset)
                ).strftime("%Y-%m-%d")

                return (
                    resolved,
                    basis,
                    match.group(0),
                    "medium",
                    "date_resolved",
                )

    # -----------------------------------------
    # 4. Republic Day contextual date
    # -----------------------------------------

    # India celebrates Republic Day on 26 January.
    # The reference year is used only when the
    # source text refers to that occasion.
    #
    # A distant historical reference is not
    # automatically assigned to the post year.

    republic = re.search(
        r"\brepublic day\b",
        lower,
    )

    if republic and anchor is not None:

        candidate = pd.Timestamp(
            year=anchor.year,
            month=1,
            day=26,
        )

        distance = (
            anchor.normalize() - candidate
        ).days

        # Restrict inference to nearby posts.
        if 0 <= distance <= 14:
            return (
                candidate.strftime("%Y-%m-%d"),
                "republic_day_context",
                republic.group(0),
                "medium",
                "date_resolved",
            )

    # -----------------------------------------
    # 5. Weekday expressions
    # -----------------------------------------

    # "on Tuesday" is potentially ambiguous.
    # We only infer a date when the publication
    # day matches the stated weekday.
    #
    # The result is explicitly labeled as an
    # assumption requiring human review.

    if anchor is not None:

        weekday_match = re.search(
            r"\bon\s+"
            r"(Monday|Tuesday|Wednesday|Thursday|"
            r"Friday|Saturday|Sunday)\b",
            text,
            flags=re.I,
        )

        if weekday_match:

            weekday_name = (
                weekday_match.group(1).lower()
            )

            if (
                anchor.weekday()
                == WEEKDAYS[weekday_name]
            ):
                return (
                    anchor.strftime("%Y-%m-%d"),
                    "same_weekday_assumption",
                    weekday_match.group(0),
                    "low",
                    "requires_review",
                )

    # -----------------------------------------
    # 6. Unresolved
    # -----------------------------------------

    return (
        "",
        "unknown_event_date",
        "",
        "none",
        "unresolved",
    )


def identify_temporal_context(text):
    """
    Adds descriptive flags.
    Does not invent an event occurrence date.
    """

    lower = normalize_text(text).lower()

    future = bool(
        re.search(
            r"\bwill\b|\bexpected to\b|"
            r"\bplan(?:ned|ning)? to\b|"
            r"\bgoing to\b|\bupcoming\b",
            lower,
        )
    )

    retrospective = bool(
        re.search(
            r"\bfresh videos\b|"
            r"\bremember\b|"
            r"\banniversary\b|"
            r"\bpreviously\b",
            lower,
        )
    )

    return future, retrospective


def create_records(df):

    required = {
        "cluster_id",
        "event_type",
        "location_candidate",
        "event_date_candidate",
        "reference_post_date",
        "date_basis",
        "representative_text",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    records = df.copy()

    records["state"] = (
        records["location_candidate"]
        .fillna("")
        .map(resolve_state)
    )

    records["resolved_event_date"] = ""
    records["resolved_date_basis"] = ""
    records["temporal_evidence"] = ""
    records["date_confidence"] = ""
    records["temporal_status"] = ""

    records["future_signal"] = False
    records["retrospective_signal"] = False

    # -----------------------------------------
    # Process each existing cluster
    # -----------------------------------------

    for idx, row in records.iterrows():

        original_date = date_string(
            row["event_date_candidate"]
        )

        text = row["representative_text"]
        post_date = row["reference_post_date"]

        if original_date:
            result = (
                original_date,
                str(row["date_basis"]),
                "existing_event_date_candidate",
                "medium",
                "date_resolved",
            )
        else:
            result = extract_temporal_evidence(
                text,
                post_date,
            )

        (
            resolved,
            basis,
            evidence,
            confidence,
            status,
        ) = result

        records.at[
            idx, "resolved_event_date"
        ] = resolved

        records.at[
            idx, "resolved_date_basis"
        ] = basis

        records.at[
            idx, "temporal_evidence"
        ] = evidence

        records.at[
            idx, "date_confidence"
        ] = confidence

        records.at[
            idx, "temporal_status"
        ] = status

        future, retrospective = (
            identify_temporal_context(text)
        )

        records.at[
            idx, "future_signal"
        ] = future

        records.at[
            idx, "retrospective_signal"
        ] = retrospective

    # -----------------------------------------
    # Detect repeated representative text
    # -----------------------------------------

    records["normalized_text"] = (
        records["representative_text"]
        .fillna("")
        .map(normalize_text)
        .str.lower()
        .str.replace(
            r"\s+", " ", regex=True
        )
        .str.strip()
    )

    records["duplicate_text_group_size"] = (
        records.groupby("normalized_text")[
            "cluster_id"
        ].transform("size")
    )

    records["duplicate_text_review"] = (
        records["duplicate_text_group_size"] > 1
    )

    # -----------------------------------------
    # Conservative aggregation eligibility
    # -----------------------------------------

    records["eligible_for_aggregation"] = (
        records["state"].fillna("").ne("")
        & records["resolved_event_date"].ne("")
        & records["temporal_status"].eq(
            "date_resolved"
        )
        & ~records["duplicate_text_review"]
    )

    # A resolved date is NOT proof that an
    # incident is verified.
    records["evidence_tier"] = (
        "provisional_unverified"
    )

    return records


def aggregate_records(records):

    eligible = records[
        records["eligible_for_aggregation"]
    ].copy()

    eligible = eligible.drop_duplicates(
        subset=["cluster_id"]
    )

    eligible["event_date_parsed"] = (
        pd.to_datetime(
            eligible["resolved_event_date"],
            errors="coerce",
        )
    )

    eligible = eligible.dropna(
        subset=["event_date_parsed"]
    )

    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "state",
                "week",
                "provisional_event_clusters",
                "distinct_candidate_types",
                "feature_status",
            ]
        )

    eligible["week"] = (
        eligible["event_date_parsed"]
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
            provisional_event_clusters=(
                "cluster_id",
                "nunique",
            ),
            distinct_candidate_types=(
                "event_type",
                "nunique",
            ),
        )
    )

    weekly["feature_status"] = (
        "provisional_unverified_not_ground_truth"
    )

    return weekly


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Source not found: {INPUT}"
        )

    df = pd.read_csv(
        INPUT,
        dtype={"cluster_id": "string"},
        low_memory=False,
    )

    records = create_records(df)
    weekly = aggregate_records(records)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records.to_csv(
        OUT / "temporal_event_records.csv",
        index=False,
    )

    weekly.to_csv(
        OUT / "provisional_state_week_features.csv",
        index=False,
    )

    review = records[
        ~records["eligible_for_aggregation"]
        | records["date_confidence"].isin(
            ["low", "medium"]
        )
    ].copy()

    review.to_csv(
        OUT / "manual_review_queue.csv",
        index=False,
    )

    audit = {
        "input_clusters": int(len(records)),
        "states_resolved": int(
            records["state"].ne("").sum()
        ),
        "dates_resolved": int(
            records[
                "resolved_event_date"
            ].ne("").sum()
        ),
        "aggregation_eligible": int(
            records[
                "eligible_for_aggregation"
            ].sum()
        ),
        "provisional_state_weeks": int(
            len(weekly)
        ),
        "manual_review_rows": int(
            len(review)
        ),
        "verified_incidents": 0,
        "date_basis_distribution": (
            records[
                "resolved_date_basis"
            ].value_counts().to_dict()
        ),
        "limitations": [
            "Date resolution does not verify occurrence.",
            "Publication dates are not automatically event dates.",
            "Duplicate clusters may describe the same event.",
            "State-week counts are provisional evidence.",
            "Not ready for forecasting without validation.",
        ],
    }

    (
        OUT / "temporal_audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print("\nOUTPUT DIRECTORY:", OUT)


if __name__ == "__main__":
    main()
