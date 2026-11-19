
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import os
import re
import unicodedata

import pandas as pd


# ============================================================
# STAGE 25 - CONSERVATIVE EVENT CONSOLIDATION
# CPU ONLY - NO MODEL TRAINING
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

STRUCTURED_DIR = ROOT / "data" / "structured"
OUTPUT_DIR = ROOT / "outputs" / "stage25"

INPUT_FILE = (
    STRUCTURED_DIR / "validated_event_evidence_v1.csv"
)

ENRICHED_FILE = (
    STRUCTURED_DIR / "enriched_event_candidates_v1.csv"
)

OUTPUT_MENTIONS = (
    OUTPUT_DIR / "consolidated_mentions_v1.csv"
)

OUTPUT_INCIDENTS = (
    OUTPUT_DIR / "provisional_incidents_v1.csv"
)

OUTPUT_UNRESOLVED = (
    OUTPUT_DIR / "unresolved_mentions_v1.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR / "consolidation_summary_v1.json"
)

# Grouping requires all three:
# event type + event location + event date.
#
# The same key is NOT sufficient proof that
# tweets describe one real-world incident.

REQUIRE_EXPLICIT_LOCATION = True

ALLOWED_TEMPORAL_STATUSES = {
    "event_near_date_candidate_unverified",
    "past_context_candidate_unverified"
}

EXCLUDED_OCCURRENCE_STATUSES = {
    "planned",
    "historical_reference",
    "negated",
    "uncertain"
}


# ============================================================
# BASIC HELPERS
# ============================================================

def get_value(row, columns):

    for column in columns:

        if column not in row.index:
            continue

        value = str(row[column]).strip()

        if value and value.lower() not in {
            "nan", "none", "null", "nat"
        }:
            return value

    return ""


def normalize_text(value):

    value = unicodedata.normalize(
        "NFKC", str(value)
    )

    value = value.lower().strip()

    value = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def normalize_location(value):

    value = normalize_text(value)

    value = re.sub(
        r"[^\w\s]",
        " ",
        value
    )

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def normalize_event_type(value):

    value = normalize_text(value)

    aliases = {
        "demonstration": "protest",
        "demonstrations": "protest",
        "protests": "protest",
        "marches": "march",
        "strikes": "strike",
        "riots": "riot",
        "clashes": "clash"
    }

    return aliases.get(value, value)


def normalize_date(value):

    if not value:
        return ""

    # Accept an actual calendar date only.
    # Never substitute the tweet posting date.

    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        value
    ):
        return ""

    try:
        return pd.Timestamp(value).strftime(
            "%Y-%m-%d"
        )

    except (ValueError, TypeError):
        return ""


def short_hash(value):

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]


def safe_save(df, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp = path.with_suffix(".tmp")

    df.to_csv(
        temp,
        index=False,
        encoding="utf-8-sig"
    )

    os.replace(temp, path)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input:\n{INPUT_FILE}"
        )

    print("\nLoading Stage 22 evidence...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        low_memory=False
    )

    print(f"Loaded {len(df):,} records")

    if "text" not in df.columns:
        raise ValueError(
            "Stage 22 must contain a text column."
        )

    if ENRICHED_FILE.exists():

        print("Checking Stage 20 enrichment...")

        enriched = pd.read_csv(
            ENRICHED_FILE,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            low_memory=False
        )

        # Avoid unsafe joins using tweet IDs:
        # older CSV processing may have changed
        # the precision of large numeric IDs.
        #
        # Use positional alignment ONLY when
        # the entire text sequence matches.

        same_order = (
            len(df) == len(enriched)
            and "text" in enriched.columns
            and df["text"].tolist()
            == enriched["text"].tolist()
        )

        if same_order:

            for column in [
                "event_type",
                "event_actors",
                "event_targets",
                "responding_entities",
                "scale_category",
                "escalation_indicators",
                "occurrence_status"
            ]:

                if column not in enriched.columns:
                    continue

                if column not in df.columns:
                    df[column] = enriched[column].values

                else:
                    missing = (
                        df[column].astype(str).str.strip()
                        == ""
                    )

                    df.loc[missing, column] = (
                        enriched.loc[missing, column].values
                    )

            print(
                "Stage 20 enrichment aligned "
                "and incorporated."
            )

        else:
            print(
                "WARNING: Stage 20 row alignment "
                "could not be verified."
            )
            print(
                "Continuing with Stage 22 only."
            )

    return df


# ============================================================
# PREPARE CONSOLIDATION EVIDENCE
# ============================================================

def prepare_mentions(df):

    records = []

    for index, row in df.iterrows():

        text = get_value(row, ["text"])

        event_type = normalize_event_type(
            get_value(
                row,
                ["event_type", "predicted_event_type"]
            )
        )

        location_status = get_value(
            row,
            ["location_validation_status"]
        )

        temporal_status = get_value(
            row,
            ["temporal_validation_status"]
        )

        occurrence_status = get_value(
            row,
            [
                "occurrence_status",
                "event_occurrence_status"
            ]
        ).lower()

        # Stage 22 evidence remains unverified.
        # Do not treat a NER place mention
        # as an event location.

        location = ""

        if (
            location_status
            == "explicit_relation_candidate_unverified"
        ):
            location = get_value(
                row,
                ["validated_event_location"]
            )

        event_date = ""

        if temporal_status in ALLOWED_TEMPORAL_STATUSES:

            event_date = normalize_date(
                get_value(
                    row,
                    ["validated_event_date"]
                )
            )

        normalized_location = normalize_location(
            location
        )

        # Conservative eligibility:
        # do not group planned, negated, historical
        # or uncertain occurrences.
        #
        # Missing occurrence evidence is allowed
        # only as an unresolved occurrence status;
        # it will be blocked from incident grouping.

        reasons = []

        if not event_type:
            reasons.append("missing_event_type")

        if not normalized_location:
            reasons.append(
                "missing_explicit_event_location"
            )

        if not event_date:
            reasons.append(
                "missing_supported_event_date"
            )

        if occurrence_status in (
            EXCLUDED_OCCURRENCE_STATUSES
        ):
            reasons.append(
                "excluded_occurrence_status"
            )

        if occurrence_status not in {
            "occurred",
            "ongoing"
        }:
            reasons.append(
                "occurrence_not_established"
            )

        eligible = len(reasons) == 0

        # Record ID is a stable local row reference.
        # It is NOT a reconstructed Twitter ID.

        record = {
            "stage25_row_id": index,
            "tweetId_original": get_value(
                row,
                ["tweetId"]
            ),
            "post_date": get_value(
                row,
                ["post_date", "date"]
            ),
            "text": text,
            "event_type": event_type,
            "event_location_candidate": location,
            "event_date_candidate": event_date,
            "location_validation_status": (
                location_status
            ),
            "temporal_validation_status": (
                temporal_status
            ),
            "occurrence_status": (
                occurrence_status
            ),
            "event_actors": get_value(
                row,
                ["event_actors", "actors"]
            ),
            "event_targets": get_value(
                row,
                ["event_targets", "targets"]
            ),
            "escalation_indicators": get_value(
                row,
                ["escalation_indicators"]
            ),
            "eligible_for_grouping": eligible,
            "unresolved_reasons": ";".join(
                dict.fromkeys(reasons)
            ),
            "provisional_group_id": "",
            "consolidation_status": ""
        }

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# CONSOLIDATE EXACT EVIDENCE MATCHES
# ============================================================

def consolidate(df):

    groups = defaultdict(list)

    # First: identify exact text duplicates.
    # This is a text-level operation, not proof
    # of unique real-world incidents.

    text_groups = defaultdict(list)

    for index, row in df.iterrows():

        normalized = normalize_text(
            row["text"]
        )

        if normalized:
            text_groups[
                short_hash(normalized)
            ].append(index)

    duplicate_text_rows = sum(
        len(indices) - 1
        for indices in text_groups.values()
        if len(indices) > 1
    )

    print(
        "Repeated normalized-text rows:",
        duplicate_text_rows
    )

    # Group only records with explicit evidence.

    for index, row in df.iterrows():

        if not row["eligible_for_grouping"]:

            df.at[
                index,
                "consolidation_status"
            ] = "unresolved"

            continue

        key = (
            row["event_type"],
            normalize_location(
                row["event_location_candidate"]
            ),
            row["event_date_candidate"]
        )

        groups[key].append(index)

    incident_records = []

    for key, indices in groups.items():

        event_type, location, date = key

        group_id = (
            "PG-"
            + short_hash("|".join(key))
        )

        # Important:
        # Matching event type, place, and date
        # produces a provisional group only.
        # Two separate incidents may still share
        # the same three attributes.

        for index in indices:

            df.at[
                index,
                "provisional_group_id"
            ] = group_id

            df.at[
                index,
                "consolidation_status"
            ] = "provisional_group_unverified"

        group_rows = df.loc[indices]

        unique_text_count = len({
            normalize_text(value)
            for value in group_rows["text"]
            if normalize_text(value)
        })

        incident_records.append({
            "provisional_group_id": group_id,
            "event_type": event_type,
            "event_location_candidate": location,
            "event_date_candidate": date,
            "mention_count": len(indices),
            "distinct_text_count": unique_text_count,
            "first_post_date": (
                group_rows["post_date"].min()
            ),
            "last_post_date": (
                group_rows["post_date"].max()
            ),
            "sample_text": (
                group_rows.iloc[0]["text"]
            ),
            "verification_status": (
                "unverified_provisional_group"
            ),
            "unique_incident_confirmed": False
        })

    incident_columns = [
        "provisional_group_id",
        "event_type",
        "event_location_candidate",
        "event_date_candidate",
        "mention_count",
        "distinct_text_count",
        "first_post_date",
        "last_post_date",
        "sample_text",
        "verification_status",
        "unique_incident_confirmed"
    ]

    incidents = pd.DataFrame(
        incident_records,
        columns=incident_columns
    )

    unresolved = df[
        df["consolidation_status"]
        == "unresolved"
    ].copy()

    return (
        df,
        incidents,
        unresolved,
        duplicate_text_rows
    )


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    mentions,
    incidents,
    unresolved,
    duplicate_text_rows
):

    summary = {
        "stage": 25,
        "method": (
            "Conservative exact event-type, "
            "location and date grouping"
        ),
        "total_candidate_mentions": int(
            len(mentions)
        ),
        "eligible_grouped_mentions": int(
            (
                mentions["consolidation_status"]
                == "provisional_group_unverified"
            ).sum()
        ),
        "provisional_groups": int(
            len(incidents)
        ),
        "unresolved_mentions": int(
            len(unresolved)
        ),
        "repeated_normalized_text_rows": int(
            duplicate_text_rows
        ),
        "confirmed_unique_incidents": 0,
        "unresolved_reason_counts": {},
        "limitations": [
            "Groups are not confirmed unique incidents",
            "Stage 22 locations and dates are unverified",
            "Tweet post dates are not event dates",
            "Missing event location/date prevents grouping",
            "Multiple distinct incidents may share a group key",
            "Tweet IDs may have lost precision upstream",
            "No independent incident-level ground truth"
        ]
    }

    reason_counts = defaultdict(int)

    for value in unresolved["unresolved_reasons"]:

        for reason in str(value).split(";"):

            if reason:
                reason_counts[reason] += 1

    summary["unresolved_reason_counts"] = dict(
        reason_counts
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 25 - EVENT CONSOLIDATION")
    print("=" * 70)

    source = load_data()

    print("\nPreparing event evidence...")

    mentions = prepare_mentions(source)

    print("\nConsolidating compatible mentions...")

    (
        mentions,
        incidents,
        unresolved,
        duplicate_text_rows
    ) = consolidate(mentions)

    print("\nSaving outputs...")

    safe_save(
        mentions,
        OUTPUT_MENTIONS
    )

    safe_save(
        incidents,
        OUTPUT_INCIDENTS
    )

    safe_save(
        unresolved,
        OUTPUT_UNRESOLVED
    )

    summary = create_summary(
        mentions,
        incidents,
        unresolved,
        duplicate_text_rows
    )

    print("\n" + "=" * 70)
    print("STAGE 25 - COMPLETED")
    print("=" * 70)

    print(
        "Candidate mentions:",
        summary["total_candidate_mentions"]
    )

    print(
        "Grouped mentions:",
        summary["eligible_grouped_mentions"]
    )

    print(
        "Provisional groups:",
        summary["provisional_groups"]
    )

    print(
        "Unresolved mentions:",
        summary["unresolved_mentions"]
    )

    print(
        "Confirmed unique incidents:",
        summary["confirmed_unique_incidents"]
    )

    print("\nOutput files:")

    print(OUTPUT_MENTIONS)
    print(OUTPUT_INCIDENTS)
    print(OUTPUT_UNRESOLVED)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
