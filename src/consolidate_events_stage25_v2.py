
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import os
import re
import unicodedata

import pandas as pd


# ============================================================
# STAGE 25 V2
# DUPLICATE MENTION DETECTION + PROVISIONAL INCIDENT GROUPING
#
# No training, no model downloads, CPU only.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "structured"
OUTPUT_DIR = ROOT / "outputs" / "stage25_v2"

INPUT_FILE = DATA_DIR / "validated_event_evidence_v1.csv"
ENRICHED_FILE = DATA_DIR / "enriched_event_candidates_v1.csv"

MENTIONS_FILE = OUTPUT_DIR / "all_mentions_v2.csv"
DUPLICATES_FILE = OUTPUT_DIR / "duplicate_mention_groups_v2.csv"
INCIDENTS_FILE = OUTPUT_DIR / "provisional_incident_groups_v2.csv"
UNRESOLVED_FILE = OUTPUT_DIR / "unresolved_mentions_v2.csv"
SUMMARY_FILE = OUTPUT_DIR / "stage25_v2_summary.json"


# Only these Stage 22 statuses support a provisional
# event-location relation.
LOCATION_STATUSES = {
    "explicit_relation_candidate_unverified"
}

# These temporal statuses support a provisional
# event-date candidate.
DATE_STATUSES = {
    "event_near_date_candidate_unverified",
    "past_context_candidate_unverified"
}

# Never treat these as current occurred incidents.
EXCLUDED_OCCURRENCES = {
    "planned",
    "planned_or_future",
    "future",
    "historical",
    "historical_reference",
    "negated",
    "not_event",
    "non_event"
}

SUPPORTED_OCCURRENCES = {
    "occurred",
    "ongoing",
    "reported_occurrence"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def value(row, names):
    """Read the first nonempty matching column."""

    for name in names:
        if name not in row.index:
            continue

        result = str(row[name]).strip()

        if result.lower() not in {
            "", "nan", "none", "null", "nat"
        }:
            return result

    return ""


def normalize_text(text):

    text = unicodedata.normalize(
        "NFKC", str(text)
    ).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_location(text):

    text = normalize_text(text)

    text = re.sub(r"[^\w\s]", " ", text)

    return re.sub(
        r"\s+", " ", text
    ).strip()


def normalize_type(text):

    text = normalize_text(text)

    aliases = {
        "protests": "protest",
        "demonstration": "protest",
        "demonstrations": "protest",
        "marches": "march",
        "strikes": "strike",
        "riots": "riot",
        "clashes": "clash"
    }

    return aliases.get(text, text)


def normalize_date(text):
    """
    Only accept YYYY-MM-DD.
    Never substitute the tweet posting date.
    """

    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        text
    ):
        return ""

    try:
        return pd.Timestamp(text).strftime(
            "%Y-%m-%d"
        )

    except (ValueError, TypeError):
        return ""


def make_hash(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:20]


def save_csv(df, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = path.with_suffix(".tmp")

    df.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig"
    )

    os.replace(temporary, path)


# ============================================================
# LOAD PREVIOUS STAGES
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing Stage 22 file: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        low_memory=False
    ).reset_index(drop=True)

    if "text" not in df.columns:
        raise ValueError(
            "Stage 22 file has no text column."
        )

    print(f"Loaded Stage 22: {len(df):,} rows")

    if ENRICHED_FILE.exists():

        enriched = pd.read_csv(
            ENRICHED_FILE,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            low_memory=False
        ).reset_index(drop=True)

        # Do not join on potentially rounded tweet IDs.
        # Only copy fields when complete row alignment
        # has been verified by comparing tweet text.

        aligned = (
            len(df) == len(enriched)
            and "text" in enriched.columns
            and df["text"].equals(enriched["text"])
        )

        if aligned:

            columns = [
                "event_type",
                "event_actors",
                "event_targets",
                "responding_entities",
                "occurrence_status",
                "event_occurrence_status",
                "escalation_indicators"
            ]

            for col in columns:

                if col not in enriched.columns:
                    continue

                if col not in df.columns:
                    df[col] = enriched[col]

                else:
                    empty = df[col].eq("")

                    df.loc[empty, col] = (
                        enriched.loc[empty, col]
                    )

            print("Stage 20 enrichment aligned.")

        else:
            print(
                "WARNING: Stage 20 alignment not "
                "verified; using Stage 22 only."
            )

    return df


# ============================================================
# BUILD MENTION RECORDS
# ============================================================

def prepare_mentions(source):

    records = []

    for index, row in source.iterrows():

        text = value(row, ["text"])

        normalized = normalize_text(text)

        event_type = normalize_type(
            value(
                row,
                ["event_type", "predicted_event_type"]
            )
        )

        location_status = value(
            row,
            ["location_validation_status"]
        )

        temporal_status = value(
            row,
            ["temporal_validation_status"]
        )

        occurrence = value(
            row,
            [
                "occurrence_status",
                "event_occurrence_status"
            ]
        ).lower()

        # A missing occurrence label is NOT
        # proof that the event occurred.
        if occurrence in SUPPORTED_OCCURRENCES:
            occurrence_evidence = "supported_label"

        elif occurrence in EXCLUDED_OCCURRENCES:
            occurrence_evidence = "excluded"

        else:
            occurrence_evidence = "unverified"

        location = ""

        if location_status in LOCATION_STATUSES:
            location = value(
                row,
                ["validated_event_location"]
            )

        event_date = ""

        if temporal_status in DATE_STATUSES:
            event_date = normalize_date(
                value(
                    row,
                    ["validated_event_date"]
                )
            )

        reasons = []

        if not text:
            reasons.append("missing_text")

        if not event_type:
            reasons.append("missing_event_type")

        if not location:
            reasons.append("missing_event_location")

        if not event_date:
            reasons.append("missing_event_date")

        if occurrence_evidence == "excluded":
            reasons.append("excluded_occurrence")

        # Missing occurrence evidence does not
        # block a provisional compatibility group.
        # Such groups remain explicitly unverified.

        eligible = (
            len(reasons) == 0
        )

        records.append({
            "stage25_row_id": index,
            "tweetId_original": value(
                row, ["tweetId"]
            ),
            "post_date": value(
                row, ["post_date", "date"]
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
            "occurrence_status": occurrence,
            "occurrence_evidence": (
                occurrence_evidence
            ),
            "event_actors": value(
                row, ["event_actors", "actors"]
            ),
            "event_targets": value(
                row, ["event_targets", "targets"]
            ),
            "escalation_indicators": value(
                row, ["escalation_indicators"]
            ),
            "normalized_text": normalized,
            "eligible_for_provisional_group": (
                eligible
            ),
            "unresolved_reasons": ";".join(
                reasons
            ),
            "duplicate_group_id": "",
            "duplicate_group_size": 0,
            "provisional_group_id": "",
            "consolidation_status": ""
        })

    return pd.DataFrame(records)


# ============================================================
# PART A - DUPLICATE TWEET CONTENT
# ============================================================

def detect_duplicates(df):

    print("\nDetecting repeated tweet content...")

    counts = Counter(
        text
        for text in df["normalized_text"]
        if text
    )

    duplicate_records = []

    for index, row in df.iterrows():

        normalized = row["normalized_text"]

        if not normalized:
            continue

        count = counts[normalized]

        if count <= 1:
            continue

        group_id = (
            "DUP-" + make_hash(normalized)
        )

        df.at[
            index,
            "duplicate_group_id"
        ] = group_id

        df.at[
            index,
            "duplicate_group_size"
        ] = count

    # One summary row per repeated text.
    grouped = defaultdict(list)

    for index, row in df.iterrows():

        group_id = row["duplicate_group_id"]

        if group_id:
            grouped[group_id].append(index)

    for group_id, indices in grouped.items():

        subset = df.loc[indices]

        duplicate_records.append({
            "duplicate_group_id": group_id,
            "mention_count": len(indices),
            "extra_repeated_rows": (
                len(indices) - 1
            ),
            "sample_text": subset.iloc[0]["text"],
            "first_post_date": (
                subset["post_date"].min()
            ),
            "last_post_date": (
                subset["post_date"].max()
            ),
            "verification_status": (
                "exact_normalized_text_match"
            ),
            "unique_incident_confirmed": False
        })

    columns = [
        "duplicate_group_id",
        "mention_count",
        "extra_repeated_rows",
        "sample_text",
        "first_post_date",
        "last_post_date",
        "verification_status",
        "unique_incident_confirmed"
    ]

    result = pd.DataFrame(
        duplicate_records,
        columns=columns
    )

    return df, result


# ============================================================
# PART B - PROVISIONAL INCIDENT GROUPING
# ============================================================

def group_incidents(df):

    print("\nGrouping compatible event evidence...")

    groups = defaultdict(list)

    for index, row in df.iterrows():

        if not row["eligible_for_provisional_group"]:
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

        event_type, location, event_date = key

        group_id = (
            "PG-" + make_hash("|".join(key))
        )

        subset = df.loc[indices]

        occurrence_values = set(
            subset["occurrence_evidence"]
        )

        if occurrence_values == {"supported_label"}:
            occurrence_group_status = (
                "occurrence_label_supported"
            )
        else:
            occurrence_group_status = (
                "occurrence_unverified"
            )

        for index in indices:

            df.at[
                index,
                "provisional_group_id"
            ] = group_id

            df.at[
                index,
                "consolidation_status"
            ] = "provisional_group_unverified"

        distinct_texts = len(
            set(subset["normalized_text"])
        )

        incident_records.append({
            "provisional_group_id": group_id,
            "event_type": event_type,
            "event_location_candidate": location,
            "event_date_candidate": event_date,
            "mention_count": len(indices),
            "distinct_text_count": distinct_texts,
            "occurrence_group_status": (
                occurrence_group_status
            ),
            "first_post_date": (
                subset["post_date"].min()
            ),
            "last_post_date": (
                subset["post_date"].max()
            ),
            "sample_text": subset.iloc[0]["text"],
            "verification_status": (
                "unverified_provisional_group"
            ),
            "unique_incident_confirmed": False
        })

    columns = [
        "provisional_group_id",
        "event_type",
        "event_location_candidate",
        "event_date_candidate",
        "mention_count",
        "distinct_text_count",
        "occurrence_group_status",
        "first_post_date",
        "last_post_date",
        "sample_text",
        "verification_status",
        "unique_incident_confirmed"
    ]

    incidents = pd.DataFrame(
        incident_records,
        columns=columns
    )

    unresolved = df[
        df["consolidation_status"] == "unresolved"
    ].copy()

    return df, incidents, unresolved


# ============================================================
# SAVE SUMMARY
# ============================================================

def create_summary(
    mentions,
    duplicates,
    incidents,
    unresolved
):

    reason_counts = Counter()

    for reasons in unresolved["unresolved_reasons"]:

        for reason in reasons.split(";"):

            if reason:
                reason_counts[reason] += 1

    repeated_rows = (
        int(duplicates["extra_repeated_rows"].sum())
        if len(duplicates)
        else 0
    )

    summary = {
        "stage": "25_v2",
        "total_candidate_mentions": int(
            len(mentions)
        ),
        "duplicate_text_groups": int(
            len(duplicates)
        ),
        "extra_repeated_text_rows": (
            repeated_rows
        ),
        "provisional_incident_groups": int(
            len(incidents)
        ),
        "provisionally_grouped_mentions": int(
            (
                mentions["consolidation_status"]
                == "provisional_group_unverified"
            ).sum()
        ),
        "unresolved_mentions": int(
            len(unresolved)
        ),
        "confirmed_unique_incidents": 0,
        "unresolved_reason_counts": dict(
            reason_counts
        ),
        "limitations": [
            "Exact text duplicates are not necessarily duplicate real-world incidents",
            "Provisional groups are not confirmed incidents",
            "Stage 22 locations and dates are unverified",
            "Missing occurrence evidence is not proof of occurrence",
            "Multiple incidents can share the same type, location and date",
            "Tweet post date is not used as event date",
            "No independent incident-level accuracy estimate"
        ]
    }

    SUMMARY_FILE.write_text(
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
    print("STAGE 25 V2 - CORRECTED EVENT CONSOLIDATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    source = load_data()

    print("\nPreparing event mentions...")
    mentions = prepare_mentions(source)

    mentions, duplicates = detect_duplicates(
        mentions
    )

    (
        mentions,
        incidents,
        unresolved
    ) = group_incidents(mentions)

    print("\nSaving output files...")

    # normalized_text is internal working data.
    # Keep output readable.
    saved_mentions = mentions.drop(
        columns=["normalized_text"]
    )

    saved_unresolved = unresolved.drop(
        columns=["normalized_text"]
    )

    save_csv(
        saved_mentions,
        MENTIONS_FILE
    )

    save_csv(
        duplicates,
        DUPLICATES_FILE
    )

    save_csv(
        incidents,
        INCIDENTS_FILE
    )

    save_csv(
        saved_unresolved,
        UNRESOLVED_FILE
    )

    summary = create_summary(
        mentions,
        duplicates,
        incidents,
        unresolved
    )

    print("\n" + "=" * 70)
    print("STAGE 25 V2 - COMPLETED")
    print("=" * 70)

    print(
        "Total candidate mentions:",
        summary["total_candidate_mentions"]
    )

    print(
        "Duplicate text groups:",
        summary["duplicate_text_groups"]
    )

    print(
        "Extra repeated text rows:",
        summary["extra_repeated_text_rows"]
    )

    print(
        "Provisional incident groups:",
        summary["provisional_incident_groups"]
    )

    print(
        "Provisionally grouped mentions:",
        summary["provisionally_grouped_mentions"]
    )

    print(
        "Unresolved mentions:",
        summary["unresolved_mentions"]
    )

    print(
        "Confirmed unique incidents:",
        summary["confirmed_unique_incidents"]
    )

    print("\nUnresolved reasons:")

    for reason, count in (
        summary["unresolved_reason_counts"].items()
    ):
        print(f"  {reason}: {count}")

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
