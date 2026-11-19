
from pathlib import Path
import json
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "consolidated"

VERSION = "v5"
INPUT = DATA / f"event_mentions_context_{VERSION}_v2.csv"

OUTPUT_EVENTS = DATA / "incident_clusters_v5_v3.csv"
OUTPUT_MENTIONS = DATA / "event_mentions_consolidated_v5_v3.csv"
OUTPUT_REVIEW = DATA / "incident_review_v5_v3.csv"
OUTPUT_SUMMARY = DATA / "incident_summary_v5_v3.json"

SIMILARITY_THRESHOLD = 0.35

def value(x):
    return "" if pd.isna(x) else str(x).strip()

def normalize_text(text):
    text = value(text).lower()
    text = re.sub(r"https?://\S+|\[user\]|\[url\]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def location_key(location):
    loc = value(location).lower()

    # Keep distinct incident sites separate.
    if "red fort" in loc:
        return "red_fort"
    if "ito" in loc:
        return "ito"
    if "singhu" in loc:
        return "singhu"
    if "ghazipur" in loc:
        return "ghazipur"
    if "tikri" in loc:
        return "tikri"
    if "gharonda" in loc:
        return "gharonda"
    if "karnal" in loc:
        return "karnal"
    if "kartarpur" in loc:
        return "kartarpur"
    if "delhi" in loc:
        return "delhi_unspecified"

    return loc

def resolve_date(row):
    existing = pd.to_datetime(
        value(row.get("resolved_event_date")),
        errors="coerce"
    )

    if pd.notna(existing):
        return str(existing.date()), "resolved_event_date"

    text = (
        value(row.get("event_evidence")) + " " +
        value(row.get("text"))
    ).lower()

    post_date = pd.to_datetime(
        row.get("reference_post_date"),
        errors="coerce"
    )

    # Historical Republic Day reports require contextual handling.
    if (
        "republic day" in text
        and pd.notna(post_date)
        and post_date.year == 2021
        and (
            "clash" in text
            or "violence" in text
            or "red fort" in text
        )
    ):
        return "2021-01-26", "republic_day_context_inferred"

    # Publication date is not a confirmed event date.
    return "", "unknown_event_date"

def is_review_only(row):
    text = (
        value(row.get("event_evidence")) + " " +
        value(row.get("text"))
    ).lower()

    # Legal proceedings are not themselves civil unrest incidents.
    if re.search(
        r"\b(?:bail granted|court granted bail|"
        r"hearing in court|released on bail)\b",
        text
    ):
        return "legal_proceeding_review"

    status = value(row.get("resolved_occurrence_status"))

    if status in (
        "planned_or_future_review",
        "historical_review"
    ):
        return status

    return ""

def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    outputs = [
        OUTPUT_EVENTS,
        OUTPUT_MENTIONS,
        OUTPUT_REVIEW,
        OUTPUT_SUMMARY
    ]

    for path in outputs:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    df = pd.read_csv(
        INPUT,
        dtype={"post_id": "string"},
        low_memory=False
    )

    required = [
        "post_id",
        "text",
        "event_type",
        "resolved_event_location",
        "reference_post_date"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["incident_id_v3"] = ""
    df["incident_status_v3"] = "needs_review"
    df["review_reason_v3"] = ""
    df["location_key_v3"] = df[
        "resolved_event_location"
    ].map(location_key)

    dates = df.apply(resolve_date, axis=1)
    df["event_date_v3"] = dates.map(lambda x: x[0])
    df["event_date_basis_v3"] = dates.map(lambda x: x[1])

    df["post_day_v3"] = pd.to_datetime(
        df["reference_post_date"],
        errors="coerce",
        utc=True
    ).dt.strftime("%Y-%m-%d")

    df["normalized_text_v3"] = df.apply(
        lambda row: normalize_text(
            value(row.get("event_evidence"))
            or value(row.get("text"))
        ),
        axis=1
    )

    for i, row in df.iterrows():
        reason = is_review_only(row)

        if reason:
            df.at[i, "review_reason_v3"] = reason
        elif not value(row["location_key_v3"]):
            df.at[i, "review_reason_v3"] = "missing_location"
        elif not value(row["event_date_v3"]):
            df.at[i, "review_reason_v3"] = "unknown_event_date"

    # Group by event type, specific location and inferred/known
    # event date. Do not automatically merge broad Delhi locations
    # with specific sites such as Red Fort or ITO.
    eligible = df.index[
        df["review_reason_v3"].eq("")
    ].tolist()

    blocks = {}

    for i in eligible:
        row = df.loc[i]

        key = (
            value(row["event_type"]).lower(),
            row["location_key_v3"],
            row["event_date_v3"]
        )

        blocks.setdefault(key, []).append(i)

    groups = []

    for indices in blocks.values():
        if len(indices) == 1:
            groups.append(indices)
            continue

        texts = [
            df.at[i, "normalized_text_v3"]
            for i in indices
        ]

        matrix = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        ).fit_transform(texts)

        similarity = cosine_similarity(matrix)

        unused = set(range(len(indices)))

        while unused:
            seed = min(unused)
            members = [seed]
            unused.remove(seed)

            for j in sorted(unused):
                if similarity[seed, j] >= SIMILARITY_THRESHOLD:
                    members.append(j)

            unused.difference_update(members)
            groups.append([indices[j] for j in members])

    records = []

    for number, indices in enumerate(
        sorted(groups, key=lambda x: min(x)),
        start=1
    ):
        incident_id = f"IND-V5-{number:04d}"
        subset = df.loc[indices]

        post_ids = sorted(
            set(subset["post_id"].dropna().astype(str))
        )

        status = (
            "provisional_multi_post"
            if len(post_ids) > 1
            else "provisional_singleton"
        )

        for i in indices:
            df.at[i, "incident_id_v3"] = incident_id
            df.at[i, "incident_status_v3"] = status

        records.append({
            "incident_id": incident_id,
            "event_type": value(subset.iloc[0]["event_type"]),
            "event_location": value(
                subset.iloc[0]["resolved_event_location"]
            ),
            "event_date": value(
                subset.iloc[0]["event_date_v3"]
            ),
            "date_basis": value(
                subset.iloc[0]["event_date_basis_v3"]
            ),
            "mention_count": len(indices),
            "supporting_post_count": len(post_ids),
            "supporting_post_ids": json.dumps(post_ids),
            "representative_text": value(
                subset.iloc[0]["text"]
            )[:500],
            "status": status,
            "verification_status": "unverified"
        })

    event_columns = [
        "incident_id",
        "event_type",
        "event_location",
        "event_date",
        "date_basis",
        "mention_count",
        "supporting_post_count",
        "supporting_post_ids",
        "representative_text",
        "status",
        "verification_status"
    ]

    events = pd.DataFrame(records, columns=event_columns)

    review = df[
        df["incident_status_v3"].eq("needs_review")
    ].copy()

    df.to_csv(OUTPUT_MENTIONS, index=False)
    events.to_csv(OUTPUT_EVENTS, index=False)
    review.to_csv(OUTPUT_REVIEW, index=False)

    summary = {
        "input_mentions": len(df),
        "provisional_incidents": len(events),
        "multi_post_incidents": int(
            (events["supporting_post_count"] > 1).sum()
        ),
        "single_post_incidents": int(
            (events["supporting_post_count"] == 1).sum()
        ),
        "unresolved_mentions": len(review),
        "note": (
            "Conservative event-date clustering. "
            "Unknown dates remain unresolved. "
            "Clusters are not independently verified."
        )
    }

    OUTPUT_SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\nSTAGE 12.2 COMPLETE")
    print(json.dumps(summary, indent=2))
    print("\nOutputs saved in:", DATA)

if __name__ == "__main__":
    main()
