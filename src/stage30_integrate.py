
"""Stage 30: conservative structured-event feature integration.

Run from project root:
    python src/stage30_integrate.py

Uses existing Stage 11 clusters.
Does not train or alter previous outputs.

Outputs are PROVISIONAL, not verified incidents.
"""

from pathlib import Path
import json
import re
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT
    / "data/consolidated/incident_clusters_v5_v2.csv"
)

OUT = ROOT / "outputs/stage30"


STATE_NAMES = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Delhi",
    "Chandigarh",
    "Jammu and Kashmir",
    "Ladakh",
    "Puducherry",
    "Lakshadweep",
    "Andaman and Nicobar Islands",
    "Dadra and Nagar Haveli and Daman and Diu",
]


CITY_STATES = {
    "ghaziabad": "Uttar Pradesh",
    "noida": "Uttar Pradesh",
    "lakhimpur": "Uttar Pradesh",
    "muzaffarnagar": "Uttar Pradesh",
    "karnal": "Haryana",
    "gharonda": "Haryana",
    "jalandhar": "Punjab",
    "kartarpur": "Punjab",
    "mumbai": "Maharashtra",
    "pune": "Maharashtra",
    "jaipur": "Rajasthan",
    "bengaluru": "Karnataka",
    "bangalore": "Karnataka",
    "kolkata": "West Bengal",
    "patna": "Bihar",
    "chennai": "Tamil Nadu",
    "hyderabad": "Telangana",
    "bhopal": "Madhya Pradesh",
    "ahmedabad": "Gujarat",
    "singhu": "Delhi",
    "tikri": "Delhi",
    "ghazipur border": "Delhi",
    "red fort": "Delhi",
    "ito": "Delhi",
}


def resolve_state(location):
    """
    Resolve a candidate location to an Indian state
    or union territory.

    Returns an empty string when unresolved.
    """

    s = str(location or "").lower().strip()

    if not s or s in ("nan", "unknown", "none"):
        return ""

    for state in sorted(
        STATE_NAMES,
        key=len,
        reverse=True,
    ):
        pattern = (
            r"(?<!\w)"
            + re.escape(state.lower())
            + r"(?!\w)"
        )

        if re.search(pattern, s):
            return state

    for city, state in sorted(
        CITY_STATES.items(),
        key=lambda kv: -len(kv[0]),
    ):
        pattern = (
            r"(?<!\w)"
            + re.escape(city)
            + r"(?!\w)"
        )

        if re.search(pattern, s):
            return state

    return ""


def integrate(df):
    """
    Convert provisional incident clusters into
    state-week features.

    Only clusters with an identifiable state and
    occurrence date are eligible for aggregation.
    """

    needed = {
        "cluster_id",
        "event_type",
        "location_candidate",
        "event_date_candidate",
        "verification_status",
    }

    absent = needed - set(df.columns)

    if absent:
        raise ValueError(
            "Missing required cluster columns: "
            f"{sorted(absent)}"
        )

    x = df.copy()

    # ----------------------------------------
    # Resolve candidate event locations
    # ----------------------------------------

    x["state"] = (
        x["location_candidate"]
        .fillna("")
        .map(resolve_state)
    )

    # ----------------------------------------
    # Parse event occurrence dates
    # ----------------------------------------

    x["event_date"] = pd.to_datetime(
        x["event_date_candidate"],
        errors="coerce",
        utc=True,
    )

    # IMPORTANT:
    # Use the event date, not the post date.
    # A post date is not necessarily the date
    # when the described event occurred.

    x["week"] = (
        x["event_date"]
        .dt.tz_localize(None)
        .dt.to_period("W-SUN")
        .dt.start_time
    )

    # ----------------------------------------
    # Check aggregation eligibility
    # ----------------------------------------

    x["eligible_for_provisional_aggregation"] = (
        x["state"].ne("")
        & x["event_date"].notna()
        & x["cluster_id"].notna()
    )

    # ----------------------------------------
    # Label evidence as unverified
    # ----------------------------------------

    x["evidence_tier"] = "unverified_candidate"

    # ----------------------------------------
    # Select eligible records
    # ----------------------------------------

    eligible = x[
        x["eligible_for_provisional_aggregation"]
    ].copy()

    # Avoid counting the same cluster repeatedly.

    eligible = eligible.drop_duplicates(
        "cluster_id"
    )

    # ----------------------------------------
    # Aggregate by state and week
    # ----------------------------------------

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

    weekly["week"] = (
        weekly["week"]
        .dt.strftime("%Y-%m-%d")
    )

    weekly["feature_status"] = (
        "provisional_unverified_not_ground_truth"
    )

    return x, weekly


def main():

    # ----------------------------------------
    # Validate input file
    # ----------------------------------------

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. "
            "First run existing "
            "src/resolve_and_consolidate_v2.py, "
            "or check your cluster filename."
        )

    # ----------------------------------------
    # Load incident clusters
    # ----------------------------------------

    df = pd.read_csv(
        INPUT,
        dtype={"cluster_id": "string"},
        low_memory=False,
    )

    # ----------------------------------------
    # Integrate structured event records
    # ----------------------------------------

    records, weekly = integrate(df)

    # ----------------------------------------
    # Create output directory
    # ----------------------------------------

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------
    # Save candidate event records
    # ----------------------------------------

    records.to_csv(
        OUT / "candidate_event_records.csv",
        index=False,
    )

    # ----------------------------------------
    # Save state-week features
    # ----------------------------------------

    weekly.to_csv(
        OUT / "provisional_state_week_features.csv",
        index=False,
    )

    # ----------------------------------------
    # Generate audit report
    # ----------------------------------------

    report = {
        "source": str(INPUT.relative_to(ROOT)),

        "candidate_clusters": len(records),

        "state_and_event_date_resolved": int(
            records[
                "eligible_for_provisional_aggregation"
            ].sum()
        ),

        "provisional_state_weeks": len(weekly),

        "verified_incidents": 0,

        "warning": (
            "Do not treat these features as "
            "verified incidents or proven "
            "predictive features. "
            "A chronological ablation and "
            "manual validation are still required."
        ),
    }

    # ----------------------------------------
    # Save audit report
    # ----------------------------------------

    (
        OUT / "integration_audit.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ----------------------------------------
    # Print results
    # ----------------------------------------

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
