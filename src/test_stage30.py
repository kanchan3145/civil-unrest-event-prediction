
import sys

from pathlib import Path

import pandas as pd


sys.path.insert(
    0,
    str(Path(__file__).parent),
)

from stage30_integrate import (
    integrate,
    resolve_state,
)


def test_state_resolution():

    assert (
        resolve_state("Karnal, Haryana")
        == "Haryana"
    )

    assert (
        resolve_state("New Delhi")
        == "Delhi"
    )

    assert (
        resolve_state("Ghaziabad, Uttar Pradesh")
        == "Uttar Pradesh"
    )

    assert resolve_state("Unknown") == ""

    assert (
        resolve_state("Downing Street, London")
        == ""
    )


def test_no_post_date_leakage_and_dedup():

    df = pd.DataFrame(
        [
            {
                "cluster_id": "a",
                "event_type": "protest",
                "location_candidate": "Delhi, India",
                "event_date_candidate": "2021-01-26",
                "verification_status": "unverified_rule_based",
            },
            {
                "cluster_id": "a",
                "event_type": "protest",
                "location_candidate": "Delhi, India",
                "event_date_candidate": "2021-01-26",
                "verification_status": "unverified_rule_based",
            },
            {
                "cluster_id": "b",
                "event_type": "clash",
                "location_candidate": "Delhi, India",
                "event_date_candidate": "",
                "verification_status": "unverified_rule_based",
            },
        ]
    )

    records, weekly = integrate(df)

    assert len(weekly) == 1

    assert (
        int(
            weekly.iloc[0].provisional_event_clusters
        )
        == 1
    )

    assert (
        records[
            "eligible_for_provisional_aggregation"
        ].sum()
        == 2
    )


if __name__ == "__main__":

    test_state_resolution()

    test_no_post_date_leakage_and_dedup()

    print("Stage 30 tests passed")
