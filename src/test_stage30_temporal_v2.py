
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)

from stage30_temporal_v2 import (
    extract_temporal_evidence,
    create_records,
    aggregate_records,
)

import pandas as pd


def test_republic_day():

    result = extract_temporal_evidence(
        "Clashes unfolded on Republic Day",
        "2021-02-03",
    )

    assert result[0] == "2021-01-26"


def test_relative_yesterday():

    result = extract_temporal_evidence(
        "Police clashed with protesters yesterday",
        "2021-01-27",
    )

    assert result[0] == "2021-01-26"


def test_same_day():

    result = extract_temporal_evidence(
        "Protesters clashed with police this morning",
        "2021-01-26",
    )

    assert result[0] == "2021-01-26"


def test_unknown_date():

    result = extract_temporal_evidence(
        "Police lathi charge farmers in Karnal",
        "2021-08-28",
    )

    assert result[0] == ""


def test_weekday_requires_review():

    result = extract_temporal_evidence(
        "Farmers clashed with police on Tuesday",
        "2021-01-26",
    )

    assert result[0] == "2021-01-26"
    assert result[4] == "requires_review"


def test_historical_reference():

    result = extract_temporal_evidence(
        "Police share fresh videos of Red Fort clashes "
        "that unfolded on Republic Day",
        "2021-02-03",
    )

    assert result[0] == "2021-01-26"
    assert result[0] != "2021-02-03"


def test_duplicate_text_exclusion():

    text = (
        "Farmers organized a Tractor Rally "
        "near Kartarpur, Jalandhar."
    )

    df = pd.DataFrame([
        {
            "cluster_id": "A",
            "event_type": "rally",
            "location_candidate": "Punjab",
            "event_date_candidate": None,
            "reference_post_date": "2021-08-15",
            "date_basis": "unknown_event_date",
            "representative_text": text,
        },
        {
            "cluster_id": "B",
            "event_type": "protest",
            "location_candidate": "Delhi",
            "event_date_candidate": None,
            "reference_post_date": "2021-08-15",
            "date_basis": "unknown_event_date",
            "representative_text": text,
        },
    ])

    records = create_records(df)

    assert records[
        "duplicate_text_review"
    ].all()

    assert not records[
        "eligible_for_aggregation"
    ].any()

    weekly = aggregate_records(records)

    assert weekly.empty


if __name__ == "__main__":

    test_republic_day()
    test_relative_yesterday()
    test_same_day()
    test_unknown_date()
    test_weekday_requires_review()
    test_historical_reference()
    test_duplicate_text_exclusion()

    print("Stage 30 V2 tests passed")
