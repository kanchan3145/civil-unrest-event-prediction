
"""
STAGE 33 - FINAL PROJECT DASHBOARD

Run from project root:
    python -m streamlit run src/stage33_dashboard.py

Reads existing Stage 31 and Stage 32 outputs.
Does not retrain models or modify prior results.
"""

from pathlib import Path
import json
import re

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
STAGE31 = ROOT / "outputs" / "stage31_final"
STAGE32 = ROOT / "outputs" / "stage32"

st.set_page_config(
    page_title="Civil Unrest Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(
        path,
        keep_default_na=False,
        low_memory=False,
    )


@st.cache_data
def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def missing_values(series):
    return series.astype(str).str.strip().str.lower().isin(
        ["", "unknown", "unclear", "none", "nan", "null", "n/a"]
    )


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, val) in zip(cols, items):
        col.metric(label, val)


def show_empty(message):
    st.info(message)


events = load_csv(STAGE32 / "event_dashboard_data.csv")
coverage = load_csv(STAGE32 / "extraction_coverage.csv")
types = load_csv(STAGE32 / "event_type_distribution.csv")
issues = load_csv(STAGE32 / "quality_issues.csv")
duplicates = load_csv(STAGE32 / "duplicate_groups.csv")
weekly = load_csv(
    STAGE32 / "exploratory_state_week_features.csv"
)
summary = load_json(STAGE32 / "evaluation_summary.json")

if events.empty:
    st.error(
        "Stage 32 dashboard data was not found. "
        "Run python src/stage32_evaluate_integrate.py first."
    )
    st.stop()


st.title("Civil Unrest Event Extraction and Prediction")
st.caption(
    "Social-media event intelligence | "
    "Automated candidate extraction | "
    "Chronological forecasting evaluation"
)

st.warning(
    "Event records shown here are automatically extracted, "
    "unverified candidates. Candidate counts must not be "
    "interpreted as independently confirmed incident counts."
)

section = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Structured Event Explorer",
        "Extraction Quality",
        "Forecasting Evaluation",
        "Data and Methodology",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Stage 33 | Read-only project dashboard")


if section == "Project Overview":
    st.header("Project Overview")

    metric_row([
        ("Extracted clusters", len(events)),
        (
            "Unique text candidates",
            summary.get("unique_text_candidates", "N/A"),
        ),
        (
            "Resolved occurrence dates",
            summary.get("resolved_event_dates", "N/A"),
        ),
        ("Verified incidents", 0),
    ])

    st.subheader("Automatically assigned event types")

    if not types.empty:
        chart = types.set_index("event_type")["count"]
        st.bar_chart(chart)
    else:
        show_empty("Event type distribution unavailable.")

    st.subheader("Occurrence status")

    if "occurrence_status" in events.columns:
        status_counts = (
            events["occurrence_status"]
            .value_counts()
            .rename_axis("status")
            .reset_index(name="count")
        )
        st.dataframe(
            status_counts,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Extraction overview")

    if not coverage.empty:
        st.bar_chart(
            coverage.set_index("field")["coverage_percent"]
        )
        st.caption(
            "Field coverage measures whether an annotation "
            "is available, not whether it is correct."
        )


elif section == "Structured Event Explorer":
    st.header("Structured Event Explorer")

    left, middle, right = st.columns(3)

    with left:
        selected_types = st.multiselect(
            "Event type",
            sorted(events["event_type"].unique()),
            default=sorted(events["event_type"].unique()),
        )

    with middle:
        selected_states = st.multiselect(
            "State",
            sorted(events["state"].unique()),
            default=sorted(events["state"].unique()),
        )

    with right:
        selected_statuses = st.multiselect(
            "Occurrence status",
            sorted(events["occurrence_status"].unique()),
            default=sorted(
                events["occurrence_status"].unique()
            ),
        )

    filtered = events[
        events["event_type"].isin(selected_types)
        & events["state"].isin(selected_states)
        & events["occurrence_status"].isin(
            selected_statuses
        )
    ].copy()

    search = st.text_input(
        "Search representative tweet text",
        placeholder="Example: Karnal, farmers, clash",
    )

    if search.strip():
        filtered = filtered[
            filtered["representative_text"].str.contains(
                re.escape(search.strip()),
                case=False,
                na=False,
                regex=True,
            )
        ]

    st.metric("Matching records", len(filtered))

    preferred = [
        "cluster_id",
        "event_type",
        "state",
        "location",
        "event_date",
        "occurrence_status",
        "actors",
        "targets",
        "scale",
        "escalation",
        "exact_text_duplicate",
        "quality_flags",
    ]

    available = [
        c for c in preferred if c in filtered.columns
    ]

    st.dataframe(
        filtered[available],
        use_container_width=True,
        hide_index=True,
    )

    if not filtered.empty:
        selected_id = st.selectbox(
            "Inspect a source record",
            filtered["cluster_id"].tolist(),
        )

        record = filtered[
            filtered["cluster_id"] == selected_id
        ].iloc[0]

        st.subheader("Source evidence")
        st.write(record["representative_text"])

        st.caption(
            "This text is source evidence for an automatic "
            "candidate annotation, not independent verification."
        )

        st.json({
            "cluster_id": str(record["cluster_id"]),
            "event_type": str(record["event_type"]),
            "state": str(record["state"]),
            "event_date": str(record["event_date"])
            or "Unknown",
            "date_basis": str(record["date_basis"]),
            "occurrence_status": str(
                record["occurrence_status"]
            ),
            "quality_flags": str(record["quality_flags"]),
            "verification_status": str(
                record["verification_status"]
            ),
        })

    st.download_button(
        "Download filtered event candidates",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_event_candidates.csv",
        mime="text/csv",
    )


elif section == "Extraction Quality":
    st.header("Extraction Quality and Uncertainty")

    st.subheader("Field availability")

    st.dataframe(
        coverage,
        use_container_width=True,
        hide_index=True,
    )

    if not coverage.empty:
        st.bar_chart(
            coverage.set_index("field")["coverage_percent"]
        )

    st.info(
        "A field being populated does not establish "
        "that its extracted value is correct."
    )

    st.subheader("Quality issue counts")

    if not issues.empty:
        issue_counts = (
            issues["issue"]
            .value_counts()
            .rename_axis("issue")
            .reset_index(name="count")
        )

        st.bar_chart(
            issue_counts.set_index("issue")["count"]
        )

        with st.expander("Inspect individual quality flags"):
            st.dataframe(
                issues,
                use_container_width=True,
                hide_index=True,
            )
    else:
        show_empty("No quality issues recorded.")

    st.subheader("Exact-text duplicate candidates")

    if not duplicates.empty:
        st.dataframe(
            duplicates,
            use_container_width=True,
            hide_index=True,
        )
    else:
        show_empty("No exact-text duplicates found.")

    st.caption(
        "Different tweets can describe the same incident. "
        "Exact-text deduplication alone does not resolve that."
    )

    st.subheader("Evaluation status")

    evaluation = summary.get("evaluation", {})

    st.write(
        "Independent precision:",
        evaluation.get("precision") or "Not measured"
    )
    st.write(
        "Independent recall:",
        evaluation.get("recall") or "Not measured"
    )
    st.write(
        "Independent F1:",
        evaluation.get("f1") or "Not measured"
    )

    st.caption(
        "These metrics require independent ground-truth labels."
    )


elif section == "Forecasting Evaluation":
    st.header("Chronological Forecasting Evaluation")

    st.info(
        "The existing forecasting experiments are separate "
        "from the Stage 31–32 unverified event candidates. "
        "No new event candidates have been approved as "
        "forecast training data."
    )

    st.subheader("Previously recorded experiment results")

    # These values are from the previously reported
    # Stage 27 experiment, not computed by this dashboard.
    historical = pd.DataFrame([
        {
            "Model": "Rolling mean baseline",
            "Count MAE": 5.223,
        },
        {
            "Model": "ACLED-only",
            "Count MAE": 5.793,
        },
        {
            "Model": "ACLED + social features",
            "Count MAE": 5.710,
        },
    ])

    st.dataframe(
        historical,
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        historical.set_index("Model")["Count MAE"]
    )

    st.caption(
        "Previously reported Stage 27 count MAE values. "
        "These are displayed for reference, not recomputed "
        "from current artifacts. Lower MAE is better."
    )

    st.write(
        "The combined ACLED + social model had a lower "
        "reported MAE than ACLED-only, but the rolling "
        "mean baseline remained stronger."
    )

    st.subheader("Exploratory state-week records")

    if not weekly.empty:
        st.dataframe(
            weekly,
            use_container_width=True,
            hide_index=True,
        )
    else:
        show_empty(
            "No state-week records satisfy the exploratory "
            "eligibility rules."
        )

    st.warning(
        "Exploratory report counts are not confirmed "
        "unique incident counts and must not be used "
        "as ground-truth forecasting targets."
    )


elif section == "Data and Methodology":
    st.header("Data and Methodology")

    st.markdown("""
### Pipeline

1. Raw social-media and ACLED data ingestion
2. Text preprocessing and candidate selection
3. NLP-based event detection and evidence extraction
4. Temporal and location resolution
5. Automated structured event annotation
6. Duplicate-candidate identification
7. Extraction coverage and quality evaluation
8. Chronological forecasting experiments
9. Interactive dashboard presentation

### Methodological safeguards

- Publication date is not automatically treated as event date.
- Unknown occurrence dates remain missing.
- Historical mentions are not automatically counted as new incidents.
- Ongoing protests do not automatically create new weekly incidents.
- Exact-text deduplication does not guarantee unique real-world events.
- Automatic labels are not treated as independent ground truth.
- Forecasting experiments are kept separate from unverified extracted events.
""")

    st.subheader("Input artifacts")

    files = [
        STAGE31 / "structured_events.csv",
        STAGE32 / "event_dashboard_data.csv",
        STAGE32 / "extraction_coverage.csv",
        STAGE32 / "quality_issues.csv",
        STAGE32 / "evaluation_summary.json",
        STAGE32 / "evaluation_report.md",
    ]

    file_status = pd.DataFrame([
        {
            "File": str(p.relative_to(ROOT)),
            "Exists": p.exists(),
        }
        for p in files
    ])

    st.dataframe(
        file_status,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Reproducibility")

    st.code(
        "python src/stage31_auto_finalize.py\n"
        "python src/stage32_evaluate_integrate.py\n"
        "python -m streamlit run src/stage33_dashboard.py",
        language="powershell",
    )


st.divider()
st.caption(
    "Civil Unrest Event Extraction and Prediction | "
    "Stage 33 | Automated research prototype"
)
