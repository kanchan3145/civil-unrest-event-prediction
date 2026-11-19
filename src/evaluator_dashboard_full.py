
"""
Civil Unrest Event Extraction and Prediction
Evaluator Dashboard — Stage 39

Run:
python -m streamlit run src/evaluator_dashboard_full.py --server.port 8504
"""

from pathlib import Path
import json
import re

import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ST36 = ROOT / "outputs/stage36/full_corpus_predictions.parquet"
ST36_SUM = ROOT / "outputs/stage36/stage36_summary.json"

ST37 = ROOT / "outputs/stage37/tweet_event_summary.parquet"
ST37_SUM = ROOT / "outputs/stage37/stage37_summary.json"

ST38_METRICS = ROOT / "outputs/stage38/model_metrics.csv"
ST38_FORECAST = ROOT / "outputs/stage38/selected_next_7days.csv"

ST27_SUM = ROOT / "outputs/stage27/stage27_summary.json"
ST27_METRICS = ROOT / "outputs/stage28/stage27_recomputed_metrics.csv"

PROCESSED = ROOT / "data/processed/tweets_processed.parquet"
ACLED = ROOT / "data/raw/acled_india.csv"


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Civil Unrest | Project Demonstration",
    page_icon="📊",
    layout="wide"
)

st.title("Civil Unrest Event Extraction and Prediction")

st.caption(
    "Raw data → preprocessing → MiniLM classification → "
    "structured extraction → historical ACLED forecasting "
    "→ seven-day social-media report forecasting"
)


# ============================================================
# 3. DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def read_json(path):
    if not path.exists():
        return {}

    return json.loads(
        path.read_text(encoding="utf-8")
    )


@st.cache_data(show_spinner="Loading structured event evidence...")
def load_events():

    if not ST37.exists():
        return pd.DataFrame()

    df = pd.read_parquet(ST37)

    if "tweetId" in df.columns:
        df["tweetId"] = df["tweetId"].astype("string")

    if "post_date" in df.columns:
        df["post_date"] = pd.to_datetime(
            df["post_date"],
            errors="coerce",
            utc=True
        )

    return df


@st.cache_data(show_spinner="Loading classification results...")
def load_predictions():

    if not ST36.exists():
        return pd.DataFrame()

    df = pd.read_parquet(ST36)

    if "tweetId" in df.columns:
        df["tweetId"] = df["tweetId"].astype("string")

    return df


@st.cache_data(show_spinner="Loading tweet sample...")
def load_processed_sample(n=5000):

    import pyarrow.parquet as pq

    if not PROCESSED.exists():
        return pd.DataFrame()

    pf = pq.ParquetFile(PROCESSED)

    frames = []
    remaining = n

    for batch in pf.iter_batches(
        batch_size=min(2048, n),
        columns=["post_id", "text", "date"]
    ):

        frame = batch.to_pandas()

        frames.append(frame.head(remaining))

        remaining -= len(frames[-1])

        if remaining <= 0:
            break

    if frames:
        return pd.concat(
            frames,
            ignore_index=True
        )

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_stage38():

    metrics = pd.DataFrame()
    forecast = pd.DataFrame()

    if ST38_METRICS.exists():
        metrics = pd.read_csv(ST38_METRICS)

    if ST38_FORECAST.exists():
        forecast = pd.read_csv(ST38_FORECAST)

    return metrics, forecast


@st.cache_data(show_spinner=False)
def load_stage27_metrics():

    if not ST27_METRICS.exists():
        return pd.DataFrame()

    return pd.read_csv(ST27_METRICS)


# ============================================================
# 4. LOAD SUMMARIES
# ============================================================

stage36 = read_json(ST36_SUM)
stage37 = read_json(ST37_SUM)

if not stage36:

    st.error(
        "Stage 36 summary missing. Check "
        "outputs/stage36/stage36_summary.json."
    )

    st.stop()


# ============================================================
# 5. TOP METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Processed tweets classified",
    f"{stage36.get('total_classified_tweets', 0):,}"
)

m2.metric(
    "Coverage",
    f"{stage36.get('classification_coverage_percent', 0):.1f}%"
)

m3.metric(
    "Predicted positive",
    f"{stage36.get('predicted_positive_tweets', 0):,}"
)

m4.metric(
    "Predicted negative",
    f"{stage36.get('predicted_negative_tweets', 0):,}"
)

st.info(
    "Predicted-positive tweets are NOT verified incidents. "
    "The MiniLM classifier was trained on 84 verified examples; "
    "its probabilities are uncalibrated."
)


# ============================================================
# 6. TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Event evidence",
    "Trends",
    "Forecast demo",
    "Historical ML evaluation"
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.subheader("End-to-end pipeline")

    st.markdown(
        "**Raw tweets (1,084,452)** → "
        "**processed tweets (902,401)** → "
        "**MiniLM predictions (902,401)** → "
        "**positive candidates (48,030)** → "
        "**structured evidence** → "
        "**validation / deduplication (ongoing)** → "
        "**forecasting prototype**"
    )

    st.write(
        "The Stage 35 audit reconciled 12,117 candidate "
        "tweets present in raw data but excluded from the "
        "processed dataset. They remain preserved separately."
    )

    st.subheader("Classification distribution")

    chart = pd.DataFrame(
        {
            "Tweets": [
                stage36.get("predicted_positive_tweets", 0),
                stage36.get("predicted_negative_tweets", 0)
            ]
        },
        index=[
            "Predicted positive",
            "Predicted negative"
        ]
    )

    st.bar_chart(chart)

    if stage37:

        st.subheader("Stage 37 extraction coverage")

        a, b, c = st.columns(3)

        a.metric(
            "Unique tweet summaries",
            f"{stage37.get('unique_tweet_event_summaries', 0):,}"
        )

        b.metric(
            "Reused extraction rows",
            f"{stage37.get('reused_extraction_rows', 0):,}"
        )

        c.metric(
            "New fallback extractions",
            f"{stage37.get('positive_tweets_with_new_fallback_extractions', 0):,}"
        )

    else:

        st.warning(
            "Stage 37 summary is not saved yet."
        )


# ============================================================
# TAB 2 — EVENT EVIDENCE
# ============================================================

with tab2:

    st.subheader("Structured event evidence browser")

    events = load_events()

    if events.empty:

        st.warning(
            "Stage 37 output missing. Run the Stage 37 "
            "full-corpus event extraction script."
        )

        sample = load_processed_sample()

        if not sample.empty:

            st.dataframe(
                sample.head(100),
                use_container_width=True
            )

    else:

        st.caption(
            f"{len(events):,} tweet-level records. "
            "Extracted fields are unverified hints, "
            "not confirmed incident facts."
        )

        if "event_type" in events.columns:

            kinds = (
                events["event_type"]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .head(15)
            )

            st.bar_chart(kinds)

        q = st.text_input(
            "Search tweet text, event type or tweet ID",
            placeholder="e.g. farmers, march, Delhi"
        )

        filtered = events

        if q:

            fields = [
                x for x in [
                    "text",
                    "event_type",
                    "tweetId",
                    "event_location"
                ]
                if x in events.columns
            ]

            mask = pd.Series(
                False,
                index=events.index
            )

            for field in fields:

                mask |= (
                    events[field]
                    .astype("string")
                    .str.contains(
                        re.escape(q),
                        case=False,
                        na=False
                    )
                )

            filtered = events.loc[mask]

        st.write(
            f"Matching records: {len(filtered):,}"
        )

        show = [
            x for x in [
                "tweetId",
                "post_date",
                "text",
                "event_type",
                "actors",
                "targets",
                "event_location",
                "event_time",
                "escalation_indicators",
                "minilm_positive_probability",
                "extraction_source"
            ]
            if x in filtered.columns
        ]

        st.dataframe(
            filtered[show].head(250),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download displayed records as CSV",
            filtered[show]
            .head(250)
            .to_csv(index=False)
            .encode("utf-8-sig"),
            file_name="event_evidence_sample.csv",
            mime="text/csv"
        )


# ============================================================
# TAB 3 — TRENDS
# ============================================================

with tab3:

    st.subheader("Social-media event-report trends")

    events = load_events()

    if events.empty or "post_date" not in events.columns:

        st.warning(
            "Stage 37 output with post_date is needed "
            "for event-report trends."
        )

    else:

        dated = events.dropna(
            subset=["post_date"]
        ).copy()

        dated["day"] = (
            dated["post_date"]
            .dt.tz_convert(None)
            .dt.floor("D")
        )

        daily = (
            dated.groupby("day")
            .size()
            .rename("Predicted-positive tweets")
            .sort_index()
        )

        st.line_chart(daily)

        st.caption(
            "Counts are classifier-positive social-media "
            "reports by posting date, NOT independently "
            "verified incident counts."
        )

        st.subheader("Top extracted event-type labels")

        if "event_type" in dated.columns:

            st.bar_chart(
                dated["event_type"]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .head(12)
            )

        if ACLED.exists():

            st.caption(
                "ACLED weekly reference data is available. "
                "Its units and coverage must be aligned "
                "before quantitative comparison."
            )


# ============================================================
# TAB 4 — FORECAST DEMO
# ============================================================

with tab4:

    st.subheader(
        "Illustrative next-7-day volume forecast"
    )

    st.warning(
        "Demonstration only: this forecasts social-media "
        "report volume, not future real-world unrest "
        "incidents. It has not been validated as an "
        "operational forecasting model."
    )

    events = load_events()

    if events.empty or "post_date" not in events.columns:

        st.info(
            "Complete Stage 37 to enable this demonstration."
        )

    else:

        dated = events.dropna(
            subset=["post_date", "tweetId"]
        ).copy()

        dated["day"] = (
            dated["post_date"]
            .dt.tz_convert(None)
            .dt.floor("D")
        )

        counts = (
            dated.groupby("day")["tweetId"]
            .nunique()
            .sort_index()
        )

        if counts.empty:

            st.warning(
                "No valid posting dates available."
            )

        else:

            calendar = pd.date_range(
                counts.index.min(),
                counts.index.max(),
                freq="D"
            )

            counts = counts.reindex(
                calendar,
                fill_value=0
            )

            # ------------------------------------------------
            # FORECAST MODEL SELECTION
            # ------------------------------------------------

            model_choice = st.radio(
                "Forecasting method",
                [
                    "Trailing moving average",
                    "EWMA-7 (Stage 38 selected model)"
                ],
                horizontal=True
            )

            window = st.slider(
                "Trailing-average window (days)",
                min_value=3,
                max_value=28,
                value=7,
                disabled=(
                    model_choice
                    != "Trailing moving average"
                )
            )

            if model_choice == "Trailing moving average":

                forecast_value = float(
                    counts.tail(window).mean()
                )

                model_name = (
                    f"{window}-day moving average"
                )

            else:

                # EWMA with span 7, matching Stage 38.
                forecast_value = float(
                    counts.ewm(
                        span=7,
                        adjust=False
                    ).mean().iloc[-1]
                )

                model_name = "EWMA-7"

            # ------------------------------------------------
            # FORECAST DATES
            # ------------------------------------------------

            historical_last_day = counts.index.max()

            future_days = pd.date_range(
                historical_last_day
                + pd.Timedelta(days=1),
                periods=7,
                freq="D"
            )

            forecast_start = future_days.min()
            forecast_end = future_days.max()

            st.success(
                f"**Forecast year: {forecast_start.year}** | "
                f"**Forecast period:** "
                f"{forecast_start:%d %B %Y} "
                f"to {forecast_end:%d %B %Y} "
                "(UTC calendar dates)"
            )

            st.caption(
                f"Last historical report date used: "
                f"{historical_last_day:%d %B %Y}. "
                "This is a historical dataset forecast, "
                f"not a live {pd.Timestamp.now().year} prediction."
            )

            # ------------------------------------------------
            # HISTORICAL + FORECAST CHART
            # ------------------------------------------------

            forecast = pd.Series(
                [forecast_value] * 7,
                index=future_days,
                name="Illustrative forecast"
            )

            history = (
                counts.tail(45)
                .rename("Historical positive reports")
            )

            combined = pd.concat(
                [history, forecast],
                axis=1
            )

            st.line_chart(combined)

            # ------------------------------------------------
            # FORECAST VALUE
            # ------------------------------------------------

            st.metric(
                "Illustrative daily report volume",
                f"{forecast_value:,.1f}"
            )

            st.caption(
                f"Forecast method: {model_name}"
            )

            # ------------------------------------------------
            # FORECAST TABLE
            # ------------------------------------------------

            st.markdown(
                "**Forecast dates and daily estimated "
                "positive tweet reports**"
            )

            forecast_table = pd.DataFrame(
                {
                    "Forecast date (UTC)":
                        future_days.strftime(
                            "%d %B %Y"
                        ),

                    "Estimated positive tweets":
                        [
                            round(forecast_value, 1)
                        ] * len(future_days)
                }
            )

            st.dataframe(
                forecast_table,
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "Forecast uses only the last available "
                "historical dates. Historical data may "
                "be old; do not interpret forecast dates "
                "as a current live prediction."
            )

            # ------------------------------------------------
            # STAGE 38 EVALUATION
            # ------------------------------------------------

            st.divider()

            with st.expander(
                "Stage 38 — Model comparison and "
                "historical evaluation",
                expanded=False
            ):

                metrics, saved_forecast = load_stage38()

                if metrics.empty:

                    st.warning(
                        "Stage 38 results missing. Run: "
                        "python src/stage38_compare_forecasts.py"
                    )

                else:

                    required = {
                        "period",
                        "model",
                        "mae",
                        "rmse",
                        "wape_pct"
                    }

                    if not required.issubset(metrics.columns):

                        st.error(
                            "Stage 38 metrics CSV is missing "
                            "required columns."
                        )

                    else:

                        selection = metrics[
                            metrics["period"] == "selection"
                        ].copy()

                        holdout = metrics[
                            metrics["period"] == "holdout"
                        ].copy()

                        st.markdown(
                            "#### Selection-period comparison"
                        )

                        if not selection.empty:

                            selection = selection.sort_values(
                                "mae"
                            )

                            st.dataframe(
                                selection,
                                hide_index=True,
                                use_container_width=True
                            )

                            st.bar_chart(
                                selection.set_index("model")[
                                    "mae"
                                ]
                            )

                        st.markdown(
                            "#### Holdout-period comparison"
                        )

                        if not holdout.empty:

                            st.dataframe(
                                holdout.sort_values("mae"),
                                hide_index=True,
                                use_container_width=True
                            )

                            st.bar_chart(
                                holdout.set_index("model")[
                                    "mae"
                                ]
                            )

                            ewma = holdout[
                                holdout["model"] == "ewma_7"
                            ]

                            if not ewma.empty:

                                result = ewma.iloc[0]

                                a, b, c = st.columns(3)

                                a.metric(
                                    "EWMA-7 Holdout MAE",
                                    f"{result['mae']:.3f}"
                                )

                                b.metric(
                                    "EWMA-7 Holdout RMSE",
                                    f"{result['rmse']:.3f}"
                                )

                                c.metric(
                                    "EWMA-7 Holdout WAPE",
                                    f"{result['wape_pct']:.2f}%"
                                )

                        st.caption(
                            "Selection-period results were "
                            "used to choose the forecasting "
                            "model. The holdout period was "
                            "reserved for evaluation. "
                            "Lower MAE and RMSE are better."
                        )

                if not saved_forecast.empty:

                    st.markdown(
                        "#### Saved Stage 38 forecast"
                    )

                    st.dataframe(
                        saved_forecast,
                        use_container_width=True,
                        hide_index=True
                    )


# ============================================================
# TAB 5 — HISTORICAL ML EVALUATION
# ============================================================

with tab5:

    st.subheader(
        "Historical forecasting experiments"
    )

    st.info(
        "Stage 27 evaluated historical state-week "
        "ACLED event-count forecasting. This is a "
        "different prediction target from Stage 38 "
        "daily positive-tweet volume."
    )

    stage27 = read_json(ST27_SUM)

    st.markdown(
        "### Stage 27 — ACLED model comparison"
    )

    if ST27_METRICS.exists():

        scores = load_stage27_metrics()

        if not scores.empty:

            st.dataframe(
                scores,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "Stage 27 metrics CSV is empty."
            )

    else:

        st.warning(
            "Stage 27 recomputed metrics CSV "
            "is not available."
        )

    st.markdown(
        "#### Verified count-forecasting MAE"
    )

    count_scores = pd.DataFrame(
        [
            {
                "Approach": "Rolling-mean baseline",
                "MAE": 5.222801
            },
            {
                "Approach": "ACLED-only model",
                "MAE": 5.793100
            },
            {
                "Approach": "ACLED + social-media model",
                "MAE": 5.710125
            }
        ]
    )

    st.dataframe(
        count_scores,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        count_scores.set_index("Approach")["MAE"]
    )

    st.caption(
        "Stage 27: 36 states/UTs, 53 aligned weeks, "
        "3 folds and 864 count-evaluation observations. "
        "Lower MAE is better. Rolling mean performed best."
    )

    st.markdown(
        "**Interpretation:** Adding national social-media "
        "features slightly improved the ACLED-only model, "
        "but did not outperform the rolling-mean baseline."
    )

    st.subheader(
        "ACLED historical reference dataset"
    )

    if ACLED.exists():

        try:

            acled = pd.read_csv(ACLED)

            st.metric(
                "ACLED reference rows",
                f"{len(acled):,}"
            )

            st.dataframe(
                acled.head(15),
                use_container_width=True,
                hide_index=True
            )

            if "events" in acled.columns:

                total = pd.to_numeric(
                    acled["events"],
                    errors="coerce"
                ).sum()

                st.metric(
                    "Sum of recorded ACLED events "
                    "(reference rows)",
                    f"{total:,.0f}"
                )

            st.caption(
                "ACLED reference data has its own "
                "units and coverage. It must not "
                "be equated with tweet counts."
            )

        except Exception as exc:

            st.warning(
                f"Could not display ACLED preview: {exc}"
            )

    else:

        st.warning(
            "ACLED source file is not present."
        )

    st.subheader(
        "Earlier separate national-weekly experiment"
    )

    st.write(
        "Previously recorded MAE: "
        "ACLED-only ≈ 85.85; "
        "ACLED + social-media ≈ 85.34. "
        "This used a different forecasting target "
        "and evaluation setup."
    )

    st.warning(
        "Historical experimental metrics are not "
        "directly comparable with Stage 38 "
        "tweet-volume forecasting metrics."
    )
