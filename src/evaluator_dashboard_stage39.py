
"""Evaluator-ready Streamlit dashboard for Civil Unrest Event Extraction.

Run from project root:
python -m streamlit run src/evaluator_dashboard_full.py --server.port 8504
"""

from pathlib import Path
import json
import re

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ST36 = ROOT / 'outputs/stage36/full_corpus_predictions.parquet'
ST36_SUM = ROOT / 'outputs/stage36/stage36_summary.json'

ST37 = ROOT / 'outputs/stage37/tweet_event_summary.parquet'
ST37_SUM = ROOT / 'outputs/stage37/stage37_summary.json'

ST38_METRICS = ROOT / 'outputs/stage38/model_metrics.csv'
ST38_FORECAST = ROOT / 'outputs/stage38/selected_next_7days.csv'

ST27_SUM = ROOT / 'outputs/stage27/stage27_summary.json'

PROCESSED = ROOT / 'data/processed/tweets_processed.parquet'
ACLED = ROOT / 'data/raw/acled_india.csv'


# ============================================================
# DASHBOARD CONFIGURATION
# ============================================================

st.set_page_config(
    page_title='Civil Unrest | Project Demonstration',
    page_icon='📊',
    layout='wide'
)

st.title('Civil Unrest Event Extraction and Prediction')

st.caption(
    'Raw data → preprocessing → MiniLM classification → '
    'structured extraction → historical ACLED forecasting '
    'experiments → evaluated seven-day tweet-volume forecast'
)


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def read_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}


@st.cache_data(show_spinner='Loading structured event evidence...')
def load_events():

    if not ST37.exists():
        return pd.DataFrame()

    df = pd.read_parquet(ST37)

    if 'tweetId' in df.columns:
        df['tweetId'] = df['tweetId'].astype('string')

    if 'post_date' in df.columns:
        df['post_date'] = pd.to_datetime(
            df['post_date'],
            errors='coerce',
            utc=True
        )

    return df


@st.cache_data(show_spinner='Loading classification results...')
def load_predictions():

    if not ST36.exists():
        return pd.DataFrame()

    df = pd.read_parquet(ST36)

    df['tweetId'] = df['tweetId'].astype('string')

    return df


@st.cache_data(show_spinner='Loading a small tweet sample...')
def load_processed_sample(n=5000):

    import pyarrow.parquet as pq

    if not PROCESSED.exists():
        return pd.DataFrame()

    pf = pq.ParquetFile(PROCESSED)

    frames = []
    left = n

    for batch in pf.iter_batches(
        batch_size=min(2048, n),
        columns=['post_id', 'text', 'date']
    ):

        frame = batch.to_pandas()

        frames.append(frame.head(left))

        left -= len(frames[-1])

        if left <= 0:
            break

    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )


# ============================================================
# LOAD STAGE SUMMARIES
# ============================================================

stage36 = read_json(ST36_SUM)
stage37 = read_json(ST37_SUM)

if not stage36:

    st.error(
        'Stage 36 output missing. '
        'Check outputs/stage36/stage36_summary.json.'
    )

    st.stop()


# ============================================================
# TOP-LEVEL METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    'Processed tweets classified',
    f"{stage36.get('total_classified_tweets', 0):,}"
)

m2.metric(
    'Coverage',
    f"{stage36.get('classification_coverage_percent', 0):.1f}%"
)

m3.metric(
    'Predicted positive',
    f"{stage36.get('predicted_positive_tweets', 0):,}"
)

m4.metric(
    'Predicted negative',
    f"{stage36.get('predicted_negative_tweets', 0):,}"
)

st.info(
    'Predicted-positive tweets are NOT verified incidents. '
    'The MiniLM classifier was trained on 84 verified examples; '
    'its probabilities are uncalibrated.'
)


# ============================================================
# DASHBOARD TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    'Overview',
    'Event evidence',
    'Trends',
    'Forecast & evaluation',
    'Historical ML evaluation'
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.subheader('End-to-end pipeline')

    st.markdown(
        '**Raw tweets (1,084,452)** → '
        '**processed tweets (902,401)** → '
        '**MiniLM predictions (902,401)** → '
        '**positive candidates (48,030)** → '
        '**structured evidence** → '
        '**validation / deduplication (ongoing)** → '
        '**forecasting prototype**'
    )

    st.write(
        'The Stage 35 audit also reconciled 12,117 candidate '
        'tweets present in raw data but excluded from the '
        'processed dataset. They remain preserved separately.'
    )

    st.subheader('Classification distribution')

    chart = pd.DataFrame(
        {
            'Tweets': [
                stage36.get('predicted_positive_tweets', 0),
                stage36.get('predicted_negative_tweets', 0)
            ]
        },
        index=[
            'Predicted positive',
            'Predicted negative'
        ]
    )

    st.bar_chart(chart)

    if stage37:

        st.subheader('Stage 37 extraction coverage')

        a, b, c = st.columns(3)

        a.metric(
            'Unique tweet summaries',
            f"{stage37.get('unique_tweet_event_summaries', 0):,}"
        )

        b.metric(
            'Reused extraction rows',
            f"{stage37.get('reused_extraction_rows', 0):,}"
        )

        c.metric(
            'New fallback extractions',
            f"{stage37.get('positive_tweets_with_new_fallback_extractions', 0):,}"
        )

    else:

        st.warning(
            'Stage 37 summary is not saved yet. '
            'The dashboard still works with Stage 36 data.'
        )


# ============================================================
# TAB 2 — EVENT EVIDENCE
# ============================================================

with tab2:

    st.subheader('Structured event evidence browser')

    events = load_events()

    if events.empty:

        st.warning(
            'Stage 37 output missing. Run: '
            'python -u src/stage37_full_corpus_event_extraction.py'
        )

        st.write('Showing a small processed-tweet sample instead:')

        sample = load_processed_sample()

        if not sample.empty:

            st.dataframe(
                sample.head(100),
                use_container_width=True
            )

    else:

        st.caption(
            f'{len(events):,} tweet-level records. '
            'Extracted fields are unverified hints, '
            'not confirmed incident facts.'
        )

        if 'event_type' in events.columns:

            kinds = (
                events['event_type']
                .fillna('unknown')
                .astype(str)
                .value_counts()
                .head(15)
            )

            st.bar_chart(kinds)

        q = st.text_input(
            'Search tweet text, event type or tweet ID',
            placeholder='e.g. farmers, march, Delhi'
        )

        filtered = events

        if q:

            fields = [
                x for x in [
                    'text',
                    'event_type',
                    'tweetId',
                    'event_location'
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
                    .astype('string')
                    .str.contains(
                        re.escape(q),
                        case=False,
                        na=False
                    )
                )

            filtered = events.loc[mask]

        st.write(f'Matching records: {len(filtered):,}')

        show = [
            x for x in [
                'tweetId',
                'post_date',
                'text',
                'event_type',
                'actors',
                'targets',
                'event_location',
                'event_time',
                'escalation_indicators',
                'minilm_positive_probability',
                'extraction_source'
            ]
            if x in filtered.columns
        ]

        st.dataframe(
            filtered[show].head(250),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            'Download displayed records as CSV',
            filtered[show]
            .head(250)
            .to_csv(index=False)
            .encode('utf-8-sig'),
            file_name='event_evidence_sample.csv',
            mime='text/csv'
        )


# ============================================================
# TAB 3 — TRENDS
# ============================================================

with tab3:

    st.subheader('Social-media event-report trends')

    events = load_events()

    if events.empty or 'post_date' not in events.columns:

        st.warning(
            'Stage 37 output with post_date is needed '
            'for event-report trends.'
        )

    else:

        dated = events.dropna(
            subset=['post_date']
        ).copy()

        dated['day'] = (
            dated['post_date']
            .dt.tz_convert(None)
            .dt.floor('D')
        )

        daily = (
            dated.groupby('day')
            .size()
            .rename('Predicted-positive tweets')
            .sort_index()
        )

        st.line_chart(daily)

        st.caption(
            'Counts are positive social-media reports by '
            'tweet posting date, NOT independently verified '
            'incident counts.'
        )

        st.subheader('Top extracted event-type labels')

        if 'event_type' in dated.columns:

            st.bar_chart(
                dated['event_type']
                .fillna('unknown')
                .astype(str)
                .value_counts()
                .head(12)
            )

        if ACLED.exists():

            st.caption(
                'ACLED weekly reference data is available '
                'in data/raw/acled_india.csv. '
                'Its units and coverage must be aligned '
                'before quantitative comparison.'
            )


# ============================================================
# TAB 4 — STAGE 38 FORECAST AND EVALUATION
# ============================================================

with tab4:

    st.subheader(
        'Stage 38 — Seven-day tweet-volume forecast '
        'and backtesting'
    )

    st.warning(
        'This forecasts the volume of classifier-positive '
        'tweets, NOT verified civil-unrest incidents. '
        'The dataset ends in 2021; this is not a live forecast.'
    )

    if (
        not ST38_METRICS.exists()
        or not ST38_FORECAST.exists()
    ):

        st.error(
            'Stage 38 results missing. '
            'Run: python src/stage38_compare_forecasts.py'
        )

        st.code(
            'python src/stage38_compare_forecasts.py',
            language='powershell'
        )

    else:

        try:

            model_metrics = pd.read_csv(ST38_METRICS)
            future = pd.read_csv(ST38_FORECAST)

            required_metrics = {
                'period',
                'model',
                'mae',
                'rmse',
                'wape_pct'
            }

            required_future = {
                'date',
                'selected_model',
                'predicted_positive_tweets'
            }

            if not required_metrics.issubset(
                model_metrics.columns
            ):

                raise ValueError(
                    'Missing metrics columns: '
                    + str(
                        sorted(
                            required_metrics
                            - set(model_metrics.columns)
                        )
                    )
                )

            if not required_future.issubset(
                future.columns
            ):

                raise ValueError(
                    'Missing forecast columns: '
                    + str(
                        sorted(
                            required_future
                            - set(future.columns)
                        )
                    )
                )

            if future.empty:

                raise ValueError(
                    'The Stage 38 future forecast file is empty.'
                )

            future['date'] = pd.to_datetime(
                future['date'],
                errors='coerce'
            )

            future['predicted_positive_tweets'] = (
                pd.to_numeric(
                    future['predicted_positive_tweets'],
                    errors='coerce'
                )
            )

            future = (
                future.dropna(
                    subset=[
                        'date',
                        'predicted_positive_tweets'
                    ]
                )
                .sort_values('date')
            )

            if future.empty:

                raise ValueError(
                    'No valid dates and forecast values '
                    'in the Stage 38 forecast file.'
                )

            selected = str(
                future['selected_model'].iloc[0]
            )

            selection = model_metrics[
                model_metrics['period'].eq('selection')
            ].copy()

            holdout = model_metrics[
                model_metrics['period'].eq('holdout')
            ].copy()

            selected_holdout = holdout[
                holdout['model'].eq(selected)
            ]

            st.success(
                f'**Selected model:** {selected} | '
                f'**Forecast period:** '
                f'{future["date"].min():%d %B %Y} '
                f'to {future["date"].max():%d %B %Y}'
            )

            st.caption(
                'Model chosen using selection-period MAE. '
                'The final holdout period was reserved '
                'for evaluation.'
            )

            if not selected_holdout.empty:

                result = selected_holdout.iloc[0]

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    'Holdout MAE (tweets/day)',
                    f'{float(result["mae"]):.3f}'
                )

                c2.metric(
                    'Holdout RMSE (tweets/day)',
                    f'{float(result["rmse"]):.3f}'
                )

                c3.metric(
                    'Holdout WAPE',
                    f'{float(result["wape_pct"]):.2f}%'
                )

                c4.metric(
                    'Forecast daily volume',
                    f'{future["predicted_positive_tweets"].mean():.2f}'
                )

            else:

                st.warning(
                    'No holdout evaluation row found '
                    f'for selected model {selected}.'
                )

            st.subheader('Final seven-day forecast')

            st.dataframe(
                future.rename(
                    columns={
                        'date': 'Forecast date (UTC)',
                        'selected_model': 'Model',
                        'predicted_positive_tweets':
                            'Estimated positive tweets'
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

            st.line_chart(
                future.set_index('date')[
                    'predicted_positive_tweets'
                ].rename('Forecast tweets/day')
            )

            st.subheader(
                'Selection-period model comparison'
            )

            if selection.empty:

                st.warning(
                    'No selection-period metrics found.'
                )

            else:

                selection = selection.sort_values('mae')

                st.dataframe(
                    selection,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    selection.set_index('model')['mae']
                )

            st.subheader('Holdout model comparison')

            if holdout.empty:

                st.warning(
                    'No holdout metrics found.'
                )

            else:

                st.dataframe(
                    holdout.sort_values('mae'),
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    holdout.set_index('model')['mae']
                )

            st.caption(
                'Lower MAE and RMSE are better. '
                'WAPE is an error percentage, not '
                '100 minus accuracy. Holdout scores '
                'must not be compared directly with '
                'Stage 37 scores from different test windows.'
            )

            st.caption(
                'Forecast is based on the last available '
                'historical tweets, not a forecast for '
                'the current calendar week.'
            )

        except Exception as exc:

            st.error(
                f'Unable to display Stage 38 results: {exc}'
            )

            st.info(
                'Check outputs/stage38/model_metrics.csv '
                'and selected_next_7days.csv, '
                'then rerun Stage 38.'
            )


# ============================================================
# TAB 5 — HISTORICAL ML EVALUATION
# ============================================================

with tab5:

    st.subheader(
        'Historical forecasting experiments — '
        'results from earlier project stages'
    )

    st.info(
        'Stage 27 historical state-week count evaluation: '
        '36 states/UTs, 53 aligned weeks, 3 folds and '
        '864 count-evaluation observations. '
        'This differs from the daily tweet-volume '
        'forecast in Stage 38.'
    )

    stage27 = read_json(ST27_SUM)

    count_results = [
        row
        for row in stage27.get('count_results', [])
        if row.get('fold') == 'overall'
    ]

    if count_results:

        scores = pd.DataFrame(count_results)

        scores = scores.rename(
            columns={
                'model': 'Approach',
                'mae': 'MAE',
                'rmse': 'RMSE',
                'rows': 'Test observations'
            }
        )

        st.dataframe(
            scores[
                [
                    'Approach',
                    'Test observations',
                    'MAE',
                    'RMSE'
                ]
            ].sort_values('MAE'),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            scores.set_index('Approach')['MAE']
        )

    else:

        st.warning(
            'Stage 27 summary missing or no overall '
            'count results found. Check '
            'outputs/stage27/stage27_summary.json.'
        )

    st.caption(
        'ACLED-based weekly event-count MAE is in events '
        'per state-week; Stage 38 MAE is in tweets per day. '
        'These metrics are not directly comparable.'
    )

    st.markdown(
        '**Interpretation:** Rolling mean achieved the '
        'lowest Stage 27 count MAE. Adding national '
        'social-media activity features slightly reduced '
        'MAE versus ACLED-only, but did not outperform '
        'the rolling-mean baseline.'
    )

    st.subheader('ACLED historical reference dataset')

    if ACLED.exists():

        try:

            acled = pd.read_csv(ACLED)

            st.metric(
                'ACLED reference rows',
                f'{len(acled):,}'
            )

            st.dataframe(
                acled.head(15),
                use_container_width=True,
                hide_index=True
            )

            if 'events' in acled.columns:

                st.metric(
                    'Sum of recorded ACLED events '
                    '(reference rows)',
                    f"{pd.to_numeric(acled['events'], errors='coerce').sum():,.0f}"
                )

            st.caption(
                'ACLED values are historical reference '
                'data, with their own units, coverage '
                'and definitions. Do not equate them '
                'with tweet counts.'
            )

        except Exception as exc:

            st.warning(
                f'Could not display ACLED preview: {exc}'
            )

    else:

        st.warning(
            'ACLED source file is not present.'
        )

    st.subheader(
        'Earlier separate national-weekly experiment'
    )

    st.write(
        'Previously recorded MAE: '
        'ACLED-only ≈ 85.85; '
        'ACLED + social-media ≈ 85.34. '
        'This was a DIFFERENT forecasting target/setup, '
        'so these values must not be compared directly '
        'with the 5.x MAE scores above.'
    )

    st.warning(
        'For a rigorous final result, rerun or retrieve '
        'the original experiment artifacts and verify '
        'the target, train/test dates, split, and MAE. '
        'This tab does not claim the experiments were rerun.'
    )
