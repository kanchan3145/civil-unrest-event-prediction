"""Evaluator-ready Streamlit dashboard for Civil Unrest Event Extraction.
Copy to src/evaluator_dashboard.py; run: streamlit run src/evaluator_dashboard.py
"""
from pathlib import Path
import json
import re
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ST36 = ROOT / 'outputs/stage36/full_corpus_predictions.parquet'
ST36_SUM = ROOT / 'outputs/stage36/stage36_summary.json'
ST37 = ROOT / 'outputs/stage37/tweet_event_summary.parquet'
ST37_SUM = ROOT / 'outputs/stage37/stage37_summary.json'
PROCESSED = ROOT / 'data/processed/tweets_processed.parquet'
ACLED = ROOT / 'data/raw/acled_india.csv'

st.set_page_config(page_title='Civil Unrest | Project Demonstration', page_icon='📊', layout='wide')
st.title('Civil Unrest Event Extraction and Prediction')
st.caption('Raw data → preprocessing → MiniLM classification → structured extraction → historical ACLED forecasting experiments → interactive report-volume baseline')

@st.cache_data(show_spinner=False)
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

@st.cache_data(show_spinner='Loading structured event evidence...')
def load_events():
    if not ST37.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ST37)
    if 'tweetId' in df.columns:
        df['tweetId'] = df['tweetId'].astype('string')
    if 'post_date' in df.columns:
        df['post_date'] = pd.to_datetime(df['post_date'], errors='coerce', utc=True)
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
    for batch in pf.iter_batches(batch_size=min(2048, n), columns=['post_id', 'text', 'date']):
        frame = batch.to_pandas()
        frames.append(frame.head(left))
        left -= len(frames[-1])
        if left <= 0:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

stage36 = read_json(ST36_SUM)
stage37 = read_json(ST37_SUM)
if not stage36:
    st.error('Stage 36 output missing. Run from the project root and check outputs/stage36/stage36_summary.json.')
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric('Processed tweets classified', f"{stage36.get('total_classified_tweets', 0):,}")
m2.metric('Coverage', f"{stage36.get('classification_coverage_percent', 0):.1f}%")
m3.metric('Predicted positive', f"{stage36.get('predicted_positive_tweets', 0):,}")
m4.metric('Predicted negative', f"{stage36.get('predicted_negative_tweets', 0):,}")
st.info('Predicted-positive tweets are NOT verified incidents. The MiniLM classifier was trained on 84 verified examples; its probabilities are uncalibrated.')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Overview', 'Event evidence', 'Trends', 'Forecast demo', 'Historical ML evaluation'])

with tab1:
    st.subheader('End-to-end pipeline')
    st.markdown('**Raw tweets (1,084,452)** → **processed tweets (902,401)** → **MiniLM predictions (902,401)** → **positive candidates (48,030)** → **structured evidence** → **validation / deduplication (ongoing)** → **forecasting prototype**')
    st.write('The Stage 35 audit also reconciled 12,117 candidate tweets present in raw data but excluded from the processed dataset. They remain preserved separately.')
    st.subheader('Classification distribution')
    chart = pd.DataFrame({'Tweets': [stage36.get('predicted_positive_tweets', 0), stage36.get('predicted_negative_tweets', 0)]}, index=['Predicted positive', 'Predicted negative'])
    st.bar_chart(chart)
    if stage37:
        st.subheader('Stage 37 extraction coverage')
        a, b, c = st.columns(3)
        a.metric('Unique tweet summaries', f"{stage37.get('unique_tweet_event_summaries', 0):,}")
        b.metric('Reused extraction rows', f"{stage37.get('reused_extraction_rows', 0):,}")
        c.metric('New fallback extractions', f"{stage37.get('positive_tweets_with_new_fallback_extractions', 0):,}")
    else:
        st.warning('Stage 37 summary is not saved yet. The dashboard still works with Stage 36 data.')

with tab2:
    st.subheader('Structured event evidence browser')
    events = load_events()
    if events.empty:
        st.warning('Stage 37 output missing. Run: python -u src/stage37_full_corpus_event_extraction.py')
        st.write('Showing a small processed-tweet sample instead:')
        sample = load_processed_sample()
        if not sample.empty:
            st.dataframe(sample.head(100), use_container_width=True)
    else:
        st.caption(f'{len(events):,} tweet-level records. Extracted fields are unverified hints, not confirmed incident facts.')
        if 'event_type' in events.columns:
            kinds = events['event_type'].fillna('unknown').astype(str).value_counts().head(15)
            st.bar_chart(kinds)
        q = st.text_input('Search tweet text, event type or tweet ID', placeholder='e.g. farmers, march, Delhi')
        filtered = events
        if q:
            fields = [x for x in ['text', 'event_type', 'tweetId', 'event_location'] if x in events.columns]
            mask = pd.Series(False, index=events.index)
            for field in fields:
                mask |= events[field].astype('string').str.contains(re.escape(q), case=False, na=False)
            filtered = events.loc[mask]
        st.write(f'Matching records: {len(filtered):,}')
        show = [x for x in ['tweetId', 'post_date', 'text', 'event_type', 'actors', 'targets', 'event_location', 'event_time', 'escalation_indicators', 'minilm_positive_probability', 'extraction_source'] if x in filtered.columns]
        st.dataframe(filtered[show].head(250), use_container_width=True, hide_index=True)
        st.download_button('Download displayed records as CSV', filtered[show].head(250).to_csv(index=False).encode('utf-8-sig'), file_name='event_evidence_sample.csv', mime='text/csv')

with tab3:
    st.subheader('Social-media event-report trends')
    events = load_events()
    if events.empty or 'post_date' not in events.columns:
        st.warning('Stage 37 output with post_date is needed for event-report trends.')
    else:
        dated = events.dropna(subset=['post_date']).copy()
        dated['day'] = dated['post_date'].dt.tz_convert(None).dt.floor('D')
        daily = dated.groupby('day').size().rename('Predicted-positive tweets').sort_index()
        st.line_chart(daily)
        st.caption('Counts are positive social-media reports by tweet posting date, NOT independently verified incident counts.')
        st.subheader('Top extracted event-type labels')
        if 'event_type' in dated.columns:
            st.bar_chart(dated['event_type'].fillna('unknown').astype(str).value_counts().head(12))
        if ACLED.exists():
            st.caption('ACLED weekly reference data is available in data/raw/acled_india.csv. Its units and coverage must be aligned before quantitative comparison.')

with tab4:
    st.subheader('Illustrative next-7-day volume forecast')
    st.warning('Demonstration only: this forecasts social-media report volume using a trailing average, not future real-world unrest incidents. It has not been validated as an operational forecasting model.')
    events = load_events()
    if events.empty or 'post_date' not in events.columns:
        st.info('Complete Stage 37 to enable this demonstration.')
    else:
        dates = pd.to_datetime(events['post_date'], utc=True, errors='coerce').dropna().dt.tz_convert(None).dt.floor('D')
        if len(dates) == 0:
            st.warning('No valid posting dates available.')
        else:
            counts = dates.value_counts().sort_index()
            calendar = pd.date_range(counts.index.min(), counts.index.max(), freq='D')
            counts = counts.reindex(calendar, fill_value=0)
            window = st.slider('Trailing-average window (days)', 3, 28, 7)
            forecast_value = float(counts.tail(window).mean())
            future_days = pd.date_range(counts.index.max() + pd.Timedelta(days=1), periods=7, freq='D')
            historical_last_day = counts.index.max()
            forecast_start = future_days.min()
            forecast_end = future_days.max()
            st.success(
                f'**Forecast year: {forecast_start.year}** | '
                f'**Forecast period:** {forecast_start:%d %B %Y} '
                f'to {forecast_end:%d %B %Y} (UTC calendar dates)'
            )
            st.caption(f'Last historical report date used: {historical_last_day:%d %B %Y}. This is a historical dataset forecast, not a live {pd.Timestamp.now().year} prediction.')
            forecast = pd.Series([forecast_value] * 7, index=future_days, name='Illustrative forecast')
            history = counts.tail(45).rename('Historical positive reports')
            st.line_chart(pd.concat([history, forecast], axis=1))
            st.metric('Illustrative daily report volume', f'{forecast_value:,.1f}')
            st.markdown('**Forecast dates and daily estimated positive tweet reports**')
            st.dataframe(pd.DataFrame({'Forecast date (UTC)': future_days.strftime('%d %B %Y'), 'Estimated positive tweets': [round(forecast_value, 1)] * len(future_days)}), hide_index=True, use_container_width=True)
            st.caption('Forecast uses only the last available historical dates. Historical data may be old; do not interpret forecast dates as a current live prediction.')


with tab5:
    st.subheader('Historical forecasting experiments — results from earlier project stages')
    st.info('These are previously recorded experimental MAE values, NOT scores computed by the current Streamlit app. The evaluation used an earlier limited cluster/prototype setup, not the completed 902,401-tweet pipeline.')
    scores = pd.DataFrame([
        {'Approach': 'Rolling-mean baseline', 'MAE': 5.223, 'Data': 'Earlier 21-cluster prototype'},
        {'Approach': 'ACLED-only model', 'MAE': 5.793, 'Data': 'Earlier 21-cluster prototype'},
        {'Approach': 'ACLED + social-media model', 'MAE': 5.710, 'Data': 'Earlier 21-cluster prototype'},
    ])
    st.dataframe(scores, use_container_width=True, hide_index=True)
    st.bar_chart(scores.set_index('Approach')['MAE'])
    st.caption('MAE = Mean Absolute Error. Lower is better. In this experiment, the rolling-mean baseline performed best; social-media features slightly improved on the ACLED-only model but did not beat the baseline.')
    st.markdown('**What was tested?** A baseline forecast, a model using ACLED historical events, and a model incorporating ACLED plus social-media signals. These are experimental comparisons, not verified real-time incident forecasts.')
    st.subheader('ACLED historical reference dataset')
    if ACLED.exists():
        try:
            acled = pd.read_csv(ACLED)
            st.metric('ACLED reference rows', f'{len(acled):,}')
            st.dataframe(acled.head(15), use_container_width=True, hide_index=True)
            if 'events' in acled.columns:
                st.metric('Sum of recorded ACLED events (reference rows)', f"{pd.to_numeric(acled['events'], errors='coerce').sum():,.0f}")
            st.caption('ACLED values are historical reference data, with their own units, coverage and definitions. Do not equate them with tweet counts.')
        except Exception as exc:
            st.warning(f'Could not display ACLED preview: {exc}')
    else:
        st.warning('ACLED source file is not present.')
    st.subheader('Earlier separate national-weekly experiment')
    st.write('Previously recorded MAE: ACLED-only ≈ 85.85; ACLED + social-media ≈ 85.34. This was a DIFFERENT forecasting target/setup, so these values must not be compared directly with the 5.x MAE scores above.')
    st.warning('For a rigorous final result, rerun or retrieve the original experiment artifacts and verify the target, train/test dates, split, and MAE. This tab does not claim the experiments were rerun.')
