
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# STAGE 14: SOCIAL MEDIA + ACLED WEEKLY FORECASTING
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TWEETS_FILE = ROOT / "data/raw/tweets.csv"
ACLED_FILE = ROOT / "data/raw/acled_india_historical_2020_2021.csv"

OUTPUT_DIR = ROOT / "data/prediction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "stage14_social_v1"

# The downloaded ACLED 'week' values are Saturday dates.
# W-SAT means each week ends on Saturday.
WEEK_FREQUENCY = "W-SAT"

CHUNK_SIZE = 20000

UNREST_PATTERN = re.compile(
    r"\b(?:"
    r"protest|protests|protesting|"
    r"riot|riots|rioting|"
    r"clash|clashes|clashed|"
    r"violence|violent|"
    r"demonstration|demonstrations|"
    r"agitation|agitations|"
    r"strike|strikes|"
    r"lathi.?charge|"
    r"police.?firing|"
    r"farmers.?protest|"
    r"tractor.?rally"
    r")\b",
    flags=re.IGNORECASE
)

# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

for path in [TWEETS_FILE, ACLED_FILE]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}"
        )

# ============================================================
# 2. LOAD AND AGGREGATE HISTORICAL ACLED DATA
# ============================================================

print("\nSTAGE 14: SOCIAL MEDIA + ACLED PREDICTION")
print("\n[1/5] Loading historical ACLED data...")

acled = pd.read_csv(ACLED_FILE, low_memory=False)

required_acled = {
    "week", "country", "event_type", "events"
}

missing = required_acled - set(acled.columns)

if missing:
    raise ValueError(
        f"Missing ACLED columns: {sorted(missing)}"
    )

acled["country"] = (
    acled["country"].astype(str).str.strip()
)

acled["event_type"] = (
    acled["event_type"].astype(str).str.strip()
)

acled = acled[
    (acled["country"] == "India")
    & (acled["event_type"].isin(["Protests", "Riots"]))
].copy()

acled["week"] = pd.to_datetime(
    acled["week"], errors="coerce"
)

acled["events"] = pd.to_numeric(
    acled["events"], errors="coerce"
)

acled = acled.dropna(
    subset=["week", "events"]
)

# Sum the pre-aggregated ACLED 'events' field.
weekly_acled = (
    acled.groupby("week")["events"]
    .sum()
    .sort_index()
    .rename("event_count")
    .to_frame()
)

# Keep exactly the observed ACLED week range.
weekly_acled = weekly_acled.asfreq(
    "7D", fill_value=0
)

print("ACLED weeks:", len(weekly_acled))
print(
    "Date range:",
    weekly_acled.index.min().date(),
    "to",
    weekly_acled.index.max().date()
)

# ============================================================
# 3. STREAM TWEETS IN SMALL CHUNKS
# ============================================================

print("\n[2/5] Processing tweets in chunks...")

tweet_counts = {}
total_rows = 0
valid_dates = 0

reader = pd.read_csv(
    TWEETS_FILE,
    usecols=["date", "renderedContent"],
    chunksize=CHUNK_SIZE,
    dtype=str,
    low_memory=False
)

for chunk_number, chunk in enumerate(reader, start=1):

    total_rows += len(chunk)

    # Parse dates consistently.
    dates = pd.to_datetime(
        chunk["date"],
        errors="coerce",
        utc=True,
        format="mixed"
    )

    valid = dates.notna()
    valid_dates += int(valid.sum())

    if not valid.any():
        continue

    texts = chunk.loc[
        valid, "renderedContent"
    ].fillna("")

    # A keyword match is a signal, not a confirmed event.
    keyword_flags = texts.str.contains(
        UNREST_PATTERN,
        na=False
    )

    # Convert UTC timestamps to calendar dates.
    valid_dates_series = (
        dates.loc[valid]
        .dt.tz_convert(None)
    )

    # Align with ACLED's Saturday-ending weeks.
    weeks = (
        valid_dates_series
        .dt.to_period(WEEK_FREQUENCY)
        .dt.end_time
        .dt.normalize()
    )

    temporary = pd.DataFrame({
        "week": weeks.to_numpy(),
        "tweet_count": 1,
        "unrest_keyword_count": (
            keyword_flags.astype(int).to_numpy()
        )
    })

    grouped = temporary.groupby("week")[
        ["tweet_count", "unrest_keyword_count"]
    ].sum()

    for week, row in grouped.iterrows():

        if week not in tweet_counts:
            tweet_counts[week] = [0, 0]

        tweet_counts[week][0] += int(
            row["tweet_count"]
        )

        tweet_counts[week][1] += int(
            row["unrest_keyword_count"]
        )

    if chunk_number % 10 == 0:
        print(
            f"Processed {total_rows:,} tweets..."
        )

if not tweet_counts:
    raise ValueError(
        "No valid tweet dates were found."
    )

weekly_tweets = pd.DataFrame.from_dict(
    tweet_counts,
    orient="index",
    columns=[
        "tweet_count",
        "unrest_keyword_count"
    ]
)

weekly_tweets.index = pd.to_datetime(
    weekly_tweets.index
)

weekly_tweets.index.name = "week"
weekly_tweets = weekly_tweets.sort_index()

print("\nTotal tweet rows:", total_rows)
print("Rows with valid dates:", valid_dates)
print("Tweet weeks:", len(weekly_tweets))

# ============================================================
# 4. ALIGN BOTH DATASETS AND CREATE FEATURES
# ============================================================

print("\n[3/5] Creating weekly features...")

# Only use weeks covered by both source timelines.
start = max(
    weekly_acled.index.min(),
    weekly_tweets.index.min()
)

end = min(
    weekly_acled.index.max(),
    weekly_tweets.index.max()
)

if start >= end:
    raise ValueError(
        "Tweet and ACLED timelines do not overlap."
    )

combined = weekly_acled.loc[start:end].copy()

combined = combined.join(
    weekly_tweets,
    how="left"
)

# Missing tweet counts inside the observed tweet date range
# are treated as zero observed posts.
combined[
    ["tweet_count", "unrest_keyword_count"]
] = combined[
    ["tweet_count", "unrest_keyword_count"]
].fillna(0)

# Normalize by weekly tweet volume.
combined["unrest_keyword_ratio"] = (
    combined["unrest_keyword_count"]
    / combined["tweet_count"].clip(lower=1)
)

# ACLED features known at the end of week t.
combined["acled_lag_1"] = combined["event_count"]
combined["acled_lag_2"] = (
    combined["event_count"].shift(1)
)
combined["acled_lag_3"] = (
    combined["event_count"].shift(2)
)
combined["acled_rolling_3"] = (
    combined["event_count"]
    .rolling(3)
    .mean()
)

# Social media features from week t.
combined["tweet_volume_log"] = np.log1p(
    combined["tweet_count"]
)

combined["keyword_volume_log"] = np.log1p(
    combined["unrest_keyword_count"]
)

combined["keyword_ratio_lag_2"] = (
    combined["unrest_keyword_ratio"].shift(1)
)

# Predict the FOLLOWING week's ACLED event count.
combined["next_week_events"] = (
    combined["event_count"].shift(-1)
)

acled_features = [
    "acled_lag_1",
    "acled_lag_2",
    "acled_lag_3",
    "acled_rolling_3"
]

social_features = [
    "tweet_volume_log",
    "keyword_volume_log",
    "unrest_keyword_ratio",
    "keyword_ratio_lag_2"
]

combined_features = (
    acled_features + social_features
)

model_data = combined.dropna(
    subset=combined_features + ["next_week_events"]
).copy()

print("Overlapping weeks:", len(combined))
print("Usable training/evaluation rows:", len(model_data))

if len(model_data) < 20:
    raise ValueError(
        "Too few overlapping weeks for a reliable split."
    )

# ============================================================
# 5. CHRONOLOGICAL TRAIN / TEST EVALUATION
# ============================================================

print("\n[4/5] Training forecasting models...")

split = int(len(model_data) * 0.8)

train = model_data.iloc[:split]
test = model_data.iloc[split:]

y_train = train["next_week_events"]
y_test = test["next_week_events"]

# Model A: ACLED-only
acled_model = Ridge(alpha=10.0)

acled_model.fit(
    train[acled_features],
    y_train
)

acled_predictions = np.maximum(
    acled_model.predict(
        test[acled_features]
    ),
    0
)

# Model B: ACLED + social media
social_model = Ridge(alpha=10.0)

social_model.fit(
    train[combined_features],
    y_train
)

social_predictions = np.maximum(
    social_model.predict(
        test[combined_features]
    ),
    0
)

# Model C: naive persistence baseline
baseline_predictions = (
    test["acled_lag_1"].to_numpy()
)

def evaluate(actual, predicted):
    return {
        "MAE": round(
            float(mean_absolute_error(
                actual, predicted
            )),
            4
        ),
        "RMSE": round(
            float(np.sqrt(
                mean_squared_error(
                    actual, predicted
                )
            )),
            4
        )
    }

metrics = {
    "baseline": evaluate(
        y_test, baseline_predictions
    ),
    "acled_only": evaluate(
        y_test, acled_predictions
    ),
    "acled_plus_social": evaluate(
        y_test, social_predictions
    )
}

print("\nMODEL COMPARISON")

for model_name, values in metrics.items():
    print(
        f"{model_name}: "
        f"MAE={values['MAE']}, "
        f"RMSE={values['RMSE']}"
    )

# ============================================================
# 6. SAVE RESULTS
# ============================================================

print("\n[5/5] Saving Stage 14 outputs...")

results = pd.DataFrame({
    "feature_week": test.index.strftime("%Y-%m-%d"),
    "target_week": (
        test.index + pd.Timedelta(days=7)
    ).strftime("%Y-%m-%d"),
    "actual_events": y_test.to_numpy(),
    "baseline_prediction": baseline_predictions,
    "acled_only_prediction": acled_predictions,
    "social_prediction": social_predictions
})

summary = {
    "stage": 14,
    "model": "Ridge Regression",
    "prediction_target": (
        "Next-week India protests and riots count"
    ),
    "tweet_rows_processed": total_rows,
    "tweet_rows_with_valid_dates": valid_dates,
    "overlapping_weeks": len(combined),
    "training_weeks": len(train),
    "testing_weeks": len(test),
    "metrics": metrics,
    "social_media_improves_mae": (
        metrics["acled_plus_social"]["MAE"]
        < metrics["acled_only"]["MAE"]
    ),
    "last_test_target_week": (
        test.index[-1] + pd.Timedelta(days=7)
    ).date().isoformat(),
    "limitations": [
        "Historical evaluation, not a current forecast",
        "Keyword matches are not verified event mentions",
        "Tweet sample may not represent all unrest in India",
        "Only a small number of weekly observations"
    ]
}

outputs = {
    OUTPUT_DIR / f"{PREFIX}_weekly_features.csv":
        combined,
    OUTPUT_DIR / f"{PREFIX}_predictions.csv":
        results,
    OUTPUT_DIR / f"{PREFIX}_summary.json":
        json.dumps(summary, indent=2)
}

# Preserve earlier Stage 13 and Stage 14 outputs.
existing = [p for p in outputs if p.exists()]

if existing:
    raise FileExistsError(
        "Output already exists; nothing overwritten: "
        + ", ".join(str(p) for p in existing)
    )

for path, content in outputs.items():

    if isinstance(content, pd.DataFrame):
        content.to_csv(
            path,
            index=(path.name.endswith(
                "_weekly_features.csv"
            ))
        )
    else:
        path.write_text(
            content,
            encoding="utf-8"
        )

print("\nSTAGE 14 COMPLETE")
print(json.dumps(summary, indent=2))
print("\nOutputs saved to:", OUTPUT_DIR)
