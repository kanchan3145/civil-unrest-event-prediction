
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data/raw/acled_india_historical_2020_2021.csv"
OUTPUT = ROOT / "data/prediction"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Read historical ACLED weekly summary
df = pd.read_csv(INPUT, low_memory=False)

required = {"week", "country", "event_type", "events"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

# India-only civil unrest events
df = df[
    (df["country"] == "India")
    & (df["event_type"].isin(["Protests", "Riots"]))
].copy()

df["week"] = pd.to_datetime(df["week"], errors="coerce")
df["events"] = pd.to_numeric(df["events"], errors="coerce")

df = df.dropna(subset=["week", "events"])

# Weekly counts: sum the 'events' field, not the row count.
weekly = (
    df.groupby("week")["events"]
    .sum()
    .sort_index()
    .rename("event_count")
    .to_frame()
)

# Fill missing weeks in the observed range.
# Check source completeness before interpreting zeros as true zeros.
weekly = weekly.asfreq("7D", fill_value=0)

print("\nWEEKLY DATA")
print("Weeks:", len(weekly))
print("Start:", weekly.index.min().date())
print("End:", weekly.index.max().date())

# Features available at the end of week t.
weekly["lag_1"] = weekly["event_count"]
weekly["lag_2"] = weekly["event_count"].shift(1)
weekly["lag_3"] = weekly["event_count"].shift(2)
weekly["rolling_3"] = (
    weekly["event_count"].rolling(3).mean()
)

# Target: events during week t+1
weekly["next_week_events"] = weekly["event_count"].shift(-1)

features = ["lag_1", "lag_2", "lag_3", "rolling_3"]

model_data = weekly.dropna(
    subset=features + ["next_week_events"]
).copy()

if len(model_data) < 20:
    raise ValueError(
        "Not enough chronological observations for evaluation."
    )

# Chronological 80/20 split
split = int(len(model_data) * 0.8)

train = model_data.iloc[:split]
test = model_data.iloc[split:]

X_train = train[features]
y_train = train["next_week_events"]

X_test = test[features]
y_test = test["next_week_events"]

# Lightweight CPU-friendly regression
model = Ridge(alpha=10.0)
model.fit(X_train, y_train)

predictions = np.maximum(model.predict(X_test), 0)

# Naive comparison: next week = current week
baseline = test["lag_1"].to_numpy()

def evaluate(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(
            mean_squared_error(actual, predicted)
        ))
    }

model_metrics = evaluate(y_test, predictions)
baseline_metrics = evaluate(y_test, baseline)

results = pd.DataFrame({
    "feature_week": test.index.strftime("%Y-%m-%d"),
    "target_week": (
        test.index + pd.Timedelta(days=7)
    ).strftime("%Y-%m-%d"),
    "actual_events": y_test.to_numpy(),
    "predicted_events": predictions,
    "baseline_events": baseline
})

# Predict one week after the latest observed week.
latest_features = weekly[features].iloc[[-1]]

future_prediction = max(
    float(model.predict(latest_features)[0]),
    0.0
)

future_week = (
    weekly.index[-1] + pd.Timedelta(days=7)
).date().isoformat()

# Save outputs, avoiding accidental overwrites.
paths = [
    OUTPUT / "weekly_acled_v1.csv",
    OUTPUT / "prediction_results_v1.csv",
    OUTPUT / "prediction_summary_v1.json"
]

for path in paths:
    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}"
        )

weekly.to_csv(paths[0])
results.to_csv(paths[1], index=False)

summary = {
    "training_weeks": len(train),
    "testing_weeks": len(test),
    "model": "Ridge Regression",
    "model_metrics": model_metrics,
    "baseline_metrics": baseline_metrics,
    "forecast_week": future_week,
    "forecast_event_count": round(future_prediction, 2),
    "forecast_type": "historical_backtest",
    "note": (
        "ACLED-only baseline. The forecast follows the "
        "last historical week, not the current calendar week."
    )
}

paths[2].write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nSTAGE 13 COMPLETE")
print(json.dumps(summary, indent=2))
print("\nSaved outputs to:", OUTPUT)
