
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# STAGE 15: FINAL EVALUATION AND VISUALIZATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PREDICTION_DIR = ROOT / "data" / "prediction"

FEATURES_FILE = (
    PREDICTION_DIR
    / "stage14_social_v1_weekly_features.csv"
)

PREDICTIONS_FILE = (
    PREDICTION_DIR
    / "stage14_social_v1_predictions.csv"
)

SUMMARY_FILE = (
    PREDICTION_DIR
    / "stage14_social_v1_summary.json"
)

OUTPUT_DIR = ROOT / "outputs" / "stage15"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD SAVED RESULTS
# ============================================================

print("\nSTAGE 15: FINAL EVALUATION AND VISUALIZATION")

for file_path in [
    FEATURES_FILE,
    PREDICTIONS_FILE,
    SUMMARY_FILE
]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

print("\n[1/5] Loading Stage 14 results...")

features = pd.read_csv(FEATURES_FILE)
predictions = pd.read_csv(PREDICTIONS_FILE)

with open(SUMMARY_FILE, "r", encoding="utf-8") as file:
    summary = json.load(file)

# The Stage 14 weekly features CSV stores the week index
# as its first column.
features = features.rename(
    columns={features.columns[0]: "week"}
)

features["week"] = pd.to_datetime(
    features["week"],
    errors="coerce"
)

predictions["target_week"] = pd.to_datetime(
    predictions["target_week"],
    errors="coerce"
)

features = features.dropna(
    subset=["week"]
).sort_values("week")

predictions = predictions.dropna(
    subset=["target_week"]
).sort_values("target_week")

required_predictions = [
    "actual_events",
    "baseline_prediction",
    "acled_only_prediction",
    "social_prediction"
]

missing = [
    col for col in required_predictions
    if col not in predictions.columns
]

if missing:
    raise ValueError(
        f"Missing prediction columns: {missing}"
    )

print("Weekly feature rows:", len(features))
print("Prediction test rows:", len(predictions))


# ============================================================
# 2. ACTUAL VS PREDICTED EVENTS
# ============================================================

print("\n[2/5] Creating actual vs predicted graph...")

plt.figure(figsize=(12, 6))

plt.plot(
    predictions["target_week"],
    predictions["actual_events"],
    marker="o",
    linewidth=2,
    label="Actual ACLED events"
)

plt.plot(
    predictions["target_week"],
    predictions["acled_only_prediction"],
    marker="s",
    linestyle="--",
    label="ACLED-only prediction"
)

plt.plot(
    predictions["target_week"],
    predictions["social_prediction"],
    marker="^",
    linestyle="--",
    label="ACLED + Social Media prediction"
)

plt.title(
    "Actual vs Predicted Civil Unrest Events"
)

plt.xlabel("Week")
plt.ylabel("Number of protests and riots")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=35)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_actual_vs_predicted.png",
    dpi=200
)

plt.close()


# ============================================================
# 3. MODEL PERFORMANCE COMPARISON
# ============================================================

print("\n[3/5] Creating model comparison graph...")

metrics = summary["metrics"]

model_names = [
    "Naive Baseline",
    "ACLED-only",
    "ACLED + Social"
]

metric_keys = [
    "baseline",
    "acled_only",
    "acled_plus_social"
]

mae_values = [
    metrics[key]["MAE"]
    for key in metric_keys
]

rmse_values = [
    metrics[key]["RMSE"]
    for key in metric_keys
]

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    mae_values,
    width,
    label="MAE"
)

plt.bar(
    x + width / 2,
    rmse_values,
    width,
    label="RMSE"
)

plt.xticks(x, model_names)
plt.ylabel("Prediction error (events)")
plt.title("Forecasting Model Performance")
plt.legend()
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_model_comparison.png",
    dpi=200
)

plt.close()


# ============================================================
# 4. HISTORICAL CIVIL UNREST TREND
# ============================================================

print("\n[4/5] Creating historical trend graphs...")

plt.figure(figsize=(12, 6))

plt.plot(
    features["week"],
    features["event_count"],
    linewidth=2,
    label="ACLED protest and riot events"
)

plt.title(
    "Historical Weekly Civil Unrest Events in India"
)

plt.xlabel("Week")
plt.ylabel("Number of reported events")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=35)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_historical_unrest_trend.png",
    dpi=200
)

plt.close()


# ============================================================
# 5. SOCIAL MEDIA ACTIVITY GRAPH
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    features["week"],
    features["tweet_count"],
    linewidth=2,
    label="Total tweets"
)

plt.plot(
    features["week"],
    features["unrest_keyword_count"],
    linewidth=2,
    label="Tweets containing unrest keywords"
)

plt.title(
    "Weekly Social Media Activity"
)

plt.xlabel("Week")
plt.ylabel("Number of tweets")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=35)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_social_media_activity.png",
    dpi=200
)

plt.close()


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

print("\n[5/5] Saving final evaluation summary...")

acled_mae = metrics["acled_only"]["MAE"]
social_mae = metrics["acled_plus_social"]["MAE"]
baseline_mae = metrics["baseline"]["MAE"]

improvement_over_acled = (
    (acled_mae - social_mae) / acled_mae
) * 100

improvement_over_baseline = (
    (baseline_mae - social_mae) / baseline_mae
) * 100

final_summary = {
    "project": (
        "Civil Unrest Event Extraction and Prediction "
        "from Social Media Data"
    ),
    "evaluation_stage": 15,
    "forecast_target": (
        "Next-week national protest and riot event count"
    ),
    "model": "Ridge Regression",
    "tweets_processed": summary["tweet_rows_processed"],
    "overlapping_weeks": summary["overlapping_weeks"],
    "training_weeks": summary["training_weeks"],
    "testing_weeks": summary["testing_weeks"],
    "baseline_metrics": metrics["baseline"],
    "acled_only_metrics": metrics["acled_only"],
    "social_model_metrics": metrics["acled_plus_social"],
    "mae_improvement_over_acled_percent": round(
        improvement_over_acled, 2
    ),
    "mae_improvement_over_baseline_percent": round(
        improvement_over_baseline, 2
    ),
    "evaluation_limitations": [
        "Historical chronological holdout evaluation",
        "Only 11 test weeks",
        "Social media keyword signals are not verified events",
        "Tweet sampling may introduce topic bias",
        "No claim of current-date forecasting",
        "Small observed improvement needs further validation"
    ]
}

summary_path = (
    OUTPUT_DIR / "final_evaluation_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        final_summary,
        file,
        indent=2
    )

print("\n" + "=" * 55)
print("STAGE 15 COMPLETE")
print("=" * 55)

print("\nFINAL MODEL RESULTS")

for name, key in [
    ("Naive baseline", "baseline"),
    ("ACLED-only", "acled_only"),
    ("ACLED + Social Media", "acled_plus_social")
]:
    print(
        f"{name}: "
        f"MAE={metrics[key]['MAE']}, "
        f"RMSE={metrics[key]['RMSE']}"
    )

print(
    "\nMAE improvement over ACLED-only:",
    f"{improvement_over_acled:.2f}%"
)

print(
    "MAE improvement over naive baseline:",
    f"{improvement_over_baseline:.2f}%"
)

print("\nGenerated files:")

for file_path in sorted(OUTPUT_DIR.iterdir()):
    print("-", file_path.name)

print("\nSaved to:", OUTPUT_DIR)
