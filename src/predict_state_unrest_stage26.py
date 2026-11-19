
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

# ============================================================
# STAGE 26 - STATE-LEVEL CIVIL UNREST PREDICTION
#
# CPU only
# Uses historical ACLED weekly aggregate data
# No transformer models or retraining of earlier stages
# ============================================================

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT
    / "data"
    / "raw"
    / "acled_india_historical_2020_2021.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "stage26"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Last 20% of target weeks are used for testing.
TEST_FRACTION = 0.20

# Minimum number of historical weeks before
# generating a forecast example.
MIN_HISTORY_WEEKS = 4

# Escalation definition:
# next week has at least 3 more events AND
# at least a 50% increase over current week.
ESCALATION_MIN_INCREASE = 3
ESCALATION_RELATIVE_INCREASE = 0.50

FEATURE_COLUMNS = [
    "count_current",
    "count_lag_1",
    "count_lag_2",
    "count_lag_3",
    "rolling_mean_3",
    "rolling_mean_4",
    "rolling_max_4",
    "rolling_std_4",
    "change_1_week",
    "active_weeks_4",
    "zero_weeks_4",
    "national_count_current",
    "national_change_1_week",
    "week_sin",
    "week_cos",
]

# ============================================================
# UTILITIES
# ============================================================

def save_json(data, filename):
    path = OUTPUT_DIR / filename
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )


def rmse(y_true, y_pred):
    return float(
        np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    )


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse": rmse(y_true, y_pred)
    }


def classification_metrics(y_true, y_pred, scores=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    result = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true, y_pred
            )
        ) if len(np.unique(y_true)) == 2 else None,
        "precision": float(
            precision_score(
                y_true, y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, y_pred,
                zero_division=0
            )
        ),
        "actual_positive": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }

    if (
        scores is not None
        and len(np.unique(y_true)) == 2
    ):
        result["roc_auc"] = float(
            roc_auc_score(y_true, scores)
        )
    else:
        result["roc_auc"] = None

    return result


# ============================================================
# STEP 1 - LOAD ACLED DATA
# ============================================================

def load_acled():

    print("\n[1/7] Loading ACLED dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"ACLED dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    print("Original rows:", len(df))
    print("Columns:", df.columns.tolist())

    required = [
        "week",
        "admin1",
        "event_type",
        "events"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing ACLED columns: {missing}"
        )

    df["week"] = pd.to_datetime(
        df["week"],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None).dt.normalize()

    df["events"] = pd.to_numeric(
        df["events"],
        errors="coerce"
    )

    if df["events"].isna().any():
        raise ValueError(
            "Some ACLED event counts are not numeric."
        )

    if (df["events"] < 0).any():
        raise ValueError(
            "ACLED event counts cannot be negative."
        )

    if not np.allclose(
        df["events"],
        np.round(df["events"])
    ):
        raise ValueError(
            "ACLED event counts must be integers."
        )

    df = df.dropna(
        subset=["week", "admin1", "event_type"]
    ).copy()

    df["admin1"] = (
        df["admin1"].astype(str).str.strip()
    )

    df["event_type"] = (
        df["event_type"].astype(str).str.strip()
    )

    df = df[
        ~df["admin1"].str.lower().isin(
            ["", "nan", "unknown"]
        )
    ].copy()

    # Focus on protest and riot event categories.
    df = df[
        df["event_type"].str.lower().isin(
            ["protests", "protest", "riots", "riot"]
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No protest or riot records found."
        )

    # ACLED week dates are expected to be spaced
    # exactly seven days apart.
    weekly = (
        df.groupby(
            ["admin1", "week"],
            as_index=False
        )["events"]
        .sum()
    )

    weekly = weekly.rename(
        columns={
            "admin1": "state",
            "events": "event_count"
        }
    )

    print("States:", weekly["state"].nunique())
    print("First week:", weekly["week"].min())
    print("Last week:", weekly["week"].max())

    return weekly


# ============================================================
# STEP 2 - CREATE STATE-WEEK PANEL
# ============================================================

def create_panel(weekly):

    print("\n[2/7] Creating state-week panel...")

    states = sorted(
        weekly["state"].unique()
    )

    first_week = weekly["week"].min()
    last_week = weekly["week"].max()

    weeks = pd.date_range(
        start=first_week,
        end=last_week,
        freq="7D"
    )

    observed_weeks = pd.DatetimeIndex(
        weekly["week"].unique()
    ).sort_values()

    if not observed_weeks.isin(weeks).all():
        raise ValueError(
            "ACLED week dates are not aligned "
            "to a consistent 7-day grid."
        )

    panel_index = pd.MultiIndex.from_product(
        [states, weeks],
        names=["state", "week"]
    )

    panel = (
        weekly.set_index(
            ["state", "week"]
        )
        .reindex(
            panel_index,
            fill_value=0
        )
        .reset_index()
    )

    panel["event_count"] = (
        panel["event_count"].astype(int)
    )

    panel = panel.sort_values(
        ["state", "week"]
    ).reset_index(drop=True)

    print("Panel rows:", len(panel))
    print("States:", len(states))
    print("Weeks:", len(weeks))

    print(
        "Note: Missing state-week combinations "
        "are treated as zero recorded events."
    )

    return panel


# ============================================================
# STEP 3 - CREATE LEAKAGE-SAFE FEATURES
# ============================================================

def create_features(panel):

    print("\n[3/7] Creating historical features...")

    df = panel.copy()

    grouped = df.groupby(
        "state",
        sort=False
    )["event_count"]

    df["count_current"] = df["event_count"]

    for lag in [1, 2, 3]:
        df[f"count_lag_{lag}"] = (
            grouped.shift(lag)
        )

    # Rolling windows include only the current
    # and previous weeks.
    #
    # At prediction time, the current week's
    # observed count is assumed available.

    df["rolling_mean_3"] = (
        df.groupby("state")["event_count"]
        .transform(
            lambda s: s.rolling(
                3, min_periods=3
            ).mean()
        )
    )

    df["rolling_mean_4"] = (
        df.groupby("state")["event_count"]
        .transform(
            lambda s: s.rolling(
                4, min_periods=4
            ).mean()
        )
    )

    df["rolling_max_4"] = (
        df.groupby("state")["event_count"]
        .transform(
            lambda s: s.rolling(
                4, min_periods=4
            ).max()
        )
    )

    df["rolling_std_4"] = (
        df.groupby("state")["event_count"]
        .transform(
            lambda s: s.rolling(
                4, min_periods=4
            ).std(ddof=0)
        )
    )

    df["change_1_week"] = (
        df["count_current"]
        - df["count_lag_1"]
    )

    df["active_weeks_4"] = (
        df.groupby("state")["event_count"]
        .transform(
            lambda s: (s > 0).astype(int)
            .rolling(4, min_periods=4)
            .sum()
        )
    )

    df["zero_weeks_4"] = (
        4 - df["active_weeks_4"]
    )

    # National count for week t is known at
    # the end of week t.
    national = (
        df.groupby("week")["event_count"]
        .sum()
        .sort_index()
    )

    national_change = national.diff()

    df["national_count_current"] = (
        df["week"].map(national)
    )

    df["national_change_1_week"] = (
        df["week"].map(national_change)
    )

    # Calendar features, not target information.
    week_number = (
        df["week"].dt.isocalendar()
        .week.astype(int)
    )

    df["week_sin"] = np.sin(
        2 * np.pi * week_number / 52.18
    )

    df["week_cos"] = np.cos(
        2 * np.pi * week_number / 52.18
    )

    # Predict week t+1 using information
    # available through the end of week t.
    df["target_count"] = (
        grouped.shift(-1)
    )

    df["target_week"] = (
        df["week"] + pd.Timedelta(days=7)
    )

    # Emergence is evaluated only when
    # the current state-week count is zero.
    df["emergence_eligible"] = (
        df["count_current"] == 0
    )

    df["target_emergence"] = (
        (
            df["count_current"] == 0
        )
        &
        (
            df["target_count"] > 0
        )
    ).astype(int)

    # Escalation: next-week count exceeds
    # both absolute and relative thresholds.
    #
    # The absolute threshold prevents a
    # tiny 0 -> 1 change being escalation.
    df["target_escalation"] = (
        (
            df["target_count"]
            - df["count_current"]
            >= ESCALATION_MIN_INCREASE
        )
        &
        (
            df["target_count"]
            >= df["count_current"]
            * (
                1 + ESCALATION_RELATIVE_INCREASE
            )
        )
    ).astype(int)

    # Drop incomplete feature histories and
    # final week without a known next-week target.
    df = df.dropna(
        subset=FEATURE_COLUMNS + ["target_count"]
    ).copy()

    df["target_count"] = (
        df["target_count"].astype(int)
    )

    df = df[
        df.groupby("state").cumcount() >= 0
    ].copy()

    df = df.sort_values(
        ["target_week", "state"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No training examples after feature creation."
        )

    print("Forecast examples:", len(df))
    print(
        "Emergence-positive examples:",
        df["target_emergence"].sum()
    )
    print(
        "Escalation-positive examples:",
        df["target_escalation"].sum()
    )

    return df


# ============================================================
# STEP 4 - CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(df):

    print("\n[4/7] Chronological train/test split...")

    target_weeks = sorted(
        df["target_week"].unique()
    )

    n_test = max(
        1,
        int(np.ceil(
            len(target_weeks) * TEST_FRACTION
        ))
    )

    if len(target_weeks) - n_test < 8:
        raise ValueError(
            "Insufficient historical weeks for "
            "a meaningful train/test split."
        )

    test_start = target_weeks[-n_test]

    train = df[
        df["target_week"] < test_start
    ].copy()

    test = df[
        df["target_week"] >= test_start
    ].copy()

    assert (
        train["target_week"].max()
        < test["target_week"].min()
    )

    # All states in the same target week
    # belong to the same split.
    print("Training rows:", len(train))
    print("Testing rows:", len(test))
    print(
        "Training target weeks:",
        train["target_week"].nunique()
    )
    print(
        "Testing target weeks:",
        test["target_week"].nunique()
    )
    print(
        "Test begins:",
        test["target_week"].min()
    )

    return train, test


# ============================================================
# STEP 5 - EVENT COUNT FORECASTING
# ============================================================

def train_count_model(train, test):

    print("\n[5/7] Training event-count model...")

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    y_train = train["target_count"]
    y_test = test["target_count"]

    # Baseline: next week = current week.
    baseline = (
        test["count_current"].to_numpy()
    )

    # Second baseline: mean of last 4 weeks.
    rolling_baseline = (
        test["rolling_mean_4"].to_numpy()
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=4,
        max_features=0.8,
        random_state=SEED,
        n_jobs=2
    )

    model.fit(X_train, y_train)

    predictions = np.maximum(
        0,
        model.predict(X_test)
    )

    results = {
        "persistence_baseline": regression_metrics(
            y_test, baseline
        ),
        "rolling_mean_baseline": regression_metrics(
            y_test, rolling_baseline
        ),
        "random_forest": regression_metrics(
            y_test, predictions
        )
    }

    output = test[
        [
            "state",
            "week",
            "target_week",
            "count_current",
            "target_count"
        ]
    ].copy()

    output["baseline_persistence"] = baseline
    output["baseline_rolling_mean"] = (
        rolling_baseline
    )
    output["prediction_random_forest"] = (
        predictions
    )

    output.to_csv(
        OUTPUT_DIR / "state_count_predictions.csv",
        index=False
    )

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    print("Event-count prediction metrics:")

    for name, metrics in results.items():
        print(
            f"  {name}: "
            f"MAE={metrics['mae']:.3f}, "
            f"RMSE={metrics['rmse']:.3f}"
        )

    return results, output


# ============================================================
# STEP 6 - CLASSIFICATION
# ============================================================

def train_binary_task(
    train,
    test,
    target_column,
    task_name
):

    print(f"\nTraining {task_name} classifier...")

    y_train = train[target_column].astype(int)
    y_test = test[target_column].astype(int)

    # Baseline: predict the most common
    # class in training.
    majority_class = int(
        y_train.value_counts().idxmax()
    )

    baseline_pred = np.full(
        len(test),
        majority_class,
        dtype=int
    )

    baseline_metrics = classification_metrics(
        y_test,
        baseline_pred
    )

    # A single-class training set cannot train
    # a meaningful binary classifier.
    if y_train.nunique() < 2:

        print(
            f"WARNING: {task_name} training data "
            "contains only one class."
        )

        model_pred = baseline_pred.copy()
        scores = np.full(
            len(test),
            float(majority_class)
        )

        model_status = "majority_fallback"

    else:

        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=7,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=2
        )

        model.fit(
            train[FEATURE_COLUMNS],
            y_train
        )

        model_pred = model.predict(
            test[FEATURE_COLUMNS]
        )

        probability = model.predict_proba(
            test[FEATURE_COLUMNS]
        )

        positive_index = list(
            model.classes_
        ).index(1)

        scores = probability[
            :, positive_index
        ]

        model_status = "trained"

    model_metrics = classification_metrics(
        y_test,
        model_pred,
        scores
    )

    output = test[
        ["state", "week", "target_week"]
    ].copy()

    output["actual"] = y_test.to_numpy()
    output["baseline_prediction"] = (
        baseline_pred
    )
    output["model_prediction"] = model_pred
    output["model_score"] = scores

    output.to_csv(
        OUTPUT_DIR / f"{task_name}_predictions.csv",
        index=False
    )

    print(
        f"{task_name}: "
        f"actual positives={int(y_test.sum())}, "
        f"predicted positives={int(model_pred.sum())}"
    )

    print(
        classification_report(
            y_test,
            model_pred,
            labels=[0, 1],
            zero_division=0
        )
    )

    return {
        "model_status": model_status,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive": int(y_train.sum()),
        "test_positive": int(y_test.sum()),
        "majority_baseline": baseline_metrics,
        "random_forest": model_metrics
    }


# ============================================================
# STEP 7 - VISUALIZATIONS
# ============================================================

def create_plots(count_predictions):

    print("\n[7/7] Creating plots...")

    weekly = (
        count_predictions.groupby(
            "target_week",
            as_index=False
        )[
            [
                "target_count",
                "baseline_persistence",
                "prediction_random_forest"
            ]
        ]
        .sum()
    )

    plt.figure(figsize=(11, 5))

    plt.plot(
        weekly["target_week"],
        weekly["target_count"],
        marker="o",
        label="Actual"
    )

    plt.plot(
        weekly["target_week"],
        weekly["baseline_persistence"],
        marker="s",
        label="Persistence baseline"
    )

    plt.plot(
        weekly["target_week"],
        weekly["prediction_random_forest"],
        marker="^",
        label="Random forest"
    )

    plt.title(
        "Historical Test Period: National Total "
        "of State-Level Forecasts"
    )
    plt.xlabel("Target week")
    plt.ylabel("Protest and riot events")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "national_aggregate_backtest.png",
        dpi=160
    )
    plt.close()

    state_error = count_predictions.copy()

    state_error["absolute_error"] = (
        state_error["target_count"]
        - state_error["prediction_random_forest"]
    ).abs()

    state_mae = (
        state_error.groupby("state")[
            "absolute_error"
        ]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .sort_values()
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        state_mae.index,
        state_mae.values
    )

    plt.xlabel("Mean absolute error")
    plt.title(
        "State-Level Forecast Error "
        "(15 Highest-MAE States)"
    )
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "state_forecast_errors.png",
        dpi=160
    )
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "STAGE 26 - STATE-LEVEL CIVIL UNREST PREDICTION"
    )
    print("=" * 72)

    weekly = load_acled()

    panel = create_panel(weekly)

    features = create_features(panel)

    features.to_csv(
        OUTPUT_DIR / "state_week_features.csv",
        index=False
    )

    train, test = chronological_split(features)

    count_results, count_predictions = (
        train_count_model(train, test)
    )

    print("\n[6/7] Evaluating emergence and escalation...")

    # Emergence is only defined for state-weeks
    # with zero events in the current week.
    emergence_train = train[
        train["emergence_eligible"]
    ].copy()

    emergence_test = test[
        test["emergence_eligible"]
    ].copy()

    if (
        len(emergence_train) > 0
        and len(emergence_test) > 0
    ):

        emergence_results = train_binary_task(
            emergence_train,
            emergence_test,
            "target_emergence",
            "emergence"
        )

    else:

        print(
            "Emergence evaluation unavailable: "
            "no eligible rows in train or test."
        )

        emergence_results = {
            "model_status": "insufficient_data"
        }

    escalation_results = train_binary_task(
        train,
        test,
        "target_escalation",
        "escalation"
    )

    create_plots(count_predictions)

    summary = {
        "stage": 26,
        "dataset": str(INPUT_FILE),
        "prediction_horizon": "next_week",
        "historical_backtest": True,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_target_weeks": int(
            train["target_week"].nunique()
        ),
        "test_target_weeks": int(
            test["target_week"].nunique()
        ),
        "test_start": str(
            test["target_week"].min().date()
        ),
        "event_count_forecasting": count_results,
        "emergence_classification": (
            emergence_results
        ),
        "escalation_classification": (
            escalation_results
        ),
        "escalation_definition": {
            "minimum_absolute_increase": (
                ESCALATION_MIN_INCREASE
            ),
            "minimum_relative_increase": (
                ESCALATION_RELATIVE_INCREASE
            )
        },
        "limitations": [
            "Short historical period",
            "Historical ACLED aggregates only",
            "Missing state-week cells assumed zero",
            "No independent social-media validation",
            "Single chronological holdout",
            "Not a real-time unrest prediction system"
        ]
    }

    save_json(
        summary,
        "stage26_evaluation_summary.json"
    )

    print("\n" + "=" * 72)
    print("STAGE 26 - COMPLETED")
    print("=" * 72)

    print("Outputs saved to:")
    print(OUTPUT_DIR)

    print("\nMain metrics:")

    for model_name, metrics in (
        count_results.items()
    ):
        print(
            f"{model_name}: "
            f"MAE={metrics['mae']:.3f}, "
            f"RMSE={metrics['rmse']:.3f}"
        )

    print("\nClassification F1 scores:")

    for name, result in [
        ("Emergence", emergence_results),
        ("Escalation", escalation_results)
    ]:

        if "random_forest" in result:

            print(
                f"{name}: "
                f"{result['random_forest']['f1']:.3f}"
            )

        else:
            print(f"{name}: unavailable")

    print("\nNo previous project stages were modified.")


if __name__ == "__main__":
    main()
