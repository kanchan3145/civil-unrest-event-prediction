

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# STAGE 26 V2
# ROLLING-ORIGIN REGIONAL FORECAST EVALUATION
#
# Uses previously saved Stage 26 features.
# Does not rerun earlier stages.
# ============================================================

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "outputs" / "stage26"
    / "state_week_features.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "stage26_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Three expanding-window evaluations.
# Each evaluates eight consecutive target weeks.
INITIAL_TRAIN_WEEKS = 33
TEST_WEEKS_PER_FOLD = 8
MAX_FOLDS = 3

# Fixed classification threshold.
# We do NOT optimize this using test labels.
CLASSIFICATION_THRESHOLD = 0.50

# Historical rate smoothing.
# Helps states with very few positive examples.
PRIOR_STRENGTH = 10.0

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
    "week_cos"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_csv(df, filename):
    df.to_csv(
        OUTPUT_DIR / filename,
        index=False,
        encoding="utf-8-sig"
    )


def save_json(obj, filename):
    (OUTPUT_DIR / filename).write_text(
        json.dumps(
            obj,
            indent=2,
            default=str,
            allow_nan=False
        ),
        encoding="utf-8"
    )


def regression_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0:
        return {"mae": None, "rmse": None}

    return {
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        )
    }


def classification_metrics(actual, predicted, scores):

    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(actual) == 0:
        return {
            "rows": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "average_precision": None,
            "roc_auc": None
        }

    has_both_classes = len(np.unique(actual)) == 2
    has_positives = int(actual.sum()) > 0

    return {
        "rows": int(len(actual)),
        "actual_positives": int(actual.sum()),
        "predicted_positives": int(predicted.sum()),
        "precision": float(
            precision_score(
                actual, predicted,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                actual, predicted,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                actual, predicted,
                zero_division=0
            )
        ),
        "average_precision": (
            float(
                average_precision_score(
                    actual, scores
                )
            )
            if has_positives else None
        ),
        "roc_auc": (
            float(roc_auc_score(actual, scores))
            if has_both_classes else None
        ),
        "confusion_matrix": (
            confusion_matrix(
                actual, predicted,
                labels=[0, 1]
            ).tolist()
        )
    }


# ============================================================
# 1. LOAD SAVED STAGE 26 FEATURES
# ============================================================

def load_features():

    print("\n[1/7] Loading Stage 26 features...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file:\n{INPUT_FILE}\n"
            "Run Stage 26 first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    required = [
        "state",
        "week",
        "target_week",
        "count_current",
        "target_count",
        "target_emergence",
        "target_escalation",
        "emergence_eligible"
    ] + FEATURE_COLUMNS

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["week"] = pd.to_datetime(
        df["week"],
        errors="raise"
    )

    df["target_week"] = pd.to_datetime(
        df["target_week"],
        errors="raise"
    )

    if df[required].isna().any().any():
        raise ValueError(
            "Input contains missing required values."
        )

    if not (
        df["target_week"]
        == df["week"] + pd.Timedelta(days=7)
    ).all():
        raise ValueError(
            "Unexpected prediction horizon."
        )

    if df.duplicated(
        ["state", "target_week"]
    ).any():
        raise ValueError(
            "Duplicate state-target_week records."
        )

    if (
        df["target_emergence"].isin([0, 1]).all()
        is False
    ):
        raise ValueError(
            "Emergence labels must be binary."
        )

    if not df["target_escalation"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "Escalation labels must be binary."
        )

    df["emergence_eligible"] = (
        df["emergence_eligible"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    df = df.sort_values(
        ["target_week", "state"]
    ).reset_index(drop=True)

    print("Rows:", len(df))
    print("States:", df["state"].nunique())
    print(
        "Target weeks:",
        df["target_week"].nunique()
    )

    return df


# ============================================================
# 2. CREATE ROLLING-ORIGIN SPLITS
# ============================================================

def create_folds(df):

    print("\n[2/7] Creating chronological folds...")

    weeks = sorted(
        df["target_week"].unique()
    )

    folds = []

    for fold_number in range(MAX_FOLDS):

        train_end = (
            INITIAL_TRAIN_WEEKS
            + fold_number * TEST_WEEKS_PER_FOLD
        )

        test_end = (
            train_end + TEST_WEEKS_PER_FOLD
        )

        if test_end > len(weeks):
            break

        train_weeks = weeks[:train_end]
        test_weeks = weeks[train_end:test_end]

        train = df[
            df["target_week"].isin(train_weeks)
        ].copy()

        test = df[
            df["target_week"].isin(test_weeks)
        ].copy()

        assert (
            train["target_week"].max()
            < test["target_week"].min()
        )

        # Features for target week t are
        # observed only through week t-1.
        assert (
            train["target_week"].max()
            <= test["week"].min()
        )

        folds.append((
            fold_number + 1,
            train,
            test
        ))

        print(
            f"Fold {fold_number + 1}: "
            f"train={len(train)} rows, "
            f"test={len(test)} rows, "
            f"test weeks="
            f"{len(test_weeks)}"
        )

    if not folds:
        raise ValueError(
            "Not enough target weeks for backtesting."
        )

    return folds


# ============================================================
# 3. REGRESSION EVALUATION
# ============================================================

def evaluate_counts(folds):

    print("\n[3/7] Evaluating event-count forecasts...")

    outputs = []

    for fold_id, train, test in folds:

        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=4,
            max_features=0.8,
            random_state=SEED,
            n_jobs=2
        )

        model.fit(
            train[FEATURE_COLUMNS],
            train["target_count"]
        )

        predicted = np.maximum(
            0,
            model.predict(test[FEATURE_COLUMNS])
        )

        output = test[
            [
                "state",
                "week",
                "target_week",
                "count_current",
                "target_count"
            ]
        ].copy()

        output["fold"] = fold_id

        output["persistence"] = (
            test["count_current"].to_numpy()
        )

        output["rolling_mean"] = (
            test["rolling_mean_4"].to_numpy()
        )

        output["random_forest"] = predicted

        outputs.append(output)

    result = pd.concat(
        outputs,
        ignore_index=True
    )

    save_csv(
        result,
        "rolling_count_predictions.csv"
    )

    return result


# ============================================================
# 4. CLASSIFICATION BASELINES
# ============================================================

def historical_rate_baseline(
    train,
    test,
    target_column
):
    """
    Smoothed historical event probability per state.

    Uses training labels only.

    Formula:
      (state positives + k * global rate)
      / (state examples + k)
    """

    global_rate = float(
        train[target_column].mean()
    )

    state_stats = (
        train.groupby("state")[target_column]
        .agg(["sum", "count"])
    )

    scores = []

    for state in test["state"]:

        if state in state_stats.index:

            positives = float(
                state_stats.loc[state, "sum"]
            )

            total = float(
                state_stats.loc[state, "count"]
            )

        else:
            positives = 0.0
            total = 0.0

        probability = (
            positives
            + PRIOR_STRENGTH * global_rate
        ) / (
            total + PRIOR_STRENGTH
        )

        scores.append(probability)

    return np.asarray(scores, dtype=float)


def evaluate_binary(
    folds,
    target_column,
    task_name,
    emergence_only=False
):

    print(f"\nEvaluating {task_name}...")

    outputs = []
    fold_details = []

    for fold_id, train_all, test_all in folds:

        if emergence_only:

            train = train_all[
                train_all["emergence_eligible"]
            ].copy()

            test = test_all[
                test_all["emergence_eligible"]
            ].copy()

        else:
            train = train_all.copy()
            test = test_all.copy()

        if len(test) == 0:
            fold_details.append({
                "fold": fold_id,
                "status": "no_eligible_test_rows"
            })
            continue

        if len(train) == 0:
            fold_details.append({
                "fold": fold_id,
                "status": "no_eligible_training_rows"
            })
            continue

        y_train = (
            train[target_column].astype(int)
        )

        baseline_scores = historical_rate_baseline(
            train,
            test,
            target_column
        )

        baseline_pred = (
            baseline_scores
            >= CLASSIFICATION_THRESHOLD
        ).astype(int)

        # If training contains only one class,
        # use a constant model prediction.
        if y_train.nunique() < 2:

            model_scores = np.full(
                len(test),
                float(y_train.iloc[0])
            )

            model_status = "single_class_fallback"

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

            probabilities = model.predict_proba(
                test[FEATURE_COLUMNS]
            )

            positive_index = list(
                model.classes_
            ).index(1)

            model_scores = probabilities[
                :, positive_index
            ]

            model_status = "trained"

        model_pred = (
            model_scores
            >= CLASSIFICATION_THRESHOLD
        ).astype(int)

        output = test[
            [
                "state",
                "week",
                "target_week",
                target_column
            ]
        ].copy()

        output = output.rename(
            columns={target_column: "actual"}
        )

        output["fold"] = fold_id
        output["baseline_score"] = baseline_scores
        output["baseline_pred"] = baseline_pred
        output["model_score"] = model_scores
        output["model_pred"] = model_pred

        outputs.append(output)

        fold_details.append({
            "fold": fold_id,
            "status": model_status,
            "training_rows": int(len(train)),
            "testing_rows": int(len(test)),
            "training_positives": int(
                y_train.sum()
            ),
            "testing_positives": int(
                test[target_column].sum()
            )
        })

        print(
            f"  Fold {fold_id}: "
            f"train={len(train)}, "
            f"test={len(test)}, "
            f"positives={int(test[target_column].sum())}"
        )

    columns = [
        "state",
        "week",
        "target_week",
        "actual",
        "fold",
        "baseline_score",
        "baseline_pred",
        "model_score",
        "model_pred"
    ]

    if outputs:
        result = pd.concat(
            outputs,
            ignore_index=True
        )
    else:
        result = pd.DataFrame(columns=columns)

    save_csv(
        result,
        f"{task_name}_rolling_predictions.csv"
    )

    return result, fold_details


# ============================================================
# 5. FOLD-LEVEL AND OVERALL METRICS
# ============================================================

def calculate_count_metrics(predictions):

    records = []

    for fold_id, subset in predictions.groupby(
        "fold"
    ):

        for model_name in [
            "persistence",
            "rolling_mean",
            "random_forest"
        ]:

            metrics = regression_metrics(
                subset["target_count"],
                subset[model_name]
            )

            records.append({
                "fold": int(fold_id),
                "model": model_name,
                "rows": int(len(subset)),
                **metrics
            })

    for model_name in [
        "persistence",
        "rolling_mean",
        "random_forest"
    ]:

        metrics = regression_metrics(
            predictions["target_count"],
            predictions[model_name]
        )

        records.append({
            "fold": "overall",
            "model": model_name,
            "rows": int(len(predictions)),
            **metrics
        })

    return pd.DataFrame(records)


def calculate_binary_metrics(
    predictions,
    task_name
):

    records = []

    if predictions.empty:
        return pd.DataFrame()

    groups = [
        (str(fold_id), subset)
        for fold_id, subset in predictions.groupby(
            "fold"
        )
    ]

    groups.append(
        ("overall", predictions)
    )

    for fold_id, subset in groups:

        for model_name, pred_col, score_col in [
            (
                "historical_rate",
                "baseline_pred",
                "baseline_score"
            ),
            (
                "random_forest",
                "model_pred",
                "model_score"
            )
        ]:

            metrics = classification_metrics(
                subset["actual"],
                subset[pred_col],
                subset[score_col]
            )

            records.append({
                "task": task_name,
                "fold": fold_id,
                "model": model_name,
                **{
                    key: val
                    for key, val in metrics.items()
                    if key != "confusion_matrix"
                }
            })

    return pd.DataFrame(records)


# ============================================================
# 6. STATE-LEVEL METRICS
# ============================================================

def state_count_metrics(predictions):

    records = []

    for state, subset in predictions.groupby(
        "state"
    ):

        for model_name in [
            "persistence",
            "rolling_mean",
            "random_forest"
        ]:

            metrics = regression_metrics(
                subset["target_count"],
                subset[model_name]
            )

            records.append({
                "state": state,
                "model": model_name,
                "test_rows": int(len(subset)),
                **metrics
            })

    return pd.DataFrame(records)


def state_binary_metrics(
    predictions,
    task_name
):

    records = []

    if predictions.empty:
        return pd.DataFrame()

    for state, subset in predictions.groupby(
        "state"
    ):

        for model_name, pred_col, score_col in [
            (
                "historical_rate",
                "baseline_pred",
                "baseline_score"
            ),
            (
                "random_forest",
                "model_pred",
                "model_score"
            )
        ]:

            metrics = classification_metrics(
                subset["actual"],
                subset[pred_col],
                subset[score_col]
            )

            records.append({
                "state": state,
                "task": task_name,
                "model": model_name,
                **{
                    key: val
                    for key, val in metrics.items()
                    if key != "confusion_matrix"
                }
            })

    return pd.DataFrame(records)


# ============================================================
# 7. ESCALATION PRECISION-RECALL ANALYSIS
# ============================================================

def escalation_pr_analysis(predictions):

    print("\n[6/7] Escalation precision-recall analysis...")

    if predictions.empty:
        return

    actual = predictions["actual"].to_numpy(
        dtype=int
    )

    scores = predictions["model_score"].to_numpy(
        dtype=float
    )

    # PR curve is descriptive evaluation.
    # Test labels are NOT used to choose the
    # operational threshold.
    if actual.sum() > 0:

        precision, recall, thresholds = (
            precision_recall_curve(
                actual,
                scores
            )
        )

        curve = pd.DataFrame({
            "precision": precision,
            "recall": recall,
            "threshold": np.append(
                thresholds,
                np.nan
            )
        })

        save_csv(
            curve,
            "escalation_precision_recall_curve.csv"
        )

        plt.figure(figsize=(8, 5))

        plt.plot(
            recall,
            precision,
            label="Random forest"
        )

        prevalence = float(actual.mean())

        plt.axhline(
            prevalence,
            linestyle="--",
            label="Positive prevalence"
        )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            "Escalation Precision-Recall "
            "(Pooled Historical Backtest)"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "escalation_pr_curve.png",
            dpi=160
        )
        plt.close()

    # Fixed thresholds are displayed to
    # understand sensitivity, not to select
    # the best threshold on test data.
    threshold_rows = []

    for threshold in [
        0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80
    ]:

        predicted = (
            scores >= threshold
        ).astype(int)

        metrics = classification_metrics(
            actual,
            predicted,
            scores
        )

        threshold_rows.append({
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "predicted_positives": (
                metrics["predicted_positives"]
            ),
            "actual_positives": (
                metrics["actual_positives"]
            )
        })

    save_csv(
        pd.DataFrame(threshold_rows),
        "escalation_threshold_sensitivity.csv"
    )


# ============================================================
# VISUALIZATION
# ============================================================

def create_plots(count_predictions, count_metrics):

    print("\n[7/7] Creating evaluation plots...")

    overall = count_metrics[
        count_metrics["fold"] == "overall"
    ].copy()

    plt.figure(figsize=(9, 5))

    plt.bar(
        overall["model"],
        overall["mae"]
    )

    plt.ylabel("Mean absolute error")
    plt.title(
        "State-Level Count Forecasting: "
        "Rolling Backtest"
    )
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "count_model_comparison.png",
        dpi=160
    )
    plt.close()

    weekly = (
        count_predictions.groupby(
            "target_week",
            as_index=False
        )[
            [
                "target_count",
                "persistence",
                "rolling_mean",
                "random_forest"
            ]
        ]
        .sum()
    )

    plt.figure(figsize=(11, 5))

    for column in [
        "target_count",
        "rolling_mean",
        "random_forest"
    ]:

        plt.plot(
            weekly["target_week"],
            weekly[column],
            label=column
        )

    plt.title(
        "National Aggregate of State-Level "
        "Historical Forecasts"
    )
    plt.xlabel("Target week")
    plt.ylabel("Events")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "rolling_national_forecasts.png",
        dpi=160
    )
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "STAGE 26 V2 - ROLLING-ORIGIN REGIONAL EVALUATION"
    )
    print("=" * 72)

    features = load_features()

    folds = create_folds(features)

    count_predictions = evaluate_counts(folds)

    print("\n[4/7] Evaluating emergence...")

    emergence_predictions, emergence_details = (
        evaluate_binary(
            folds,
            "target_emergence",
            "emergence",
            emergence_only=True
        )
    )

    print("\n[5/7] Evaluating escalation...")

    escalation_predictions, escalation_details = (
        evaluate_binary(
            folds,
            "target_escalation",
            "escalation",
            emergence_only=False
        )
    )

    # Fold and pooled metrics.
    count_metrics = calculate_count_metrics(
        count_predictions
    )

    emergence_metrics = calculate_binary_metrics(
        emergence_predictions,
        "emergence"
    )

    escalation_metrics = calculate_binary_metrics(
        escalation_predictions,
        "escalation"
    )

    save_csv(
        count_metrics,
        "count_backtest_metrics.csv"
    )

    save_csv(
        emergence_metrics,
        "emergence_backtest_metrics.csv"
    )

    save_csv(
        escalation_metrics,
        "escalation_backtest_metrics.csv"
    )

    # State-level performance.
    save_csv(
        state_count_metrics(count_predictions),
        "state_count_metrics.csv"
    )

    save_csv(
        state_binary_metrics(
            emergence_predictions,
            "emergence"
        ),
        "state_emergence_metrics.csv"
    )

    save_csv(
        state_binary_metrics(
            escalation_predictions,
            "escalation"
        ),
        "state_escalation_metrics.csv"
    )

    escalation_pr_analysis(
        escalation_predictions
    )

    create_plots(
        count_predictions,
        count_metrics
    )

    # Compact overall evaluation summary.
    overall_count = count_metrics[
        count_metrics["fold"] == "overall"
    ]

    overall_emergence = (
        emergence_metrics[
            emergence_metrics["fold"] == "overall"
        ]
        if not emergence_metrics.empty
        else pd.DataFrame()
    )

    overall_escalation = (
        escalation_metrics[
            escalation_metrics["fold"] == "overall"
        ]
        if not escalation_metrics.empty
        else pd.DataFrame()
    )

    summary = {
        "stage": "26_v2",
        "method": "expanding_window_rolling_origin",
        "number_of_folds": len(folds),
        "initial_training_weeks": (
            INITIAL_TRAIN_WEEKS
        ),
        "test_weeks_per_fold": (
            TEST_WEEKS_PER_FOLD
        ),
        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),
        "count_metrics": (
            overall_count.to_dict(
                orient="records"
            )
        ),
        "emergence_metrics": (
            overall_emergence.to_dict(
                orient="records"
            )
        ),
        "escalation_metrics": (
            overall_escalation.to_dict(
                orient="records"
            )
        ),
        "emergence_fold_details": (
            emergence_details
        ),
        "escalation_fold_details": (
            escalation_details
        ),
        "limitations": [
            "Only 61 historical weeks in source data",
            "State-week observations are not independent",
            "Missing state-weeks were zero-filled in Stage 26",
            "Historical event labels are not current forecasts",
            "Escalation thresholds are project-defined",
            "PR threshold sweep is descriptive, not tuning",
            "Per-state classification metrics may be unstable",
            "Random forest scores are not calibrated probabilities",
            "No independent incident-level social-media validation"
        ]
    }

    save_json(
        summary,
        "stage26_v2_summary.json"
    )

    print("\n" + "=" * 72)
    print("STAGE 26 V2 - COMPLETED")
    print("=" * 72)

    print("\nCOUNT FORECASTING:")

    for _, row in overall_count.iterrows():

        print(
            f"{row['model']}: "
            f"MAE={row['mae']:.3f}, "
            f"RMSE={row['rmse']:.3f}"
        )

    print("\nEMERGENCE CLASSIFICATION:")

    for _, row in overall_emergence.iterrows():

        print(
            f"{row['model']}: "
            f"Precision={row['precision']:.3f}, "
            f"Recall={row['recall']:.3f}, "
            f"F1={row['f1']:.3f}, "
            f"AP={row['average_precision']}"
        )

    print("\nESCALATION CLASSIFICATION:")

    for _, row in overall_escalation.iterrows():

        print(
            f"{row['model']}: "
            f"Precision={row['precision']:.3f}, "
            f"Recall={row['recall']:.3f}, "
            f"F1={row['f1']:.3f}, "
            f"AP={row['average_precision']}"
        )

    print("\nFiles saved to:")
    print(OUTPUT_DIR)

    print(
        "\nEarlier stages and outputs "
        "remain unchanged."
    )


if __name__ == "__main__":
    main()
