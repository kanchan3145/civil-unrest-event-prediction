
from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
)

# ============================================================
# STAGE 28
# FINAL MODEL SELECTION AND ERROR ANALYSIS
#
# INPUTS:
#   outputs/stage26_v2/
#   outputs/stage27/
#
# OUTPUT:
#   outputs/stage28/
#
# No model training.
# No tweet processing.
# No modification of earlier stages.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

STAGE26 = ROOT / "outputs" / "stage26_v2"
STAGE27 = ROOT / "outputs" / "stage27"
OUT = ROOT / "outputs" / "stage28"

OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50

COUNT_MODELS = [
    "persistence",
    "rolling_mean",
    "acled_only",
    "acled_plus_social",
]

CLASS_MODELS = [
    "historical_rate",
    "acled_only",
    "acled_plus_social",
]

TASKS = ["count", "emergence", "escalation"]

DISPLAY_NAMES = {
    "persistence": "Persistence",
    "rolling_mean": "Rolling mean",
    "historical_rate": "Historical rate",
    "random_forest": "Random forest",
    "acled_only": "ACLED-only RF",
    "acled_plus_social": "ACLED + social RF",
}

STAGE27_FILES = {
    "count": "count_ablation_predictions.csv",
    "emergence": "emergence_ablation_predictions.csv",
    "escalation": "escalation_ablation_predictions.csv",
}

STAGE26_METRIC_FILES = {
    "count": "count_backtest_metrics.csv",
    "emergence": "emergence_backtest_metrics.csv",
    "escalation": "escalation_backtest_metrics.csv",
}

# ============================================================
# 1. UTILITIES
# ============================================================

def save_csv(df, filename):
    path = OUT / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def clean_for_json(value):
    if isinstance(value, dict):
        return {
            str(k): clean_for_json(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [clean_for_json(v) for v in value]

    if isinstance(value, np.generic):
        return clean_for_json(value.item())

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def save_json(obj, filename):
    path = OUT / filename
    path.write_text(
        json.dumps(
            clean_for_json(obj),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def display_name(model):
    return DISPLAY_NAMES.get(model, model)


def safe_ap(actual, scores):
    actual = np.asarray(actual, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(actual) == 0 or actual.sum() == 0:
        return np.nan

    return float(average_precision_score(actual, scores))


def count_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0:
        return {
            "rows": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
        }

    error = predicted - actual

    return {
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "bias": float(np.mean(error)),
    }


def binary_metrics(actual, scores):
    actual = np.asarray(actual, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(actual) == 0:
        return {
            "rows": 0,
            "actual_positive": 0,
            "predicted_positive": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "average_precision": np.nan,
        }

    predicted = (scores >= THRESHOLD).astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    return {
        "rows": int(len(actual)),
        "actual_positive": int(actual.sum()),
        "predicted_positive": int(predicted.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "average_precision": safe_ap(actual, scores),
    }


def validate_numeric(df, columns, filename):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[columns].isna().any().any():
        bad = df[columns].columns[df[columns].isna().any()].tolist()
        raise ValueError(
            f"{filename} contains missing or nonnumeric values in {bad}"
        )

    values = df[columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{filename} contains infinite numeric values."
        )


# ============================================================
# 2. LOAD STAGE 27 PREDICTIONS
# ============================================================

def load_stage27():
    print("\n[1/9] Loading Stage 27 predictions...")

    data = {}

    for task, filename in STAGE27_FILES.items():
        path = STAGE27 / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Stage 27 predictions:\n{path}\n"
                "Do not rerun earlier stages yet. "
                "Check the filename in outputs/stage27."
            )

        df = pd.read_csv(path)

        models = COUNT_MODELS if task == "count" else CLASS_MODELS

        required = [
            "state",
            "week",
            "target_week",
            "fold",
            "actual",
        ] + models

        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"{filename} is missing columns: {missing}"
            )

        df["week"] = pd.to_datetime(df["week"], errors="raise")
        df["target_week"] = pd.to_datetime(
            df["target_week"], errors="raise"
        )

        if df[["state", "week", "target_week", "fold"]].isna().any().any():
            raise ValueError(
                f"{filename} contains missing identifiers."
            )

        validate_numeric(df, ["actual"] + models, filename)

        if df.duplicated(["state", "target_week"]).any():
            raise ValueError(
                f"{filename} has duplicate state-target_week rows."
            )

        if not (
            df["target_week"] == df["week"] + pd.Timedelta(days=7)
        ).all():
            raise ValueError(
                f"{filename} contains inconsistent forecast dates."
            )

        if task != "count":
            if not df["actual"].isin([0, 1]).all():
                raise ValueError(
                    f"{filename} has nonbinary targets."
                )

            for model in models:
                if not df[model].between(0, 1).all():
                    raise ValueError(
                        f"{filename}: {model} scores must be in [0, 1]."
                    )

        else:
            if (df["actual"] < 0).any():
                raise ValueError(
                    f"{filename} contains negative event counts."
                )

        data[task] = df.sort_values(
            ["target_week", "state"]
        ).reset_index(drop=True)

        print(
            f"{task}: {len(df)} rows, "
            f"{df['state'].nunique()} states, "
            f"{df['fold'].nunique()} folds"
        )

    # Validate the identical evaluation sample for the
    # count and escalation tasks.
    count_keys = set(
        map(
            tuple,
            data["count"][["state", "target_week", "fold"]].to_numpy()
        )
    )

    escalation_keys = set(
        map(
            tuple,
            data["escalation"][["state", "target_week", "fold"]].to_numpy()
        )
    )

    if count_keys != escalation_keys:
        warnings.warn(
            "Count and escalation use different evaluation rows. "
            "They will be analyzed separately."
        )

    return data


# ============================================================
# 3. RECOMPUTE STAGE 27 METRICS
# ============================================================

def evaluate_stage27(data):
    print("\n[2/9] Recomputing Stage 27 metrics...")

    records = []

    for task, df in data.items():
        models = COUNT_MODELS if task == "count" else CLASS_MODELS

        for model in models:
            if task == "count":
                result = count_metrics(df["actual"], df[model])
            else:
                result = binary_metrics(df["actual"], df[model])

            records.append({
                "task": task,
                "model": model,
                **result,
            })

    metrics = pd.DataFrame(records)

    save_csv(metrics, "stage27_recomputed_metrics.csv")

    return metrics


# ============================================================
# 4. LOAD STAGE 26 V2 METRICS
# ============================================================

def load_stage26_metrics():
    print("\n[3/9] Reading Stage 26 V2 metrics...")

    tables = []

    for task, filename in STAGE26_METRIC_FILES.items():
        path = STAGE26 / filename

        if not path.exists():
            print(f"Warning: Missing {path}")
            continue

        df = pd.read_csv(path)

        print(f"{task}: {len(df)} metric rows")
        print("Columns:", df.columns.tolist())

        # FIX: Stage 26 V2 files do not all have
        # identical schemas. Some already have 'task'.
        if "task" not in df.columns:
            df.insert(0, "task", task)
        else:
            df["task"] = task

        if "source" not in df.columns:
            df.insert(1, "source", "stage26_v2")
        else:
            df["source"] = "stage26_v2"

        tables.append(df)

    if not tables:
        print(
            "Warning: No Stage 26 V2 metrics available. "
            "Stage 27 analysis will still continue."
        )
        return pd.DataFrame()

    combined = pd.concat(
        tables,
        ignore_index=True,
        sort=False,
    )

    save_csv(
        combined,
        "stage26_v2_original_metrics.csv",
    )

    print(
        f"Consolidated Stage 26 V2 metrics: {len(combined)} rows"
    )

    return combined


# ============================================================
# 5. DESCRIPTIVE MODEL RANKING
# ============================================================

def select_stage27_models(metrics):
    print("\n[4/9] Ranking Stage 27 models...")

    rankings = []
    winners = []

    for task in TASKS:
        subset = metrics.loc[
            metrics["task"] == task
        ].copy()

        if task == "count":
            subset = subset.sort_values(
                ["mae", "rmse"],
                ascending=[True, True],
            )
            criterion = "lowest_mae"

        else:
            subset = subset.sort_values(
                ["average_precision", "f1"],
                ascending=[False, False],
                na_position="last",
            )
            criterion = "highest_average_precision"

        subset["rank"] = np.arange(1, len(subset) + 1)
        subset["selection_criterion"] = criterion

        rankings.append(subset)

        winner = subset.iloc[0]

        winners.append({
            "task": task,
            "selected_model": winner["model"],
            "criterion": criterion,
            "selection_source": "stage27_historical_backtest",
            "note": (
                "Descriptive test-set ranking; not an "
                "independent validation or deployment decision."
            ),
        })

    ranking_table = pd.concat(
        rankings,
        ignore_index=True,
    )

    selection_table = pd.DataFrame(winners)

    save_csv(
        ranking_table,
        "stage27_full_model_rankings.csv",
    )

    save_csv(
        selection_table,
        "stage27_model_ranking.csv",
    )

    return selection_table


# ============================================================
# 6. SOCIAL-MEDIA ABLATION
# ============================================================

def compare_social_ablation(metrics):
    records = []

    for task in TASKS:
        subset = metrics.loc[
            metrics["task"] == task
        ].set_index("model")

        acled = subset.loc["acled_only"]
        social = subset.loc["acled_plus_social"]

        metric_names = (
            ["mae", "rmse"]
            if task == "count"
            else [
                "precision",
                "recall",
                "f1",
                "average_precision",
            ]
        )

        for metric in metric_names:
            before = float(acled[metric])
            after = float(social[metric])

            if task == "count":
                difference = before - after
                improvement = (
                    100 * difference / before
                    if before != 0
                    else np.nan
                )
                better = after < before
            else:
                difference = after - before
                improvement = np.nan
                better = after > before

            records.append({
                "task": task,
                "metric": metric,
                "acled_only": before,
                "acled_plus_social": after,
                "difference": difference,
                "improvement_percent": improvement,
                "better_with_social": bool(better),
            })

    comparison = pd.DataFrame(records)

    save_csv(
        comparison,
        "social_ablation_comparison.csv",
    )

    return comparison


# ============================================================
# 7. STATE-LEVEL COUNT ERROR ANALYSIS
# ============================================================

def analyze_count_errors(df):
    print("\n[5/9] Analyzing state-level count errors...")

    all_errors = []
    state_rows = []

    keys = [
        "state",
        "week",
        "target_week",
        "fold",
        "actual",
    ]

    for model in COUNT_MODELS:
        temp = df[keys].copy()

        temp["model"] = model
        temp["predicted"] = df[model].to_numpy()

        temp["signed_error"] = (
            temp["predicted"] - temp["actual"]
        )

        temp["absolute_error"] = (
            temp["signed_error"].abs()
        )

        temp["squared_error"] = (
            temp["signed_error"] ** 2
        )

        all_errors.append(temp)

        for state, group in temp.groupby("state"):
            state_rows.append({
                "state": state,
                "model": model,
                **count_metrics(
                    group["actual"],
                    group["predicted"],
                ),
            })

    errors = pd.concat(
        all_errors,
        ignore_index=True,
    )

    state_metrics = pd.DataFrame(state_rows)

    save_csv(
        errors,
        "count_all_prediction_errors.csv",
    )

    save_csv(
        state_metrics,
        "state_count_error_metrics.csv",
    )

    save_csv(
        errors.sort_values(
            "absolute_error",
            ascending=False,
        ).head(50),
        "worst_50_count_errors.csv",
    )

    # Paired state-week comparison between the
    # ACLED-only and ACLED+social random forests.
    comparison = df[keys].copy()

    comparison["acled_abs_error"] = (
        df["acled_only"] - df["actual"]
    ).abs()

    comparison["social_abs_error"] = (
        df["acled_plus_social"] - df["actual"]
    ).abs()

    comparison["social_error_reduction"] = (
        comparison["acled_abs_error"]
        - comparison["social_abs_error"]
    )

    comparison["outcome"] = np.select(
        [
            comparison["social_error_reduction"] > 1e-9,
            comparison["social_error_reduction"] < -1e-9,
        ],
        [
            "social_better",
            "social_worse",
        ],
        default="tie",
    )

    save_csv(
        comparison,
        "count_social_error_comparison.csv",
    )

    state_change = (
        comparison.groupby("state")
        .agg(
            rows=("outcome", "size"),
            social_better_weeks=(
                "outcome",
                lambda x: int((x == "social_better").sum()),
            ),
            social_worse_weeks=(
                "outcome",
                lambda x: int((x == "social_worse").sum()),
            ),
            ties=(
                "outcome",
                lambda x: int((x == "tie").sum()),
            ),
            mean_error_reduction=(
                "social_error_reduction",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_error_reduction",
            ascending=False,
        )
    )

    save_csv(
        state_change,
        "state_social_count_improvement.csv",
    )

    return errors, state_metrics, comparison, state_change


# ============================================================
# 8. FALSE POSITIVE / FALSE NEGATIVE ANALYSIS
# ============================================================

def analyze_classification(df, task):
    print(f"\n[6/9] Analyzing {task} classification...")

    records = []
    state_rows = []

    keys = [
        "state",
        "week",
        "target_week",
        "fold",
        "actual",
    ]

    for model in CLASS_MODELS:
        temp = df[keys].copy()

        temp["model"] = model
        temp["score"] = df[model].to_numpy()

        temp["predicted"] = (
            temp["score"] >= THRESHOLD
        ).astype(int)

        actual = temp["actual"]
        predicted = temp["predicted"]

        temp["error_type"] = np.select(
            [
                (actual == 1) & (predicted == 1),
                (actual == 0) & (predicted == 0),
                (actual == 0) & (predicted == 1),
                (actual == 1) & (predicted == 0),
            ],
            ["TP", "TN", "FP", "FN"],
            default="unknown",
        )

        # A ranking score for error inspection,
        # not a calibrated measure of confidence.
        temp["error_severity"] = np.where(
            temp["error_type"] == "FP",
            temp["score"],
            np.where(
                temp["error_type"] == "FN",
                1.0 - temp["score"],
                0.0,
            ),
        )

        records.append(temp)

        for state, group in temp.groupby("state"):
            state_rows.append({
                "state": state,
                "model": model,
                **binary_metrics(
                    group["actual"],
                    group["score"],
                ),
            })

    outcomes = pd.concat(
        records,
        ignore_index=True,
    )

    state_metrics = pd.DataFrame(state_rows)

    save_csv(
        outcomes,
        f"{task}_all_classification_outcomes.csv",
    )

    save_csv(
        state_metrics,
        f"{task}_state_metrics.csv",
    )

    for error_type in ["FP", "FN"]:
        cases = outcomes.loc[
            outcomes["error_type"] == error_type
        ].sort_values(
            "error_severity",
            ascending=False,
        )

        suffix = error_type.lower()

        save_csv(
            cases,
            f"{task}_{suffix}_cases.csv",
        )

        save_csv(
            cases.head(50),
            f"{task}_top50_{suffix}.csv",
        )

    confusion = (
        outcomes.groupby(["model", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=["TP", "TN", "FP", "FN"],
            fill_value=0,
        )
        .reset_index()
    )

    save_csv(
        confusion,
        f"{task}_confusion_summary.csv",
    )

    return outcomes, state_metrics, confusion


# ============================================================
# 9. FOLD-BY-FOLD STABILITY
# ============================================================

def fold_analysis(data):
    print("\n[7/9] Evaluating fold stability...")

    records = []

    for task, df in data.items():
        models = (
            COUNT_MODELS
            if task == "count"
            else CLASS_MODELS
        )

        for fold, subset in df.groupby("fold"):
            for model in models:
                if task == "count":
                    result = count_metrics(
                        subset["actual"],
                        subset[model],
                    )
                else:
                    result = binary_metrics(
                        subset["actual"],
                        subset[model],
                    )

                records.append({
                    "task": task,
                    "fold": fold,
                    "model": model,
                    **result,
                })

    fold_metrics = pd.DataFrame(records)

    save_csv(
        fold_metrics,
        "fold_stability_metrics.csv",
    )

    return fold_metrics


# ============================================================
# 10. PLOTTING
# ============================================================

def plot_model_metric(metrics, task, metric, filename):
    subset = metrics.loc[
        metrics["task"] == task
    ].copy()

    if subset.empty:
        return

    plt.figure(figsize=(9, 5))

    plt.bar(
        [display_name(m) for m in subset["model"]],
        subset[metric].astype(float),
    )

    plt.ylabel(metric.replace("_", " ").title())
    plt.title(
        f"{task.title()} - {metric.replace('_', ' ').title()}"
    )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(OUT / filename, dpi=180)
    plt.close()


def plot_state_count_errors(state_metrics):
    subset = state_metrics.loc[
        state_metrics["model"] == "rolling_mean"
    ].sort_values(
        "mae",
        ascending=False,
    ).head(15)

    if subset.empty:
        return

    plt.figure(figsize=(10, 7))

    plt.barh(
        subset["state"],
        subset["mae"],
    )

    plt.gca().invert_yaxis()
    plt.xlabel("Mean Absolute Error")
    plt.title("15 States with Largest Rolling-Mean Errors")
    plt.tight_layout()

    plt.savefig(
        OUT / "worst_states_count_mae.png",
        dpi=180,
    )
    plt.close()


def plot_confusion_errors(confusion, task):
    rows = []

    for model in CLASS_MODELS:
        subset = confusion.loc[
            confusion["model"] == model
        ]

        if subset.empty:
            rows.append([0, 0])
        else:
            row = subset.iloc[0]
            rows.append([
                int(row["FP"]),
                int(row["FN"]),
            ])

    values = np.asarray(rows)
    x = np.arange(len(CLASS_MODELS))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - width / 2,
        values[:, 0],
        width,
        label="False positives",
    )

    plt.bar(
        x + width / 2,
        values[:, 1],
        width,
        label="False negatives",
    )

    plt.xticks(
        x,
        [display_name(m) for m in CLASS_MODELS],
        rotation=15,
        ha="right",
    )

    plt.ylabel("Number of Predictions")
    plt.title(f"{task.title()} - FP and FN Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT / f"{task}_fp_fn_comparison.png",
        dpi=180,
    )
    plt.close()


def plot_pr_curves(df, task):
    actual = df["actual"].to_numpy(dtype=int)

    if actual.sum() == 0:
        return

    plt.figure(figsize=(8, 6))

    for model in CLASS_MODELS:
        scores = df[model].to_numpy(dtype=float)

        precision, recall, _ = precision_recall_curve(
            actual,
            scores,
        )

        ap = safe_ap(actual, scores)

        plt.plot(
            recall,
            precision,
            label=f"{display_name(model)} (AP={ap:.3f})",
        )

    plt.axhline(
        actual.mean(),
        linestyle="--",
        label="Positive prevalence",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{task.title()} Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT / f"{task}_precision_recall.png",
        dpi=180,
    )
    plt.close()


def plot_social_state_change(state_change):
    subset = state_change.sort_values(
        "mean_error_reduction",
        ascending=False,
    )

    if subset.empty:
        return

    plt.figure(figsize=(11, 9))

    plt.barh(
        subset["state"],
        subset["mean_error_reduction"],
    )

    plt.axvline(0, linestyle="--")
    plt.xlabel(
        "Mean Absolute Error Reduction "
        "(Positive = Social Better)"
    )
    plt.title(
        "State-Level Effect of Adding Social Signals"
    )
    plt.tight_layout()

    plt.savefig(
        OUT / "state_social_error_reduction.png",
        dpi=180,
    )
    plt.close()


def generate_plots(
    data,
    metrics,
    state_count,
    state_change,
    emergence_confusion,
    escalation_confusion,
):
    print("\n[8/9] Generating report-ready plots...")

    plot_model_metric(
        metrics,
        "count",
        "mae",
        "count_mae_comparison.png",
    )

    plot_model_metric(
        metrics,
        "count",
        "rmse",
        "count_rmse_comparison.png",
    )

    plot_model_metric(
        metrics,
        "emergence",
        "f1",
        "emergence_f1_comparison.png",
    )

    plot_model_metric(
        metrics,
        "emergence",
        "average_precision",
        "emergence_ap_comparison.png",
    )

    plot_model_metric(
        metrics,
        "escalation",
        "f1",
        "escalation_f1_comparison.png",
    )

    plot_model_metric(
        metrics,
        "escalation",
        "average_precision",
        "escalation_ap_comparison.png",
    )

    plot_state_count_errors(state_count)

    plot_social_state_change(state_change)

    plot_confusion_errors(
        emergence_confusion,
        "emergence",
    )

    plot_confusion_errors(
        escalation_confusion,
        "escalation",
    )

    plot_pr_curves(
        data["emergence"],
        "emergence",
    )

    plot_pr_curves(
        data["escalation"],
        "escalation",
    )


# ============================================================
# 11. STAGE 26 V2 SUMMARY (SEPARATE BACKTEST)
# ============================================================

def summarize_stage26(stage26):
    if stage26.empty:
        return pd.DataFrame()

    # Stage 26 V2's saved files contain fold-level
    # metrics. Summarize them without pretending
    # they are pooled prediction-level metrics.
    metric_columns = [
        "mae",
        "rmse",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
    ]

    available = [
        c for c in metric_columns
        if c in stage26.columns
    ]

    for column in available:
        stage26[column] = pd.to_numeric(
            stage26[column],
            errors="coerce",
        )

    if "model" not in stage26.columns:
        print(
            "Warning: Stage 26 V2 has no model column; "
            "skipping model-level summary."
        )
        return pd.DataFrame()

    summary = (
        stage26.groupby(
            ["task", "model"],
            dropna=False,
        )[available]
        .mean()
        .reset_index()
    )

    summary["aggregation_note"] = (
        "Unweighted mean of available fold-level metrics; "
        "not pooled out-of-fold performance."
    )

    save_csv(
        summary,
        "stage26_v2_mean_fold_metrics.csv",
    )

    return summary


# ============================================================
# 12. FINAL REPORT
# ============================================================

def markdown_table(df, columns, decimals=3):
    if df.empty:
        return "_No data available._"

    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"

    rows = [header, separator]

    for _, row in df.iterrows():
        cells = []

        for column in columns:
            value = row.get(column, "")

            if pd.isna(value):
                cells.append("N/A")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.{decimals}f}")
            else:
                cells.append(str(value))

        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def create_report(
    data,
    metrics,
    selections,
    ablation,
    stage26_summary,
    emergence_confusion,
    escalation_confusion,
):
    print("\n[9/9] Creating final research report...")

    lines = [
        "# Stage 28 - Final Model Evaluation",
        "",
        "## Project",
        "",
        "Civil Unrest Event Extraction and Prediction "
        "from Social Media Data",
        "",
        "## Evaluation scope",
        "",
        "This report evaluates historical next-week "
        "state-level civil unrest forecasts.",
        "",
        "Stage 27 uses identical chronological folds "
        "for its ACLED-only and combined models.",
        "",
        "Stage 26 V2 is a separate backtest and must "
        "not be compared numerically as though it "
        "used the same test examples.",
        "",
        "No model training or tweet processing was "
        "performed during Stage 28.",
        "",
        "## Stage 27 - Event count forecasting",
        "",
    ]

    count = metrics.loc[
        metrics["task"] == "count",
        ["model", "mae", "rmse", "bias"],
    ].copy()

    lines.append(
        markdown_table(
            count,
            ["model", "mae", "rmse", "bias"],
        )
    )

    lines += [
        "",
        "Lower MAE and RMSE indicate better "
        "count prediction accuracy.",
        "",
    ]

    for task in ["emergence", "escalation"]:
        lines += [
            f"## Stage 27 - {task.title()} classification",
            "",
        ]

        subset = metrics.loc[
            metrics["task"] == task,
            [
                "model",
                "precision",
                "recall",
                "f1",
                "average_precision",
                "fp",
                "fn",
            ],
        ].copy()

        lines.append(
            markdown_table(
                subset,
                [
                    "model",
                    "precision",
                    "recall",
                    "f1",
                    "average_precision",
                    "fp",
                    "fn",
                ],
            )
        )

        lines.append("")

    lines += [
        "## Social-media ablation",
        "",
        "Positive count error reduction indicates "
        "an improvement after adding social features.",
        "",
    ]

    for _, row in ablation.iterrows():
        if row["task"] == "count":
            lines.append(
                f"- {row['task']} / {row['metric']}: "
                f"{row['improvement_percent']:+.2f}% "
                "relative error reduction."
            )
        else:
            lines.append(
                f"- {row['task']} / {row['metric']}: "
                f"{row['difference']:+.4f} "
                "absolute metric change."
            )

    lines += [
        "",
        "## Descriptive Stage 27 model rankings",
        "",
        markdown_table(
            selections,
            ["task", "selected_model", "criterion"],
        ),
        "",
        "These rankings are based on the historical "
        "evaluation set. They are descriptive and "
        "must not be treated as independent "
        "model validation.",
        "",
        "## Stage 26 V2 - Separate backtest",
        "",
    ]

    if not stage26_summary.empty:
        for task in TASKS:
            subset = stage26_summary.loc[
                stage26_summary["task"] == task
            ]

            if subset.empty:
                continue

            lines += [
                f"### {task.title()}",
                "",
            ]

            if task == "count":
                cols = [
                    "model",
                    "mae",
                    "rmse",
                ]
            else:
                cols = [
                    "model",
                    "f1",
                    "average_precision",
                ]

            lines.append(
                markdown_table(subset, cols)
            )
            lines.append("")

        lines.append(
            "Stage 26 V2 values above are means of "
            "available fold-level metrics, not pooled "
            "out-of-fold predictions."
        )

    else:
        lines.append(
            "Stage 26 V2 summary metrics were unavailable."
        )

    lines += [
        "",
        "## Error analysis",
        "",
        "Detailed state-level errors and individual "
        "false-positive and false-negative examples "
        "are saved in the accompanying CSV files.",
        "",
        "## Important limitations",
        "",
        "- Historical backtesting does not guarantee "
        "future forecasting performance.",
        "- National tweet activity is not a substitute "
        "for verified state-specific event signals.",
        "- Keyword matches and event mentions are "
        "not necessarily unique verified incidents.",
        "- Random-forest classification scores should "
        "not be assumed to be calibrated probabilities.",
        "- Classification metrics use a fixed "
        "threshold of 0.50.",
        "- Some states have relatively few eligible "
        "emergence observations.",
        "- Stage 26 V2 and Stage 27 use different "
        "historical evaluation windows.",
        "- Ranking models on test results does not "
        "constitute independent model selection.",
        "",
        "## Reproducibility",
        "",
        "All Stage 28 calculations use existing saved "
        "Stage 26 V2 and Stage 27 outputs.",
        "",
        "No earlier-stage outputs were modified.",
        "",
    ]

    (OUT / "final_evaluation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    summary = {
        "stage": 28,
        "stage27_rows": {
            task: int(len(df))
            for task, df in data.items()
        },
        "stage27_selections": selections.to_dict(
            orient="records"
        ),
        "stage26_metrics_available": (
            not stage26_summary.empty
        ),
        "classification_threshold": THRESHOLD,
        "no_retraining": True,
        "no_tweet_reprocessing": True,
        "methodological_note": (
            "Stage 26 V2 and Stage 27 are separate "
            "historical evaluations. Stage 27 rankings "
            "are descriptive, not independent validation."
        ),
    }

    save_json(
        summary,
        "stage28_summary.json",
    )


# ============================================================
# 13. MAIN
# ============================================================

def main():
    print("=" * 72)
    print("STAGE 28 - FINAL MODEL AND ERROR ANALYSIS")
    print("=" * 72)

    # 1. Saved Stage 27 predictions
    data = load_stage27()

    # 2. Recompute pooled Stage 27 metrics
    metrics = evaluate_stage27(data)

    # 3. Read saved Stage 26 V2 metrics
    stage26 = load_stage26_metrics()

    # 4. Descriptive Stage 27 model ranking
    selections = select_stage27_models(metrics)

    # 5. Controlled social-media ablation
    ablation = compare_social_ablation(metrics)

    # 6. Count error analysis
    (
        count_errors,
        state_count,
        count_comparison,
        state_change,
    ) = analyze_count_errors(data["count"])

    # 7. Emergence classification errors
    (
        emergence_outcomes,
        emergence_states,
        emergence_confusion,
    ) = analyze_classification(
        data["emergence"],
        "emergence",
    )

    # 8. Escalation classification errors
    (
        escalation_outcomes,
        escalation_states,
        escalation_confusion,
    ) = analyze_classification(
        data["escalation"],
        "escalation",
    )

    # 9. Fold stability
    fold_metrics = fold_analysis(data)

    # 10. Stage 26 V2 fold-level summary
    stage26_summary = summarize_stage26(stage26)

    # 11. Plots
    generate_plots(
        data,
        metrics,
        state_count,
        state_change,
        emergence_confusion,
        escalation_confusion,
    )

    # 12. Report
    create_report(
        data,
        metrics,
        selections,
        ablation,
        stage26_summary,
        emergence_confusion,
        escalation_confusion,
    )

    print("\n" + "=" * 72)
    print("STAGE 28 COMPLETED")
    print("=" * 72)

    print("\nSTAGE 27 MODEL RANKINGS:")

    for _, row in selections.iterrows():
        print(
            f"{row['task']}: "
            f"{row['selected_model']} "
            f"({row['criterion']})"
        )

    print("\nSOCIAL ABLATION:")

    for _, row in ablation.iterrows():
        if row["task"] == "count":
            print(
                f"{row['task']} {row['metric']}: "
                f"{row['improvement_percent']:+.2f}%"
            )

        elif row["metric"] in [
            "f1",
            "average_precision",
        ]:
            print(
                f"{row['task']} {row['metric']}: "
                f"{row['difference']:+.4f}"
            )

    print("\nOUTPUT DIRECTORY:")
    print(OUT)

    print("\nPrevious stages remain unchanged.")


if __name__ == "__main__":
    main()
