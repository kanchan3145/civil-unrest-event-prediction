
from pathlib import Path
import json
import re
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
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# STAGE 27 - CORRECTED SOCIAL-MEDIA ABLATION
#
# CPU friendly
# No raw tweet processing
# No transformer training
# Uses existing Stage 26 features
# ============================================================

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent

ACLED_FILE = (
    ROOT / "outputs" / "stage26"
    / "state_week_features.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "stage27"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If automatic discovery cannot identify the correct
# Stage 14 weekly aggregate, enter its path here.
#
# Example:
# SOCIAL_FILE = ROOT / "outputs" / "stage14" / "weekly.csv"

SOCIAL_FILE = (
    ROOT / "data" / "prediction"
    / "stage14_social_v1_weekly_features.csv"
)

# Do not change dates automatically.
# A mismatch must be resolved from the source
# aggregation's documented week convention.

INITIAL_TRAIN_WEEKS = 30
TEST_WEEKS = 8
MAX_FOLDS = 3

THRESHOLD = 0.50
PRIOR_STRENGTH = 10.0

MAX_SOCIAL_FILE_BYTES = 50_000_000

ACLED_FEATURES = [
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

DATE_NAMES = {
    "week",
    "week_start",
    "week_end",
    "week_ending",
    "week_date",
    "week_start_date",
    "week_end_date"
}

# These are valid names for weekly tweet counts.
# A generic "post_id" is NOT a count.

COUNT_PATTERN = re.compile(
    r"(tweet|post|social|keyword|mention|"
    r"protest|riot|strike|clash|march|"
    r"volume|activity|total)",
    re.IGNORECASE
)

REJECT_COLUMN = re.compile(
    r"(^|_)(id|url|text|content|label|target|"
    r"prediction|predicted|actual|acled|"
    r"fatalities|population|probability|"
    r"score|ratio|percent|percentage|"
    r"future|next|lead|lag|rolling|"
    r"rate|mean|average|std|date|time)"
    r"($|_)",
    re.IGNORECASE
)

REJECT_FILE = re.compile(
    r"(tweets_processed|tweet_level|"
    r"full_event_candidates|"
    r"full_event_predictions|"
    r"enriched_event_candidates|"
    r"raw_tweets|annotations|"
    r"review_sample|"
    r"stage26|stage27)",
    re.IGNORECASE
)

# ============================================================
# UTILITIES
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


def count_metrics(actual, predicted):

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


def binary_metrics(actual, scores):

    actual = np.asarray(actual, dtype=int)
    scores = np.asarray(scores, dtype=float)

    predicted = (
        scores >= THRESHOLD
    ).astype(int)

    positives = int(actual.sum())

    return {
        "rows": int(len(actual)),
        "actual_positives": positives,
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
            if positives > 0 else None
        ),
        "confusion_matrix": (
            confusion_matrix(
                actual, predicted,
                labels=[0, 1]
            ).tolist()
        )
    }


def parse_dates(series):

    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True
    ).dt.tz_localize(None).dt.normalize()


# ============================================================
# STEP 1 - LOAD EXISTING STAGE 26 FEATURES
# ============================================================

def load_acled():

    print("\n[1/8] Loading Stage 26 features...")

    if not ACLED_FILE.exists():
        raise FileNotFoundError(
            f"Missing Stage 26 features:\n{ACLED_FILE}"
        )

    df = pd.read_csv(
        ACLED_FILE,
        low_memory=False
    )

    required = ACLED_FEATURES + [
        "state",
        "week",
        "target_week",
        "target_count",
        "target_emergence",
        "target_escalation",
        "emergence_eligible"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Stage 26 columns: {missing}"
        )

    df["week"] = parse_dates(df["week"])
    df["target_week"] = parse_dates(
        df["target_week"]
    )

    if df[required].isna().any().any():
        raise ValueError(
            "Stage 26 features contain missing values."
        )

    if not (
        df["target_week"]
        == df["week"] + pd.Timedelta(days=7)
    ).all():
        raise ValueError(
            "Stage 26 target horizon is not 7 days."
        )

    if df.duplicated(
        ["state", "target_week"]
    ).any():
        raise ValueError(
            "Duplicate state-week records."
        )

    df["emergence_eligible"] = (
        df["emergence_eligible"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    print("Rows:", len(df))
    print("States:", df["state"].nunique())
    print("Target weeks:", df["target_week"].nunique())

    return df


# ============================================================
# STEP 2 - DISCOVER WEEKLY SOCIAL DATA
# ============================================================

def inspect_candidate(path):

    """
    Validate file schema and weekly structure.

    Never accept a tweet-level CSV simply because
    it has a date and a post identifier.
    """

    if not path.exists():
        return None

    if path.stat().st_size > MAX_SOCIAL_FILE_BYTES:
        return None

    if REJECT_FILE.search(path.name):
        return None

    try:
        header = pd.read_csv(
            path,
            nrows=0
        )

        columns = list(header.columns)

        date_candidates = [
            col for col in columns
            if col.strip().lower() in DATE_NAMES
        ]

        if not date_candidates:
            return None

        date_col = date_candidates[0]

        count_candidates = []

        for col in columns:

            normalized = col.strip().lower()

            if col == date_col:
                continue

            if REJECT_COLUMN.search(normalized):
                continue

            if COUNT_PATTERN.search(normalized):
                count_candidates.append(col)

        if not count_candidates:
            return None

        # A weekly file should be small enough
        # to inspect without heavy processing.
        data = pd.read_csv(
            path,
            usecols=[date_col] + count_candidates,
            low_memory=False
        )

        if not (10 <= len(data) <= 150):
            return None

        dates = parse_dates(data[date_col])

        if dates.isna().any():
            return None

        if dates.duplicated().any():
            return None

        dates = dates.sort_values()

        gaps = dates.diff().dropna()

        if not (
            gaps == pd.Timedelta(days=7)
        ).all():
            return None

        valid_counts = []

        for col in count_candidates:

            values = pd.to_numeric(
                data[col],
                errors="coerce"
            )

            if values.isna().any():
                continue

            if (values < 0).any():
                continue

            if not np.allclose(
                values,
                np.round(values)
            ):
                continue

            valid_counts.append(col)

        if not valid_counts:
            return None

        return {
            "path": path,
            "date_column": date_col,
            "count_columns": valid_counts,
            "rows": len(data),
            "first_week": str(dates.min().date()),
            "last_week": str(dates.max().date())
        }

    except (
        OSError,
        ValueError,
        pd.errors.ParserError,
        UnicodeError
    ):
        return None


def discover_social_file():

    print("\n[2/8] Discovering weekly social data...")

    if SOCIAL_FILE is not None:

        path = Path(SOCIAL_FILE)

        if not path.is_absolute():
            path = ROOT / path

        info = inspect_candidate(path)

        if info is None:
            raise ValueError(
                "The selected SOCIAL_FILE is not a "
                "valid weekly social-count CSV."
            )

        print("Selected:", path)

        return info

    search_dirs = [
        ROOT / "outputs",
        ROOT / "data" / "processed",
        ROOT / "data" / "features",
        ROOT / "data" / "aggregated",
        ROOT / "data" / "structured"
    ]

    candidates = []
    seen = set()

    for folder in search_dirs:

        if not folder.exists():
            continue

        for path in folder.rglob("*.csv"):

            if path in seen:
                continue

            seen.add(path)

            if REJECT_FILE.search(path.name):
                continue

            # Avoid scanning raw tweet datasets.
            if path.stat().st_size > MAX_SOCIAL_FILE_BYTES:
                continue

            info = inspect_candidate(path)

            if info is not None:
                candidates.append(info)

    if not candidates:

        raise FileNotFoundError(
            "\nNo valid saved national weekly social "
            "aggregate was found.\n\n"
            "The original tweet-level files are "
            "intentionally excluded.\n"
            "Check whether Stage 14 saved its weekly "
            "aggregates as CSV. If so, set SOCIAL_FILE "
            "to that exact file.\n"
            "No earlier stage needs to be rerun yet."
        )

    print("\nValidated weekly candidates:")

    for info in candidates:

        print("\nFile:", info["path"])
        print("Rows:", info["rows"])
        print("Week column:", info["date_column"])
        print("Count columns:", info["count_columns"])
        print(
            "Range:",
            info["first_week"],
            "to",
            info["last_week"]
        )

    if len(candidates) > 1:

        raise ValueError(
            "\nMultiple valid weekly files found.\n"
            "Set SOCIAL_FILE to the intended Stage 14 "
            "national weekly aggregation CSV.\n"
            "Do not combine unrelated files automatically."
        )

    info = candidates[0]

    print("\nUsing:", info["path"])

    return info


# ============================================================
# STEP 3 - LOAD WEEKLY SOCIAL COUNTS
# ============================================================

def load_social(info):

    print("\n[3/8] Loading weekly social counts...")

    date_col = info["date_column"]
    count_cols = info["count_columns"]

    social = pd.read_csv(
        info["path"],
        usecols=[date_col] + count_cols
    )

    social = social.rename(
        columns={date_col: "week"}
    )

    social["week"] = parse_dates(
        social["week"]
    )

    for col in count_cols:

        social[col] = pd.to_numeric(
            social[col],
            errors="raise"
        ).astype(float)

    social = social.sort_values(
        "week"
    ).reset_index(drop=True)

    if social["week"].duplicated().any():
        raise ValueError(
            "Duplicate national social weeks."
        )

    if social[count_cols].isna().any().any():
        raise ValueError(
            "Missing weekly social counts."
        )

    print("Weekly observations:", len(social))
    print("Columns:", count_cols)
    print("First week:", social["week"].min())
    print("Last week:", social["week"].max())

    return social, count_cols


# ============================================================
# STEP 4 - ALIGN OBSERVATION WEEKS
# ============================================================

def align_data(acled, social, count_cols):

    print("\n[4/8] Aligning observation weeks...")

    acled_weeks = set(acled["week"])
    social_weeks = set(social["week"])

    shared = sorted(
        acled_weeks.intersection(social_weeks)
    )

    print("Exact shared weeks:", len(shared))

    if len(shared) < 20:

        acled_days = sorted({
            pd.Timestamp(w).day_name()
            for w in acled_weeks
        })

        social_days = sorted({
            pd.Timestamp(w).day_name()
            for w in social_weeks
        })

        raise ValueError(
            "\nInsufficient exact weekly overlap.\n"
            f"ACLED week weekdays: {acled_days}\n"
            f"Social week weekdays: {social_days}\n"
            "Check the Stage 14 week-start/week-end "
            "definition before changing dates.\n"
            "No automatic shifting is performed."
        )

    df = acled.merge(
        social,
        on="week",
        how="inner",
        validate="many_to_one"
    )

    # Require complete state coverage.
    expected_states = acled["state"].nunique()

    state_counts = (
        df.groupby("week")["state"].nunique()
    )

    if not (
        state_counts == expected_states
    ).all():
        raise ValueError(
            "Some shared weeks have incomplete "
            "state coverage."
        )

    shared_weeks = pd.DatetimeIndex(
        sorted(df["week"].unique())
    )

    if not (
        shared_weeks.to_series()
        .diff()
        .dropna()
        == pd.Timedelta(days=7)
    ).all():
        raise ValueError(
            "Shared observation weeks contain gaps."
        )

    social_features = []
    change_features = []

    # log1p reduces the influence of
    # very large tweet-count spikes.
    for index, col in enumerate(count_cols):

        name = f"social_count_{index + 1}"

        df[name] = np.log1p(
            df[col].astype(float)
        )

        social_features.append(name)

    # Calculate national week-over-week changes
    # BEFORE repeating them across states.
    national = (
        df[["week"] + social_features]
        .drop_duplicates("week")
        .sort_values("week")
        .copy()
    )

    for feature in social_features:

        change_name = feature + "_change"

        national[change_name] = (
            national[feature].diff()
        )

        change_features.append(change_name)

    df = df.merge(
        national[
            ["week"] + change_features
        ],
        on="week",
        how="left",
        validate="many_to_one"
    )

    # The first shared week cannot have a
    # week-over-week change feature.
    df = df.dropna(
        subset=change_features
    ).copy()

    all_social_features = (
        social_features + change_features
    )

    if not np.isfinite(
        df[all_social_features].to_numpy()
    ).all():
        raise ValueError(
            "Nonfinite social features detected."
        )

    df = df.sort_values(
        ["target_week", "state"]
    ).reset_index(drop=True)

    print("Aligned rows:", len(df))
    print(
        "Aligned target weeks:",
        df["target_week"].nunique()
    )
    print(
        "Social features:",
        all_social_features
    )

    return df, all_social_features


# ============================================================
# STEP 5 - CHRONOLOGICAL BACKTEST FOLDS
# ============================================================

def create_folds(df):

    print("\n[5/8] Creating chronological folds...")

    weeks = sorted(
        df["target_week"].unique()
    )

    # Use the last 24 weeks as three
    # nonoverlapping eight-week test folds,
    # provided at least 16 training weeks remain.
    test_total = TEST_WEEKS * MAX_FOLDS

    initial_train = len(weeks) - test_total

    if initial_train < 16:

        raise ValueError(
            f"Only {len(weeks)} aligned target weeks.\n"
            "Need at least 40 weeks for three "
            "8-week test folds and 16 training weeks."
        )

    folds = []

    for i in range(MAX_FOLDS):

        train_end = (
            initial_train + i * TEST_WEEKS
        )

        test_end = (
            train_end + TEST_WEEKS
        )

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

        assert (
            train["target_week"].max()
            <= test["week"].min()
        )

        folds.append((i + 1, train, test))

        print(
            f"Fold {i+1}: "
            f"train={len(train)}, "
            f"test={len(test)}, "
            f"test_start="
            f"{test['target_week'].min().date()}"
        )

    return folds


# ============================================================
# STEP 6 - EVENT COUNT ABLATION
# ============================================================

def evaluate_count(folds, social_features):

    print("\n[6/8] Event-count ablation...")

    feature_sets = {
        "acled_only": ACLED_FEATURES,
        "acled_plus_social": (
            ACLED_FEATURES + social_features
        )
    }

    outputs = []

    for fold_id, train, test in folds:

        output = test[
            ["state", "week", "target_week"]
        ].copy()

        output["fold"] = fold_id
        output["actual"] = (
            test["target_count"].to_numpy()
        )

        output["persistence"] = (
            test["count_current"].to_numpy()
        )

        output["rolling_mean"] = (
            test["rolling_mean_4"].to_numpy()
        )

        for name, columns in feature_sets.items():

            model = RandomForestRegressor(
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=4,
                max_features=0.8,
                random_state=SEED,
                n_jobs=2
            )

            model.fit(
                train[columns],
                train["target_count"]
            )

            output[name] = np.maximum(
                0,
                model.predict(test[columns])
            )

        outputs.append(output)

        print(f"Count fold {fold_id} completed.")

    predictions = pd.concat(
        outputs,
        ignore_index=True
    )

    save_csv(
        predictions,
        "count_ablation_predictions.csv"
    )

    return predictions


# ============================================================
# STEP 7 - CLASSIFICATION BASELINES
# ============================================================

def historical_rate(train, test, target):

    global_rate = float(
        train[target].mean()
    )

    stats = (
        train.groupby("state")[target]
        .agg(["sum", "count"])
    )

    scores = []

    for state in test["state"]:

        if state in stats.index:

            positive = float(
                stats.loc[state, "sum"]
            )

            total = float(
                stats.loc[state, "count"]
            )

        else:

            positive = 0.0
            total = 0.0

        score = (
            positive
            + PRIOR_STRENGTH * global_rate
        ) / (
            total + PRIOR_STRENGTH
        )

        scores.append(score)

    return np.asarray(scores)


def evaluate_classification(
    folds,
    social_features,
    target,
    task,
    emergence_only=False
):

    print(f"\nEvaluating {task}...")

    feature_sets = {
        "acled_only": ACLED_FEATURES,
        "acled_plus_social": (
            ACLED_FEATURES + social_features
        )
    }

    outputs = []

    for fold_id, train, test in folds:

        if emergence_only:

            train = train[
                train["emergence_eligible"]
            ].copy()

            test = test[
                test["emergence_eligible"]
            ].copy()

        if train.empty or test.empty:
            print(
                f"{task} fold {fold_id}: "
                "no eligible rows."
            )
            continue

        output = test[
            ["state", "week", "target_week"]
        ].copy()

        output["fold"] = fold_id

        output["actual"] = (
            test[target].astype(int).to_numpy()
        )

        output["historical_rate"] = (
            historical_rate(
                train, test, target
            )
        )

        y_train = (
            train[target].astype(int)
        )

        for name, columns in feature_sets.items():

            if y_train.nunique() < 2:

                scores = np.full(
                    len(test),
                    float(y_train.iloc[0])
                )

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
                    train[columns],
                    y_train
                )

                probabilities = (
                    model.predict_proba(
                        test[columns]
                    )
                )

                positive_index = list(
                    model.classes_
                ).index(1)

                scores = probabilities[
                    :, positive_index
                ]

            output[name] = scores

        outputs.append(output)

        print(
            f"{task} fold {fold_id}: "
            f"test={len(test)}, "
            f"positives={int(output['actual'].sum())}"
        )

    if not outputs:
        raise ValueError(
            f"No valid {task} evaluation folds."
        )

    result = pd.concat(
        outputs,
        ignore_index=True
    )

    save_csv(
        result,
        f"{task}_ablation_predictions.csv"
    )

    return result


# ============================================================
# STEP 8 - METRICS
# ============================================================

COUNT_MODELS = [
    "persistence",
    "rolling_mean",
    "acled_only",
    "acled_plus_social"
]

BINARY_MODELS = [
    "historical_rate",
    "acled_only",
    "acled_plus_social"
]


def summarize_count(predictions):

    rows = []

    groups = list(
        predictions.groupby("fold")
    )

    groups.append(("overall", predictions))

    for fold, subset in groups:

        for model in COUNT_MODELS:

            rows.append({
                "fold": str(fold),
                "model": model,
                "rows": int(len(subset)),
                **count_metrics(
                    subset["actual"],
                    subset[model]
                )
            })

    return pd.DataFrame(rows)


def summarize_binary(predictions, task):

    rows = []

    groups = list(
        predictions.groupby("fold")
    )

    groups.append(("overall", predictions))

    for fold, subset in groups:

        for model in BINARY_MODELS:

            result = binary_metrics(
                subset["actual"],
                subset[model]
            )

            rows.append({
                "task": task,
                "fold": str(fold),
                "model": model,
                **{
                    key: value
                    for key, value in result.items()
                    if key != "confusion_matrix"
                }
            })

    return pd.DataFrame(rows)


def save_state_metrics(
    count_pred,
    emergence_pred,
    escalation_pred
):

    count_rows = []

    for state, subset in count_pred.groupby("state"):

        for model in COUNT_MODELS:

            count_rows.append({
                "state": state,
                "model": model,
                "rows": len(subset),
                **count_metrics(
                    subset["actual"],
                    subset[model]
                )
            })

    save_csv(
        pd.DataFrame(count_rows),
        "state_count_ablation_metrics.csv"
    )

    for task, predictions in [
        ("emergence", emergence_pred),
        ("escalation", escalation_pred)
    ]:

        rows = []

        for state, subset in predictions.groupby(
            "state"
        ):

            for model in BINARY_MODELS:

                result = binary_metrics(
                    subset["actual"],
                    subset[model]
                )

                rows.append({
                    "state": state,
                    "task": task,
                    "model": model,
                    **{
                        key: value
                        for key, value in result.items()
                        if key != "confusion_matrix"
                    }
                })

        save_csv(
            pd.DataFrame(rows),
            f"state_{task}_ablation_metrics.csv"
        )


# ============================================================
# ESCALATION PRECISION-RECALL ANALYSIS
# ============================================================

def save_escalation_pr(predictions):

    actual = predictions["actual"].to_numpy(
        dtype=int
    )

    if actual.sum() == 0:
        print(
            "No escalation positives: "
            "PR curve unavailable."
        )
        return

    plt.figure(figsize=(8, 5))

    for model in BINARY_MODELS:

        scores = predictions[model].to_numpy(
            dtype=float
        )

        precision, recall, thresholds = (
            precision_recall_curve(
                actual, scores
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
            f"escalation_pr_{model}.csv"
        )

        plt.plot(
            recall,
            precision,
            label=model
        )

    plt.axhline(
        float(actual.mean()),
        linestyle="--",
        label="Positive prevalence"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Escalation Precision-Recall Comparison"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "escalation_pr_comparison.png",
        dpi=160
    )

    plt.close()


# ============================================================
# PLOTS
# ============================================================

def save_plots(count_df, escalation_df):

    count_overall = count_df[
        count_df["fold"] == "overall"
    ]

    plt.figure(figsize=(10, 5))

    plt.bar(
        count_overall["model"],
        count_overall["mae"]
    )

    plt.ylabel("MAE (lower is better)")
    plt.title(
        "Stage 27: Event-Count Ablation"
    )
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "count_ablation.png",
        dpi=160
    )

    plt.close()

    escalation_overall = escalation_df[
        escalation_df["fold"] == "overall"
    ]

    plt.figure(figsize=(10, 5))

    plt.bar(
        escalation_overall["model"],
        escalation_overall["average_precision"]
    )

    plt.ylabel("Average precision")
    plt.title(
        "Stage 27: Escalation Ablation"
    )
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "escalation_ap_ablation.png",
        dpi=160
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "STAGE 27 - CORRECTED SOCIAL-MEDIA ABLATION"
    )
    print("=" * 72)

    acled = load_acled()

    info = discover_social_file()

    social, count_cols = load_social(info)

    aligned, social_features = align_data(
        acled,
        social,
        count_cols
    )

    save_csv(
        aligned[
            [
                "state",
                "week",
                "target_week",
                "target_count",
                "target_emergence",
                "target_escalation",
                "emergence_eligible"
            ]
            + ACLED_FEATURES
            + social_features
        ],
        "aligned_state_social_features.csv"
    )

    folds = create_folds(aligned)

    count_pred = evaluate_count(
        folds,
        social_features
    )

    emergence_pred = evaluate_classification(
        folds,
        social_features,
        "target_emergence",
        "emergence",
        emergence_only=True
    )

    escalation_pred = evaluate_classification(
        folds,
        social_features,
        "target_escalation",
        "escalation"
    )

    print("\n[7/8] Calculating metrics...")

    count_df = summarize_count(count_pred)

    emergence_df = summarize_binary(
        emergence_pred,
        "emergence"
    )

    escalation_df = summarize_binary(
        escalation_pred,
        "escalation"
    )

    save_csv(
        count_df,
        "count_ablation_metrics.csv"
    )

    save_csv(
        emergence_df,
        "emergence_ablation_metrics.csv"
    )

    save_csv(
        escalation_df,
        "escalation_ablation_metrics.csv"
    )

    save_state_metrics(
        count_pred,
        emergence_pred,
        escalation_pred
    )

    print("\n[8/8] Saving plots and summary...")

    save_escalation_pr(escalation_pred)

    save_plots(
        count_df,
        escalation_df
    )

    overall_count = count_df[
        count_df["fold"] == "overall"
    ]

    overall_emergence = emergence_df[
        emergence_df["fold"] == "overall"
    ]

    overall_escalation = escalation_df[
        escalation_df["fold"] == "overall"
    ]

    summary = {
        "stage": 27,
        "social_source": str(info["path"]),
        "social_source_columns": count_cols,
        "social_features": social_features,
        "states": int(aligned["state"].nunique()),
        "aligned_weeks": int(
            aligned["target_week"].nunique()
        ),
        "folds": len(folds),
        "classification_threshold": THRESHOLD,
        "count_results": overall_count.to_dict(
            orient="records"
        ),
        "emergence_results": (
            overall_emergence.to_dict(
                orient="records"
            )
        ),
        "escalation_results": (
            overall_escalation.to_dict(
                orient="records"
            )
        ),
        "limitations": [
            "National social signals only",
            "No verified state-specific tweet locations",
            "Keyword activity is not confirmed incidents",
            "Exact week alignment required",
            "Social source week semantics require verification",
            "Only historical evaluation",
            "No causal inference",
            "Random forest scores are uncalibrated",
            "Fixed threshold is not optimized",
            "State-level test samples may be small"
        ]
    }

    save_json(
        summary,
        "stage27_summary.json"
    )

    print("\n" + "=" * 72)
    print("STAGE 27 - COMPLETED")
    print("=" * 72)

    print("\nCOUNT FORECASTING:")

    for _, row in overall_count.iterrows():

        print(
            f"{row['model']}: "
            f"MAE={row['mae']:.3f}, "
            f"RMSE={row['rmse']:.3f}"
        )

    print("\nEMERGENCE:")

    for _, row in overall_emergence.iterrows():

        print(
            f"{row['model']}: "
            f"F1={row['f1']:.3f}, "
            f"AP={row['average_precision']}"
        )

    print("\nESCALATION:")

    for _, row in overall_escalation.iterrows():

        print(
            f"{row['model']}: "
            f"F1={row['f1']:.3f}, "
            f"AP={row['average_precision']}"
        )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nEarlier stages remain unchanged.")


if __name__ == "__main__":
    main()
