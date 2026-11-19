
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# STAGE 29: CIVIL UNREST FORECASTING DASHBOARD
#
# Uses saved Stage 26, Stage 27 and Stage 28 outputs.
#
# No model training.
# No tweet processing.
# No external API.
# Historical backtest results only.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

S26 = ROOT / "outputs" / "stage26"
S26V2 = ROOT / "outputs" / "stage26_v2"
S27 = ROOT / "outputs" / "stage27"
S28 = ROOT / "outputs" / "stage28"

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

LABELS = {
    "persistence": "Persistence",
    "rolling_mean": "Rolling mean",
    "historical_rate": "Historical rate",
    "acled_only": "ACLED-only RF",
    "acled_plus_social": "ACLED + social RF",
}

PRED_FILES = {
    "count": "count_ablation_predictions.csv",
    "emergence": "emergence_ablation_predictions.csv",
    "escalation": "escalation_ablation_predictions.csv",
}

# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Civil Unrest | Historical Forecast Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("Civil Unrest — Historical Forecast Evaluation")

st.caption(
    "B.Tech research dashboard | Historical ACLED targets "
    "and saved model backtests | Not live monitoring "
    "or operational alerts"
)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv(path_str):
    path = Path(path_str)

    if not path.is_file():
        return None

    return pd.read_csv(path)


def optional(folder, name):
    return read_csv(str(folder / name))


def prepare_predictions(task):
    filename = PRED_FILES[task]
    df = optional(S27, filename)

    if df is None:
        return None, f"Missing {S27 / filename}"

    models = (
        COUNT_MODELS
        if task == "count"
        else CLASS_MODELS
    )

    needed = [
        "state",
        "week",
        "target_week",
        "fold",
        "actual",
    ] + models

    missing = [
        x for x in needed
        if x not in df.columns
    ]

    if missing:
        return None, (
            f"{filename}: missing columns {missing}"
        )

    df = df.copy()

    for col in ["week", "target_week"]:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce",
        )

    for col in ["actual"] + models:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(subset=needed)

    if task != "count":
        df = df[df["actual"].isin([0, 1])]

        for model in models:
            df = df[
                df[model].between(0, 1)
            ]

    df = df.sort_values(
        ["target_week", "state"]
    ).reset_index(drop=True)

    return df, None


# ============================================================
# FILTERS
# ============================================================

def filter_rows(df, key):
    states = sorted(
        df["state"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    folds = sorted(
        df["fold"]
        .dropna()
        .unique()
        .tolist()
    )

    c1, c2 = st.columns([2, 1])

    with c1:
        selected = st.multiselect(
            "States",
            states,
            default=states,
            key=f"{key}_states",
        )

    with c2:
        chosen_folds = st.multiselect(
            "Backtest folds",
            folds,
            default=folds,
            key=f"{key}_folds",
        )

    out = df[
        df["state"].astype(str).isin(selected)
        & df["fold"].isin(chosen_folds)
    ].copy()

    if out.empty:
        st.info(
            "No observations match the selected filters."
        )

    return out


# ============================================================
# EVALUATION METRICS
# ============================================================

def count_metrics(df, model):
    err = (
        df[model].to_numpy(float)
        - df["actual"].to_numpy(float)
    )

    return {
        "Model": LABELS.get(model, model),
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "Bias": err.mean(),
        "Rows": len(df),
    }


def class_metrics(df, model, threshold):
    y = df["actual"].to_numpy(int)
    score = df[model].to_numpy(float)

    pred = (
        score >= threshold
    ).astype(int)

    tp = int(
        ((y == 1) & (pred == 1)).sum()
    )

    tn = int(
        ((y == 0) & (pred == 0)).sum()
    )

    fp = int(
        ((y == 0) & (pred == 1)).sum()
    )

    fn = int(
        ((y == 1) & (pred == 0)).sum()
    )

    ap = (
        float(average_precision_score(y, score))
        if y.sum()
        else np.nan
    )

    return {
        "Model": LABELS.get(model, model),
        "Precision": precision_score(
            y, pred, zero_division=0
        ),
        "Recall": recall_score(
            y, pred, zero_division=0
        ),
        "F1": f1_score(
            y, pred, zero_division=0
        ),
        "AP": ap,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Rows": len(df),
    }


def fmt(df):
    numeric = [
        "MAE",
        "RMSE",
        "Bias",
        "Precision",
        "Recall",
        "F1",
        "AP",
    ]

    return df.style.format({
        c: "{:.3f}"
        for c in numeric
        if c in df.columns
    })


def show_download(df, name):
    st.download_button(
        "Download filtered CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        name,
        "text/csv",
    )


# ============================================================
# LOAD SAVED PROJECT DATA
# ============================================================

count, count_err = prepare_predictions("count")

emergence, emergence_err = prepare_predictions(
    "emergence"
)

escalation, escalation_err = prepare_predictions(
    "escalation"
)

features = optional(
    S26,
    "state_week_features.csv",
)

if features is not None:
    features = features.copy()

    for c in ["week", "target_week"]:
        if c in features.columns:
            features[c] = pd.to_datetime(
                features[c],
                errors="coerce",
            )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.header("Navigation")

    page = st.radio(
        "Section",
        [
            "Overview",
            "Historical unrest",
            "Next-week count forecasts",
            "Emergence & escalation",
            "Model comparison",
            "Error analysis",
        ],
    )

    st.divider()

    st.caption(
        "All charts are historical. A model score "
        "is not a calibrated probability. Threshold "
        "changes are exploratory, not validated."
    )

    st.caption(f"Project root: {ROOT}")


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":
    st.subheader("Saved-data coverage")

    a, b, c, d = st.columns(4)

    a.metric(
        "States in historical features",
        int(features["state"].nunique())
        if features is not None
        and "state" in features
        else "N/A",
    )

    b.metric(
        "Count backtest rows",
        len(count) if count is not None else "N/A",
    )

    c.metric(
        "Emergence-eligible test rows",
        len(emergence)
        if emergence is not None
        else "N/A",
    )

    d.metric(
        "Escalation test rows",
        len(escalation)
        if escalation is not None
        else "N/A",
    )

    for msg in [
        count_err,
        emergence_err,
        escalation_err,
    ]:
        if msg:
            st.warning(msg)

    st.info(
        "Stage 27 comparisons use identical test "
        "examples within each task. Stage 26 V2 "
        "uses different historical evaluation "
        "windows; do not directly compare its "
        "scores against Stage 27 as if paired."
    )

    if count is not None:
        st.subheader(
            "Stage 27 count forecasting "
            "(all saved test rows)"
        )

        result = pd.DataFrame([
            count_metrics(count, m)
            for m in COUNT_MODELS
        ])

        st.dataframe(
            fmt(result),
            use_container_width=True,
            hide_index=True,
        )

    if (
        emergence is not None
        and escalation is not None
    ):
        st.subheader(
            "Stage 27 classification "
            "(0.50 threshold)"
        )

        for task, df in [
            ("Emergence", emergence),
            ("Escalation", escalation),
        ]:
            st.markdown(f"**{task}**")

            result = pd.DataFrame([
                class_metrics(df, m, 0.5)
                for m in CLASS_MODELS
            ])

            st.dataframe(
                fmt(result),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# PAGE 2: HISTORICAL UNREST
# ============================================================

elif page == "Historical unrest":
    st.subheader(
        "Historical ACLED state-week patterns"
    )

    if features is None:
        st.error(
            f"Missing {S26 / 'state_week_features.csv'}"
        )

    elif not {
        "state",
        "week",
        "count_current",
    }.issubset(features.columns):
        st.error(
            "Historical features must contain "
            "state, week and count_current."
        )

    else:
        st.caption(
            "count_current represents the ACLED "
            "event count in the observation week, "
            "not a count of tweets."
        )

        states = sorted(
            features["state"]
            .dropna()
            .unique()
            .tolist()
        )

        chosen = st.multiselect(
            "Choose states",
            states,
            default=states[:min(5, len(states))],
        )

        subset = features[
            features["state"].isin(chosen)
        ].dropna(subset=["week"]).copy()

        if subset.empty:
            st.info("Choose at least one state.")

        else:
            fig = px.line(
                subset,
                x="week",
                y="count_current",
                color="state",
                markers=True,
                title="Weekly observed ACLED events",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            summary = (
                subset.groupby(
                    "state",
                    as_index=False,
                )
                .agg(
                    weeks=("week", "nunique"),
                    total_events=(
                        "count_current",
                        "sum",
                    ),
                    mean_events=(
                        "count_current",
                        "mean",
                    ),
                )
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

            show_download(
                summary,
                "historical_state_summary.csv",
            )

        overall = (
            features.groupby(
                "state",
                as_index=False,
            )["count_current"]
            .sum()
            .sort_values(
                "count_current",
                ascending=False,
            )
        )

        fig = px.bar(
            overall.head(20),
            x="state",
            y="count_current",
            title=(
                "Top 20 states by summed "
                "observed weekly counts"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# PAGE 3: NEXT-WEEK COUNT FORECASTS
# ============================================================

elif page == "Next-week count forecasts":
    st.subheader(
        "Next-week event counts — "
        "historical out-of-sample backtests"
    )

    if count is None:
        st.error(count_err)

    else:
        df = filter_rows(count, "count")

        if not df.empty:
            model = st.selectbox(
                "Forecast model",
                COUNT_MODELS,
                format_func=lambda x: LABELS[x],
            )

            result = pd.DataFrame([
                count_metrics(df, m)
                for m in COUNT_MODELS
            ])

            st.dataframe(
                fmt(result),
                use_container_width=True,
                hide_index=True,
            )

            series = (
                df.groupby(
                    "target_week",
                    as_index=False,
                )[["actual", model]]
                .sum()
            )

            series = series.rename(
                columns={
                    "actual": "Observed ACLED count",
                    model: "Predicted count",
                }
            )

            long = series.melt(
                id_vars="target_week",
                var_name="Series",
                value_name="Events",
            )

            fig = px.line(
                long,
                x="target_week",
                y="Events",
                color="Series",
                markers=True,
                title=(
                    "Summed test-state event counts "
                    "by target week"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            state = st.selectbox(
                "Inspect one state",
                sorted(df["state"].unique()),
            )

            one = df[
                df["state"] == state
            ].copy()

            fig = px.line(
                one,
                x="target_week",
                y=["actual", model],
                markers=True,
                title=(
                    f"{state}: observed vs predicted "
                    "next-week events"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            one["absolute_error"] = (
                one[model] - one["actual"]
            ).abs()

            st.dataframe(
                one[
                    [
                        "state",
                        "week",
                        "target_week",
                        "fold",
                        "actual",
                        model,
                        "absolute_error",
                    ]
                ].sort_values("target_week"),
                use_container_width=True,
                hide_index=True,
            )

            show_download(
                one,
                "count_forecasts_filtered.csv",
            )


# ============================================================
# PAGE 4: EMERGENCE AND ESCALATION
# ============================================================

elif page == "Emergence & escalation":
    st.subheader(
        "Historical next-week binary classification"
    )

    task = st.radio(
        "Classification task",
        ["emergence", "escalation"],
        horizontal=True,
    )

    source = (
        emergence
        if task == "emergence"
        else escalation
    )

    err = (
        emergence_err
        if task == "emergence"
        else escalation_err
    )

    if source is None:
        st.error(err)

    else:
        st.caption(
            "Emergence includes only eligible "
            "state-weeks in the saved test set. "
            "The slider is for retrospective "
            "exploration and does not validate "
            "a new operating threshold."
        )

        df = filter_rows(source, task)

        if not df.empty:
            model = st.selectbox(
                "Model",
                CLASS_MODELS,
                format_func=lambda x: LABELS[x],
            )

            threshold = st.slider(
                "Exploratory score threshold",
                0.0,
                1.0,
                0.50,
                0.05,
            )

            result_table = pd.DataFrame([
                class_metrics(
                    df,
                    m,
                    threshold,
                )
                for m in CLASS_MODELS
            ])

            st.dataframe(
                fmt(result_table),
                use_container_width=True,
                hide_index=True,
            )

            result = class_metrics(
                df,
                model,
                threshold,
            )

            matrix = np.array([
                [result["TN"], result["FP"]],
                [result["FN"], result["TP"]],
            ])

            fig = px.imshow(
                matrix,
                x=[
                    "Predicted 0",
                    "Predicted 1",
                ],
                y=[
                    "Actual 0",
                    "Actual 1",
                ],
                text_auto=True,
                color_continuous_scale="Blues",
                title=(
                    f"{LABELS[model]} "
                    "confusion matrix"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            detailed = df[
                [
                    "state",
                    "week",
                    "target_week",
                    "fold",
                    "actual",
                    model,
                ]
            ].copy()

            detailed["predicted"] = (
                detailed[model] >= threshold
            ).astype(int)

            detailed["outcome"] = np.select(
                [
                    (detailed["actual"] == 1)
                    & (detailed["predicted"] == 1),

                    (detailed["actual"] == 0)
                    & (detailed["predicted"] == 0),

                    (detailed["actual"] == 0)
                    & (detailed["predicted"] == 1),

                    (detailed["actual"] == 1)
                    & (detailed["predicted"] == 0),
                ],
                [
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                ],
                default="unknown",
            )

            choices = st.multiselect(
                "Outcome types",
                ["TP", "TN", "FP", "FN"],
                default=["FP", "FN"],
            )

            shown = detailed[
                detailed["outcome"].isin(choices)
            ]

            st.dataframe(
                shown,
                use_container_width=True,
                hide_index=True,
            )

            show_download(
                shown,
                f"{task}_classification_filtered.csv",
            )


# ============================================================
# PAGE 5: MODEL COMPARISON
# ============================================================

elif page == "Model comparison":
    st.subheader(
        "Controlled Stage 27 social-media ablation"
    )

    if count is not None:
        st.markdown(
            "**Count prediction "
            "(lower error is better)**"
        )

        metric_df = pd.DataFrame([
            count_metrics(count, m)
            for m in COUNT_MODELS
        ])

        fig = px.bar(
            metric_df,
            x="Model",
            y=["MAE", "RMSE"],
            barmode="group",
            title="Event-count error",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    for task, source in [
        ("emergence", emergence),
        ("escalation", escalation),
    ]:
        if source is not None:
            st.markdown(
                f"**{task.title()} "
                "(higher is better)**"
            )

            table = pd.DataFrame([
                class_metrics(
                    source,
                    m,
                    0.5,
                )
                for m in CLASS_MODELS
            ])

            fig = px.bar(
                table,
                x="Model",
                y=["F1", "AP"],
                barmode="group",
                title=(
                    f"{task.title()}: "
                    "F1 and average precision"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.divider()

    stage26 = optional(
        S28,
        "stage26_v2_mean_fold_metrics.csv",
    )

    if stage26 is not None:
        st.subheader(
            "Stage 26 V2 — separate "
            "chronological backtest"
        )

        st.caption(
            "These are unweighted averages of "
            "available fold metrics, not pooled "
            "predictions; different test windows "
            "from Stage 27."
        )

        st.dataframe(
            stage26,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "Stage 26 V2 fold summary not found. "
            "Stage 27 comparison remains available."
        )

    ablation = optional(
        S28,
        "social_ablation_comparison.csv",
    )

    if ablation is not None:
        st.subheader(
            "Saved Stage 28 ablation table"
        )

        st.dataframe(
            ablation,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 6: ERROR ANALYSIS
# ============================================================

elif page == "Error analysis":
    st.subheader(
        "Historical prediction errors"
    )

    mode = st.radio(
        "Error category",
        [
            "Count errors",
            "Emergence FP/FN",
            "Escalation FP/FN",
        ],
        horizontal=True,
    )

    if mode == "Count errors":
        if count is None:
            st.error(count_err)

        else:
            df = filter_rows(
                count,
                "errors_count",
            )

            if not df.empty:
                model = st.selectbox(
                    "Model",
                    COUNT_MODELS,
                    format_func=lambda x: LABELS[x],
                )

                errors = df[
                    [
                        "state",
                        "week",
                        "target_week",
                        "fold",
                        "actual",
                        model,
                    ]
                ].copy()

                errors["signed_error"] = (
                    errors[model] - errors["actual"]
                )

                errors["absolute_error"] = (
                    errors["signed_error"].abs()
                )

                states = (
                    errors.groupby(
                        "state",
                        as_index=False,
                    )
                    .agg(
                        MAE=(
                            "absolute_error",
                            "mean",
                        ),
                        Bias=(
                            "signed_error",
                            "mean",
                        ),
                        Rows=(
                            "actual",
                            "size",
                        ),
                    )
                    .sort_values(
                        "MAE",
                        ascending=False,
                    )
                )

                fig = px.bar(
                    states.head(20),
                    x="state",
                    y="MAE",
                    title=(
                        "Largest state-level MAE "
                        "(selected test subset)"
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.dataframe(
                    states,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown(
                    "**Largest individual errors**"
                )

                st.dataframe(
                    errors.sort_values(
                        "absolute_error",
                        ascending=False,
                    ).head(100),
                    use_container_width=True,
                    hide_index=True,
                )

                show_download(
                    errors,
                    "count_errors_filtered.csv",
                )

    else:
        task = (
            "emergence"
            if mode.startswith("Emergence")
            else "escalation"
        )

        source = (
            emergence
            if task == "emergence"
            else escalation
        )

        err = (
            emergence_err
            if task == "emergence"
            else escalation_err
        )

        if source is None:
            st.error(err)

        else:
            df = filter_rows(
                source,
                f"errors_{task}",
            )

            if not df.empty:
                model = st.selectbox(
                    "Model",
                    CLASS_MODELS,
                    format_func=lambda x: LABELS[x],
                )

                scored = df[
                    [
                        "state",
                        "week",
                        "target_week",
                        "fold",
                        "actual",
                        model,
                    ]
                ].copy()

                scored["predicted"] = (
                    scored[model] >= 0.5
                ).astype(int)

                scored["error_type"] = np.select(
                    [
                        (scored["actual"] == 0)
                        & (scored["predicted"] == 1),

                        (scored["actual"] == 1)
                        & (scored["predicted"] == 0),
                    ],
                    [
                        "FP",
                        "FN",
                    ],
                    default="correct",
                )

                mistakes = scored[
                    scored["error_type"] != "correct"
                ].copy()

                summary = (
                    mistakes.groupby(
                        ["state", "error_type"]
                    )
                    .size()
                    .unstack(fill_value=0)
                    .reindex(
                        columns=["FP", "FN"],
                        fill_value=0,
                    )
                )

                if summary.empty:
                    st.success(
                        "No classification errors "
                        "for the selected subset "
                        "and model."
                    )

                else:
                    fig = px.bar(
                        summary.reset_index(),
                        x="state",
                        y=["FP", "FN"],
                        barmode="stack",
                        title=(
                            "False alerts and "
                            "missed events by state"
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                    st.dataframe(
                        summary.reset_index(),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.dataframe(
                        mistakes.sort_values(
                            [
                                "error_type",
                                "target_week",
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    show_download(
                        mistakes,
                        f"{task}_errors_filtered.csv",
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research limitations: Historical ACLED "
    "aggregates are the ground truth used for "
    "these tasks. National social signals are "
    "not geolocated incident evidence. Model "
    "scores are not necessarily calibrated. "
    "No live inference, verified incident "
    "detection, or operational risk assessment "
    "is performed."
)
