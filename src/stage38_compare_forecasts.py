
"""Stage 38: leakage-safe seven-day tweet-volume forecast comparison.
Run from project root: python src/stage38_compare_forecasts.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path('outputs/stage37/tweet_event_summary.parquet')
OUT = Path('outputs/stage38')

HORIZON = 7
MIN_HISTORY = 28
HOLDOUT_FRACTION = 0.25


def forecast(history, model):
    values = history.to_numpy(dtype=float)

    if model.startswith('mean_'):
        n = int(model.split('_')[1])
        return np.repeat(values[-n:].mean(), HORIZON)

    if model == 'ewma_7':
        return np.repeat(
            history.ewm(span=7, adjust=False).mean().iloc[-1],
            HORIZON
        )

    if model == 'weekday_seasonal_4w':
        # Average the same weekday over the previous four weeks.
        return np.array([
            values[-28 + d::7].mean()
            for d in range(HORIZON)
        ])

    if model == 'last_week':
        return values[-7:].copy()

    raise ValueError(model)


def metrics(frame):
    err = (
        frame['predicted'].to_numpy()
        - frame['actual'].to_numpy()
    )

    actual = frame['actual'].to_numpy()
    denom = np.abs(actual).sum()

    return {
        'windows': int(frame['window'].nunique()),
        'evaluated_days': len(frame),
        'mae': float(np.abs(err).mean()),
        'rmse': float(np.sqrt(np.square(err).mean())),
        'wape_pct': (
            float(100 * np.abs(err).sum() / denom)
            if denom else np.nan
        ),
        'bias': float(err.mean())
    }


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f'Missing {SOURCE}; complete Stage 37 first.'
        )

    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(
        SOURCE,
        columns=['tweetId', 'post_date']
    )

    df['post_date'] = pd.to_datetime(
        df['post_date'],
        utc=True,
        errors='coerce'
    )

    df = df.dropna(
        subset=['post_date', 'tweetId']
    )

    df['day'] = df['post_date'].dt.floor('D')

    daily = (
        df.groupby('day')['tweetId']
        .nunique()
        .sort_index()
        .asfreq('D', fill_value=0)
    )

    if len(daily) < MIN_HISTORY + HORIZON * 8:
        raise ValueError(
            'Not enough daily data for comparison and holdout.'
        )

    # Non-overlapping seven-day historical test windows.
    # Each forecast uses only information available before its start.
    first = MIN_HISTORY + (
        (len(daily) - MIN_HISTORY) % HORIZON
    )

    starts = list(
        range(
            first,
            len(daily) - HORIZON + 1,
            HORIZON
        )
    )

    if len(starts) < 8:
        raise ValueError('Not enough test windows.')

    n_holdout = max(
        2,
        int(np.ceil(len(starts) * HOLDOUT_FRACTION))
    )

    split = len(starts) - n_holdout

    models = [
        'mean_7',
        'mean_14',
        'mean_28',
        'ewma_7',
        'weekday_seasonal_4w',
        'last_week'
    ]

    records = []

    for window, start in enumerate(starts, 1):
        history = daily.iloc[:start]
        actual = daily.iloc[start:start + HORIZON]

        period = (
            'selection'
            if window <= split
            else 'holdout'
        )

        for model in models:
            pred = forecast(history, model)

            for day, y, p in zip(
                actual.index,
                actual.to_numpy(),
                pred
            ):
                records.append({
                    'period': period,
                    'window': window,
                    'model': model,
                    'date': day.date().isoformat(),
                    'actual': int(y),
                    'predicted': float(p)
                })

    predictions = pd.DataFrame(records)

    predictions.to_csv(
        OUT / 'backtest_predictions.csv',
        index=False
    )

    summary = pd.DataFrame([
        {
            'period': period,
            'model': model,
            **metrics(part)
        }
        for (period, model), part
        in predictions.groupby(
            ['period', 'model'],
            sort=False
        )
    ])

    summary.to_csv(
        OUT / 'model_metrics.csv',
        index=False
    )

    # Select model using selection data only.
    selected = (
        summary[summary.period == 'selection']
        .sort_values(['mae', 'rmse', 'model'])
        .iloc[0]['model']
    )

    # Evaluate selected model on untouched holdout.
    holdout = summary[
        (summary.period == 'holdout')
        & (summary.model == selected)
    ].iloc[0]

    # Forecast seven days after the final historical date.
    final_pred = forecast(daily, selected)

    dates = pd.date_range(
        daily.index[-1] + pd.Timedelta(days=1),
        periods=HORIZON,
        freq='D'
    )

    future = pd.DataFrame({
        'date': dates.date.astype(str),
        'selected_model': selected,
        'predicted_positive_tweets': final_pred
    })

    future.to_csv(
        OUT / 'selected_next_7days.csv',
        index=False
    )

    print(
        '\n=== STAGE 38: SEVEN-DAY FORECAST MODEL COMPARISON ==='
    )

    print(
        f'Data: {daily.index[0].date()} '
        f'to {daily.index[-1].date()}'
    )

    print(
        f'Windows: {len(starts)} total; '
        f'{split} selection, {n_holdout} holdout'
    )

    print('\nSELECTION RESULTS (lower is better):')

    print(
        summary[summary.period == 'selection']
        .sort_values('mae')[
            ['model', 'mae', 'rmse', 'wape_pct']
        ]
        .round(3)
        .to_string(index=False)
    )

    print(
        f'\nSelected using selection MAE: {selected}'
    )

    print(
        f'Untouched holdout MAE: {holdout.mae:.3f}; '
        f'RMSE: {holdout.rmse:.3f}; '
        f'WAPE: {holdout.wape_pct:.2f}%'
    )

    print('\nFINAL SEVEN-DAY FORECAST:')
    print(future.to_string(index=False))

    print(f'\nSaved results under: {OUT}')

    print(
        'Caution: predictions are positive tweet volumes, '
        'not verified civil-unrest incidents.'
    )

    print(
        'Stage 38 windows may differ from Stage 37; '
        'compare model scores within this run only.'
    )


if __name__ == '__main__':
    main()

