"""Leakage-free rolling-origin backtest for the dashboard's 7-day forecast."""
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path('outputs/stage37/tweet_event_summary.parquet')
OUT = Path('outputs/stage37/forecast_backtest')
WINDOW = 7
HORIZON = 7


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Missing input: {SOURCE}')
    df = pd.read_parquet(SOURCE, columns=['tweetId', 'post_date'])
    df['post_date'] = pd.to_datetime(df['post_date'], utc=True, errors='coerce')
    invalid = int(df['post_date'].isna().sum())
    df = df.dropna(subset=['post_date', 'tweetId'])
    df['day'] = df['post_date'].dt.floor('D')
    daily = df.groupby('day')['tweetId'].nunique().sort_index().asfreq('D', fill_value=0)
    if len(daily) < WINDOW + HORIZON:
        raise ValueError(f'Need at least {WINDOW + HORIZON} days; found {len(daily)}')

    records = []
    # Origins separated by 7 days: non-overlapping evaluation target windows.
    # Start at first origin with 7 days of history, and require 7 future observed days.
    for end_idx in range(WINDOW - 1, len(daily) - HORIZON, HORIZON):
        history = daily.iloc[end_idx - WINDOW + 1:end_idx + 1]
        future = daily.iloc[end_idx + 1:end_idx + 1 + HORIZON]
        estimate = float(history.mean())
        origin = daily.index[end_idx]
        for lead, (date, actual) in enumerate(future.items(), start=1):
            records.append({
                'forecast_origin_utc': origin.date().isoformat(),
                'forecast_date_utc': date.date().isoformat(),
                'horizon_day': lead,
                'predicted_positive_tweets': estimate,
                'actual_positive_tweets': int(actual),
                'absolute_error': abs(float(actual) - estimate),
                'squared_error': (float(actual) - estimate) ** 2,
            })
    results = pd.DataFrame.from_records(records)
    if results.empty:
        raise ValueError('No complete 7-day forecast windows were found.')
    mae = float(results['absolute_error'].mean())
    rmse = float(np.sqrt(results['squared_error'].mean()))
    actual_sum = float(results['actual_positive_tweets'].sum())
    wape = 100.0 * float(results['absolute_error'].sum()) / actual_sum if actual_sum > 0 else float('nan')

    # The actual dashboard prediction, after the last historical day, is NOT testable yet.
    future_index = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=HORIZON, freq='D')
    current_estimate = float(daily.iloc[-WINDOW:].mean())
    future_forecast = pd.DataFrame({
        'forecast_date_utc': future_index.date.astype(str),
        'predicted_positive_tweets': [current_estimate] * HORIZON,
    })

    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / 'historical_7day_predictions.csv', index=False)
    future_forecast.to_csv(OUT / 'dashboard_next_7days.csv', index=False)
    pd.DataFrame([{
        'method': '7-day trailing mean; fixed prediction for all 7 future days',
        'evaluation': 'non-overlapping rolling-origin 7-day backtest',
        'backtest_windows': int(len(results) // HORIZON),
        'evaluated_days': int(len(results)),
        'mae_tweets_per_day': mae,
        'rmse_tweets_per_day': rmse,
        'wape_percent': wape,
        'historical_first_day_utc': daily.index.min().date().isoformat(),
        'historical_last_day_utc': daily.index.max().date().isoformat(),
        'invalid_post_dates_dropped': invalid,
    }]).to_csv(OUT / 'metrics.csv', index=False)

    print('\n=== HISTORICAL 7-DAY FORECAST BACKTEST ===')
    print(f'Available daily data: {daily.index.min().date()} to {daily.index.max().date()}')
    print(f'Backtest windows: {len(results) // HORIZON}; evaluated days: {len(results)}')
    print(f'MAE:  {mae:.4f} positive tweets/day')
    print(f'RMSE: {rmse:.4f} positive tweets/day')
    print(f'WAPE: {wape:.2f}%' if np.isfinite(wape) else 'WAPE: undefined (zero actual total)')
    print(f'Latest dashboard forecast: {current_estimate:.2f} positive tweets/day')
    print(f'Forecast dates: {future_index.min().date()} to {future_index.max().date()}')
    print('\nBacktest predictions and metrics saved under:', OUT)
    print('Note: historical backtest error is NOT the error of the future seven dates.')


if __name__ == '__main__':
    main()
