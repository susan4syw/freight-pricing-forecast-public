"""Calendar and daily lag features adapted from the original internship pipeline.

Targets from the current day are excluded. Missing dates are explicitly reindexed.
Only reusable functions are included; internal documentation and data are omitted.
"""
from __future__ import annotations
import datetime as dt
import pandas as pd

LAG_DAYS = [7, 30]
ROLLING_WINDOWS = [7, 30]
NEAR_HOLIDAY_WINDOW = 3

def add_calendar_features(df: pd.DataFrame, date_col: str, holidays: set[dt.date]) -> pd.DataFrame:
    df = df.copy()
    dates = df[date_col].dt.date
    df['is_weekend'] = df[date_col].dt.weekday >= 5
    df['is_holiday'] = dates.isin(holidays)
    near_holiday = pd.Series(False, index=df.index)
    for offset in range(1, NEAR_HOLIDAY_WINDOW + 1):
        near_holiday |= (dates + pd.Timedelta(days=offset)).isin(holidays)
        near_holiday |= (dates - pd.Timedelta(days=offset)).isin(holidays)
    df['is_near_holiday'] = near_holiday & ~df['is_holiday']
    df['is_month_end'] = df[date_col].dt.is_month_end
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end & df['is_month_end']
    return df

def add_lag_rolling_features(df: pd.DataFrame, group_col: str='lane_group', date_col: str='order_date', target_col: str='unit_price', lags: list[int]=LAG_DAYS, rolling_windows: list[int]=ROLLING_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    df['_date'] = df[date_col].dt.normalize()
    daily = df.groupby([group_col, '_date'])[target_col].mean().reset_index().rename(columns={target_col: '_daily_unit_price'})

    def _per_group(name, g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index('_date').sort_index().asfreq('D')
        for lag in lags:
            g[f'lag_{lag}'] = g['_daily_unit_price'].shift(lag)
        for w in rolling_windows:
            g[f'rolling_mean_{w}'] = g['_daily_unit_price'].shift(1).rolling(w, min_periods=1).mean()
        g = g.reset_index()
        g[group_col] = name
        return g
    daily_feat = pd.concat([_per_group(name, g) for name, g in daily.groupby(group_col, group_keys=False)], ignore_index=True)
    feature_cols = [f'lag_{lag}' for lag in lags] + [f'rolling_mean_{w}' for w in rolling_windows]
    df = df.merge(daily_feat[[group_col, '_date', *feature_cols]], on=[group_col, '_date'], how='left')
    return df.drop(columns=['_date'])
