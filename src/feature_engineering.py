"""
feature_engineering.py
-----------------------
Creates time, lag, and rolling features on a cleaned DataFrame.
All lag and rolling computations are grouped to prevent data leakage
between different products or stores.
"""

import pandas as pd
import numpy as np


def _get_group_cols(df: pd.DataFrame) -> list:
    """Return the list of columns to group by.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain *product_id* and/or *store_id*.

    Returns
    -------
    list
        Column names to group by; empty list for a single-series dataset.
    """
    cols = []
    if "product_id" in df.columns and "store_id" in df.columns:
        cols = ["product_id", "store_id"]
    elif "product_id" in df.columns:
        cols = ["product_id"]
    elif "store_id" in df.columns:
        cols = ["store_id"]
    return cols


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based time features derived from the *date* column.

    New columns added:
    * ``day_of_week``     – Integer 0 (Monday) … 6 (Sunday).
    * ``month``           – Integer 1 … 12.
    * ``week_of_year``    – ISO week number (1 … 53).
    * ``quarter``         – Integer 1 … 4.
    * ``is_weekend``      – 1 if Saturday or Sunday, else 0.
    * ``is_month_start``  – 1 on the first calendar day of a month, else 0.
    * ``is_month_end``    – 1 on the last calendar day of a month, else 0.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with a *date* column of dtype datetime64.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with the seven new columns appended.
    """
    df = df.copy()
    dates = pd.to_datetime(df["date"])

    df["day_of_week"] = dates.dt.dayofweek
    df["month"] = dates.dt.month
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["quarter"] = dates.dt.quarter
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dates.dt.is_month_start.astype(int)
    df["is_month_end"] = dates.dt.is_month_end.astype(int)

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged sales features, computed within each group.

    Shifting is performed **within** each group so that lag values for
    one product/store never bleed into another (no data leakage).

    New columns added:
    * ``sales_lag_7``  – Sales 7 days ago.
    * ``sales_lag_14`` – Sales 14 days ago.
    * ``sales_lag_28`` – Sales 28 days ago.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with a *sales_qty* column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the three lag columns appended.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    group_cols = _get_group_cols(df)

    for lag in [7, 14, 28]:
        col = f"sales_lag_{lag}"
        if group_cols:
            df[col] = df.groupby(group_cols)["sales_qty"].shift(lag)
        else:
            df[col] = df["sales_qty"].shift(lag)

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling-window statistics, computed within each group.

    The series is shifted by 1 before the rolling window is applied so
    that the current-day value is never included in its own rolling
    statistic (no data leakage).

    New columns added:
    * ``rolling_mean_7``  – 7-day rolling mean (shifted by 1).
    * ``rolling_mean_28`` – 28-day rolling mean (shifted by 1).
    * ``rolling_std_7``   – 7-day rolling standard deviation (shifted by 1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains lag features.

    Returns
    -------
    pd.DataFrame
        DataFrame with the three rolling columns appended.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    group_cols = _get_group_cols(df)

    def _rolling_stats(series: pd.Series) -> pd.DataFrame:
        shifted = series.shift(1)
        mean_7 = shifted.rolling(window=7, min_periods=1).mean()
        mean_28 = shifted.rolling(window=28, min_periods=1).mean()
        std_7 = shifted.rolling(window=7, min_periods=1).std()
        return pd.DataFrame(
            {
                "rolling_mean_7": mean_7,
                "rolling_mean_28": mean_28,
                "rolling_std_7": std_7,
            }
        )

    if group_cols:
        rolling_parts = []
        for _, grp in df.groupby(group_cols, sort=False):
            stats = _rolling_stats(grp["sales_qty"])
            stats.index = grp.index
            rolling_parts.append(stats)
        rolling_df = pd.concat(rolling_parts).sort_index()
    else:
        rolling_df = _rolling_stats(df["sales_qty"])

    df["rolling_mean_7"] = rolling_df["rolling_mean_7"].values
    df["rolling_mean_28"] = rolling_df["rolling_mean_28"].values
    df["rolling_std_7"] = rolling_df["rolling_std_7"].values

    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering steps to a cleaned DataFrame.

    Calls ``create_time_features`` → ``create_lag_features`` →
    ``create_rolling_features`` in that order. Optional columns
    (*is_promotion*, *holiday_flag*, *price*) are passed through unchanged
    when they already exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame returned by ``preprocessing.clean_dataframe``.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with all engineered features.
    """
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    # Optional columns are already present; no extra action needed.
    return df
