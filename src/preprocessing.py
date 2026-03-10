"""
preprocessing.py
----------------
Cleans a validated DataFrame by filling date gaps and removing likely
store-closure days before feature engineering.
"""

import pandas as pd
import numpy as np


def _get_group_cols(df: pd.DataFrame) -> list:
    """Return the list of grouping columns appropriate for this dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Any validated DataFrame that may or may not contain *product_id* /
        *store_id* columns.

    Returns
    -------
    list
        A list of column names to group by (may be empty for single-series
        datasets).
    """
    cols = []
    if "product_id" in df.columns and "store_id" in df.columns:
        cols = ["product_id", "store_id"]
    elif "product_id" in df.columns:
        cols = ["product_id"]
    elif "store_id" in df.columns:
        cols = ["store_id"]
    return cols


def detect_frequency(df: pd.DataFrame) -> str:
    """Detect the dominant time step of the dataset.

    Examines the median gap between consecutive dates in the highest-volume
    series and returns ``'W'`` when the gap is ≥ 5 days (weekly data) or
    ``'D'`` for daily data.

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame containing a *date* column.

    Returns
    -------
    str
        ``'D'`` (daily) or ``'W'`` (weekly).
    """
    group_cols = _get_group_cols(df)
    if group_cols:
        totals = df.groupby(group_cols)["sales_qty"].sum()
        best_key = totals.idxmax()
        if isinstance(best_key, tuple):
            mask = pd.Series(True, index=df.index)
            for col, val in zip(group_cols, best_key):
                mask &= df[col] == val
            sample = df[mask]
        else:
            sample = df[df[group_cols[0]] == best_key]
    else:
        sample = df

    diffs = sample.sort_values("date")["date"].diff().dropna()
    if diffs.empty:
        return "D"
    return "W" if diffs.dt.days.median() >= 5 else "D"


def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each group to a continuous date range and impute missing sales.

    For every group (determined by product_id / store_id presence):
    * A complete date range is created from the group's min to max date.
    * Missing *sales_qty* values are filled by forward-filling up to 2 days,
      then any remaining gaps are filled with the group's median sales.
    * A boolean column *is_imputed* is added to mark rows that were created
      during reindexing.

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with continuous daily dates, imputed sales, and an
        *is_imputed* flag column.
    """
    group_cols = _get_group_cols(df)
    freq = detect_frequency(df)

    def _fill_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.set_index("date").sort_index()
        if freq == "W":
            # Preserve original weekday cadence (e.g. Fridays for Walmart)
            start, end = group.index.min(), group.index.max()
            n = max(1, int(round((end - start).days / 7)) + 1)
            full_range = pd.date_range(start=start, periods=n, freq="7D")
        else:
            full_range = pd.date_range(
                group.index.min(), group.index.max(), freq="D"
            )
        group = group.reindex(full_range)
        group.index.name = "date"

        # Track which rows were synthetically created
        group["is_imputed"] = group["sales_qty"].isna()

        # Forward fill (1 period for weekly, 2 days for daily), then median
        ffill_limit = 1 if freq == "W" else 2
        median_sales = group["sales_qty"].median()
        group["sales_qty"] = group["sales_qty"].ffill(limit=ffill_limit)
        group["sales_qty"] = group["sales_qty"].fillna(median_sales)

        # Fill optional flag/numeric columns so Prophet/XGBoost never see NaN
        for flag_col in ["holiday_flag", "is_promotion"]:
            if flag_col in group.columns:
                group[flag_col] = group[flag_col].fillna(0).astype(int)
        if "price" in group.columns:
            group["price"] = group["price"].ffill().bfill()

        return group.reset_index()

    if not group_cols:
        # Single series
        filled = _fill_group(df.copy())
    else:
        parts = []
        for key, grp in df.groupby(group_cols, sort=False):
            grp_filled = _fill_group(grp.copy())
            # Re-attach group key columns
            if isinstance(key, tuple):
                for col, val in zip(group_cols, key):
                    grp_filled[col] = val
            else:
                grp_filled[group_cols[0]] = key
            parts.append(grp_filled)
        filled = pd.concat(parts, ignore_index=True)

    sort_cols = [c for c in ["store_id", "product_id", "date"] if c in filled.columns]
    filled = filled.sort_values(sort_cols).reset_index(drop=True)
    return filled


def remove_closed_store_days(df: pd.DataFrame) -> pd.DataFrame:
    """Remove consecutive zero-sales blocks that likely represent store closures.

    Isolated single-day zero-sales rows are kept (they may represent genuine
    low demand). Only blocks of **two or more** consecutive days with
    ``sales_qty == 0`` are removed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame (may contain an *is_imputed* column from
        ``fill_missing_dates``).

    Returns
    -------
    pd.DataFrame
        DataFrame with closure-day rows removed and the index reset.
    """
    group_cols = _get_group_cols(df)

    def _remove_closures(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").reset_index(drop=True)
        is_zero = group["sales_qty"] == 0

        # Label consecutive runs
        run_id = (is_zero != is_zero.shift()).cumsum()
        run_lengths = is_zero.groupby(run_id).transform("sum")

        # Keep a zero row only when its run length is exactly 1
        mask_keep = ~(is_zero & (run_lengths >= 2))
        return group[mask_keep]

    if not group_cols:
        result = _remove_closures(df.copy())
    else:
        parts = []
        for _, grp in df.groupby(group_cols, sort=False):
            parts.append(_remove_closures(grp.copy()))
        result = pd.concat(parts, ignore_index=True)

    return result.sort_values("date").reset_index(drop=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline on a validated DataFrame.

    Chains ``fill_missing_dates`` → ``remove_closed_store_days``.

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame returned by ``data_validation.validate_dataset``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    freq = detect_frequency(df)
    df = fill_missing_dates(df)
    # Store-closure removal is only meaningful for daily data
    if freq == "D":
        df = remove_closed_store_days(df)
    return df
