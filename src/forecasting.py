"""
forecasting.py
--------------
Trains Prophet and XGBoost models, evaluates them, selects the best one,
and generates a forward-looking forecast DataFrame.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZON = 42  # forecast horizon in days
SPLIT_DAYS = 42  # test-set size for time-based evaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_freq(df: pd.DataFrame) -> str:
    """Return 'W' for weekly data, 'D' for daily (based on median date gap)."""
    group_cols = []
    if "product_id" in df.columns and "store_id" in df.columns:
        group_cols = ["product_id", "store_id"]
    elif "product_id" in df.columns:
        group_cols = ["product_id"]
    elif "store_id" in df.columns:
        group_cols = ["store_id"]

    if group_cols:
        totals = df.groupby(group_cols)["sales_qty"].sum()
        best = totals.idxmax()
        if isinstance(best, tuple):
            mask = pd.Series(True, index=df.index)
            for c, v in zip(group_cols, best):
                mask &= df[c] == v
            sample = df[mask]
        else:
            sample = df[df[group_cols[0]] == best]
    else:
        sample = df

    diffs = sample.sort_values("date")["date"].diff().dropna()
    if diffs.empty:
        return "D"
    return "W" if diffs.dt.days.median() >= 5 else "D"


def _get_representative_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return the single time-series with the highest total sales volume.

    For multi-series datasets Prophet is trained on the largest volume series
    so it captures seasonality as representatively as possible.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.

    Returns
    -------
    pd.DataFrame
        Subset containing only the highest-volume series (or the full
        DataFrame when no grouping columns are present).
    """
    group_cols = []
    if "product_id" in df.columns and "store_id" in df.columns:
        group_cols = ["product_id", "store_id"]
    elif "product_id" in df.columns:
        group_cols = ["product_id"]
    elif "store_id" in df.columns:
        group_cols = ["store_id"]

    if not group_cols:
        return df

    totals = df.groupby(group_cols)["sales_qty"].sum()
    best_key = totals.idxmax()

    if isinstance(best_key, tuple):
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in zip(group_cols, best_key):
            mask = mask & (df[col] == val)
    else:
        mask = df[group_cols[0]] == best_key

    return df[mask].copy()


def get_representative_series(df: pd.DataFrame) -> tuple:
    """Public wrapper: return the representative series and a human-readable label.

    Returns
    -------
    tuple[pd.DataFrame, str]
        (filtered_df, label) where label describes which store/product was selected.
    """
    group_cols = []
    if "product_id" in df.columns and "store_id" in df.columns:
        group_cols = ["product_id", "store_id"]
    elif "product_id" in df.columns:
        group_cols = ["product_id"]
    elif "store_id" in df.columns:
        group_cols = ["store_id"]

    if not group_cols:
        return df.copy(), "All data"

    totals = df.groupby(group_cols)["sales_qty"].sum()
    best_key = totals.idxmax()

    if isinstance(best_key, tuple):
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in zip(group_cols, best_key):
            mask = mask & (df[col] == val)
        label = " / ".join(f"{c}={v}" for c, v in zip(group_cols, best_key))
    else:
        mask = df[group_cols[0]] == best_key
        label = f"{group_cols[0]}={best_key}"

    return df[mask].copy(), label


def _time_split(df: pd.DataFrame) -> tuple:
    """Split a DataFrame into train/test using a time-based cut.

    The last ``SPLIT_DAYS`` rows (by date) form the test set; everything
    before forms the training set.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sorted by date, containing a *date* column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    df = df.sort_values("date").reset_index(drop=True)
    cutoff = df["date"].max() - pd.Timedelta(days=SPLIT_DAYS)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute MAE, RMSE, and MAPE between actual and predicted values.

    Zero-valued actuals are excluded from the MAPE calculation to avoid
    division-by-zero errors.

    Parameters
    ----------
    actual : pd.Series
        Ground-truth sales quantities.
    predicted : pd.Series
        Model-predicted sales quantities.

    Returns
    -------
    dict
        ``{'MAE': float, 'RMSE': float, 'MAPE': float}`` where MAPE is
        expressed as a percentage (e.g. 12.5 means 12.5 %).
    """
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    non_zero = actual != 0
    if non_zero.sum() == 0:
        mape = 0.0
    else:
        mape = float(
            np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero]))
            * 100
        )

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def train_prophet(df: pd.DataFrame) -> tuple:
    """Train a Prophet model on the representative (highest-volume) series.

    Configuration:
    * ``yearly_seasonality=True``
    * ``weekly_seasonality=True``
    * ``daily_seasonality=False``
    * ``seasonality_mode='multiplicative'``
    * ``changepoint_prior_scale=0.05``

    Additional regressors (*is_promotion*, *holiday_flag*) are added
    automatically when those columns are present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.

    Returns
    -------
    tuple[Prophet, dict]
        The fitted Prophet model and a metrics dictionary
        ``{'MAE': …, 'RMSE': …, 'MAPE': …}``.
    """
    series = _get_representative_series(df)
    train, test = _time_split(series)

    regressors = []
    if "is_promotion" in series.columns:
        regressors.append("is_promotion")
    if "holiday_flag" in series.columns:
        regressors.append("holiday_flag")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )

    for reg in regressors:
        model.add_regressor(reg)

    prophet_train = train[["date", "sales_qty"] + regressors].rename(
        columns={"date": "ds", "sales_qty": "y"}
    )
    model.fit(prophet_train)

    # Build future DataFrame for the test period only
    future = test[["date"] + regressors].rename(columns={"date": "ds"})
    forecast = model.predict(future)

    predicted = forecast["yhat"].values
    actual = test["sales_qty"].values
    metrics = evaluate_forecast(actual, predicted)

    return model, metrics


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(df: pd.DataFrame) -> tuple:
    """Train an XGBoost regressor using engineered features.

    Feature columns used (when present):
    * Lag features: ``sales_lag_7``, ``sales_lag_14``, ``sales_lag_28``
    * Rolling features: ``rolling_mean_7``, ``rolling_mean_28``,
      ``rolling_std_7``
    * Calendar features: ``day_of_week``, ``month``, ``week_of_year``,
      ``quarter``, ``is_weekend``, ``is_month_start``, ``is_month_end``
    * Optional: ``price``, ``is_promotion``, ``holiday_flag``

    Config: ``n_estimators=500``, ``learning_rate=0.05``, ``max_depth=6``,
    ``subsample=0.8``, ``colsample_bytree=0.8``, ``random_state=42``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.

    Returns
    -------
    tuple[XGBRegressor, dict, list]
        The fitted XGBoost model, a metrics dictionary
        ``{'MAE': …, 'RMSE': …, 'MAPE': …}``, and the list of feature
        column names used during training.
    """
    candidate_features = [
        "day_of_week", "month", "week_of_year", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
        "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "rolling_mean_7", "rolling_mean_28", "rolling_std_7",
        "price", "is_promotion", "holiday_flag",
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]

    # Use the full multi-series data for XGBoost (it learns patterns globally)
    df_clean = df[["date", "sales_qty"] + feature_cols].dropna(
        subset=feature_cols
    ).copy()
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    train, test = _time_split(df_clean)

    if train.empty or test.empty:
        raise ValueError(
            "Not enough data to perform a time-based train/test split for XGBoost."
        )

    X_train = train[feature_cols]
    y_train = train["sales_qty"]
    X_test = test[feature_cols]
    y_test = test["sales_qty"]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    metrics = evaluate_forecast(y_test.values, predicted)

    return model, metrics, feature_cols


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_best_model(results: dict) -> str:
    """Return the name of the model with the lowest MAPE.

    Parameters
    ----------
    results : dict
        Mapping of model name → metrics dict, e.g.::

            {
                'Prophet': {'MAE': …, 'RMSE': …, 'MAPE': …},
                'XGBoost': {'MAE': …, 'RMSE': …, 'MAPE': …},
            }

    Returns
    -------
    str
        ``'Prophet'`` or ``'XGBoost'`` (or whichever name has the lowest
        MAPE in the supplied dict).
    """
    best = min(results, key=lambda name: results[name]["MAPE"])
    return best


# ---------------------------------------------------------------------------
# Forecast generation
# ---------------------------------------------------------------------------

def generate_forecast(
    model,
    df: pd.DataFrame,
    model_name: str,
    feature_cols: list,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Generate a forward-looking forecast from the last known date.

    Parameters
    ----------
    model : Prophet or XGBRegressor
        Fitted model returned by ``train_prophet`` or ``train_xgboost``.
    df : pd.DataFrame
        Full feature-engineered DataFrame (used for the last-known values
        required by XGBoost lag features).
    model_name : str
        ``'Prophet'`` or ``'XGBoost'``.
    feature_cols : list
        List of feature column names used when training XGBoost.
        Ignored for Prophet.
    horizon : int, optional
        Number of days to forecast (default 42).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        * ``date``            – future calendar dates.
        * ``predicted_sales`` – point forecast (clipped to ≥ 0).
        * ``lower_bound``     – lower confidence bound.
        * ``upper_bound``     – upper confidence bound.
    """
    if model_name == "Prophet":
        return _forecast_prophet(model, df, horizon)
    elif model_name == "XGBoost":
        return _forecast_xgboost(model, df, feature_cols, horizon)
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'")


def _forecast_prophet(model: Prophet, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Generate a Prophet forecast.

    Parameters
    ----------
    model : Prophet
        Fitted Prophet model.
    df : pd.DataFrame
        Historical DataFrame containing at least a *date* column and any
        regressor columns the model was trained with.
    horizon : int
        Number of future days to forecast.

    Returns
    -------
    pd.DataFrame
        Columns: *date*, *predicted_sales*, *lower_bound*, *upper_bound*.
    """
    # Use weekly cadence when the source data is weekly
    data_freq = _detect_freq(df)
    if data_freq == "W":
        future = model.make_future_dataframe(periods=max(1, horizon // 7), freq="7D")
    else:
        future = model.make_future_dataframe(periods=horizon, freq="D")

    # Populate regressors on future dates using the most recent known values
    for reg in model.extra_regressors:
        if reg in df.columns:
            # Merge known values; fill future dates with the last known value
            reg_series = df[["date", reg]].drop_duplicates("date").rename(
                columns={"date": "ds"}
            )
            future = future.merge(reg_series, on="ds", how="left")
            future[reg] = future[reg].ffill().fillna(0)

    raw = model.predict(future)
    forecast_rows = raw.tail(horizon).copy()

    result = pd.DataFrame(
        {
            "date": forecast_rows["ds"].values,
            "predicted_sales": np.clip(forecast_rows["yhat"].values, 0, None),
            "lower_bound": np.clip(forecast_rows["yhat_lower"].values, 0, None),
            "upper_bound": np.clip(forecast_rows["yhat_upper"].values, 0, None),
        }
    )
    return result


def _forecast_xgboost(
    model: XGBRegressor,
    df: pd.DataFrame,
    feature_cols: list,
    horizon: int,
) -> pd.DataFrame:
    """Generate an XGBoost forecast using iterative lag propagation.

    Future lag features are computed iteratively: as each day is predicted,
    its forecasted value is appended to the history buffer so that
    subsequent lag features reference it correctly.

    Parameters
    ----------
    model : XGBRegressor
        Fitted XGBoost model.
    df : pd.DataFrame
        Full historical DataFrame containing all engineered features.
    feature_cols : list
        Ordered list of feature columns used during training.
    horizon : int
        Number of future days to forecast.

    Returns
    -------
    pd.DataFrame
        Columns: *date*, *predicted_sales*, *lower_bound*, *upper_bound*.
    """
    # Use the representative (single-store) series so that the history buffer
    # is a proper time series and lag_7 truly means "7 days ago".
    series_df = _get_representative_series(df).sort_values("date").reset_index(drop=True)
    last_date = series_df["date"].max()

    # Detect frequency to set correct lag sizes, window sizes, and step cadence
    data_freq = _detect_freq(df)
    if data_freq == "W":
        # 1 row = 1 week: lag_7=1row, lag_14=2rows, lag_28=4rows; windows 4/13
        lag7, lag14, lag28 = 1, 2, 4
        win_short, win_long = 4, 13
        max_lag = 4
        n_periods = max(1, horizon // 7)
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=7), periods=n_periods, freq="7D"
        )
    else:
        lag7, lag14, lag28 = 7, 14, 28
        win_short, win_long = 7, 28
        max_lag = 28
        n_periods = horizon
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=n_periods, freq="D"
        )

    # Build a rolling history of sales for lag computation
    history = list(series_df["sales_qty"].tail(max_lag + n_periods).values)

    # Retain the last row's optional features (price, promotion, holiday)
    last_row = series_df.iloc[-1]
    optional_static = {
        col: last_row[col]
        for col in ["price", "is_promotion", "holiday_flag"]
        if col in series_df.columns
    }

    predictions = []
    for i, fdate in enumerate(future_dates):
        row = {}
        # Calendar features
        row["day_of_week"] = fdate.dayofweek
        row["month"] = fdate.month
        row["week_of_year"] = fdate.isocalendar()[1]
        row["quarter"] = fdate.quarter
        row["is_weekend"] = int(fdate.dayofweek >= 5)
        row["is_month_start"] = int(fdate.is_month_start)
        row["is_month_end"] = int(fdate.is_month_end)

        # Lag features from rolling history (row-counts match training shifts)
        offset = len(history)
        row["sales_lag_7"] = history[offset - lag7] if offset >= lag7 else np.nan
        row["sales_lag_14"] = history[offset - lag14] if offset >= lag14 else np.nan
        row["sales_lag_28"] = history[offset - lag28] if offset >= lag28 else np.nan

        # Rolling features (shifted by 1 period before window)
        win_s = [history[j] for j in range(max(0, offset - win_short - 1), offset - 1)]
        win_l = [history[j] for j in range(max(0, offset - win_long - 1), offset - 1)]
        row["rolling_mean_7"] = float(np.mean(win_s)) if win_s else 0.0
        row["rolling_mean_28"] = float(np.mean(win_l)) if win_l else 0.0
        row["rolling_std_7"] = float(np.std(win_s)) if len(win_s) > 1 else 0.0

        # Optional static features
        for col, val in optional_static.items():
            row[col] = val

        # Build feature vector in the exact order used during training
        feature_vector = []
        for col in feature_cols:
            feature_vector.append(row.get(col, 0.0))

        X = np.array(feature_vector, dtype=float).reshape(1, -1)
        pred = float(model.predict(X)[0])
        pred = max(pred, 0.0)  # clip negatives
        predictions.append(pred)
        history.append(pred)

    predictions_arr = np.array(predictions)
    result = pd.DataFrame(
        {
            "date": future_dates,
            "predicted_sales": predictions_arr,
            "lower_bound": predictions_arr * 0.9,
            "upper_bound": predictions_arr * 1.1,
        }
    )
    return result
