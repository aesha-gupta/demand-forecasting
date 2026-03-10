"""
data_validation.py
------------------
Validates and normalises raw uploaded DataFrames before any processing occurs.
"""

import pandas as pd


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean an uploaded DataFrame.

    Steps (executed in order):
    1. Strip whitespace from all column names and lowercase them.
    2. Assert that *date* and *sales_qty* columns are present.
    3. Parse the *date* column as datetime.
    4. Cast *sales_qty* to numeric.
    5. Drop rows whose *sales_qty* is NaN after coercion.
    6. Raise an error when any remaining *sales_qty* value is negative.
    7. Sort the DataFrame by date ascending.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame read directly from the uploaded CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with normalised column names, parsed dates, and
        validated sales quantities.

    Raises
    ------
    ValueError
        When required columns are missing, the date column cannot be parsed,
        the sales_qty column cannot be coerced to numeric, or negative values
        are present in sales_qty.
    """
    # Step 1 — normalise column names
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Step 2 — check required columns
    required = {"date", "sales_qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Step 3 — parse dates
    # Try explicit formats in order: dd-mm-yyyy, yyyy-mm-dd, then dayfirst inference
    _FORMATS = ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"]
    parsed = None
    for fmt in _FORMATS:
        try:
            parsed = pd.to_datetime(df["date"], format=fmt)
            break
        except (ValueError, TypeError):
            continue
    if parsed is None:
        try:
            parsed = pd.to_datetime(df["date"], dayfirst=True)
        except Exception as exc:
            raise ValueError(
                f"Column 'date' could not be parsed as datetime: {exc}"
            ) from exc
    df["date"] = parsed

    # Step 4 — coerce sales_qty to numeric
    original_sales = df["sales_qty"].copy()
    df["sales_qty"] = pd.to_numeric(df["sales_qty"], errors="coerce")
    if df["sales_qty"].isna().any() and not original_sales.isna().all():
        non_numeric_count = df["sales_qty"].isna().sum()
        if non_numeric_count == len(df):
            raise ValueError(
                "Column 'sales_qty' could not be converted to numeric values."
            )

    # Step 5 — drop NaN sales rows
    df = df.dropna(subset=["sales_qty"])

    if df.empty:
        raise ValueError(
            "No valid rows remain after dropping NaN sales_qty values."
        )

    # Step 6 — reject negative sales
    if (df["sales_qty"] < 0).any():
        raise ValueError(
            "Column 'sales_qty' contains negative values, which are not allowed."
        )

    # Step 7 — sort by group columns then date (handles per-store sequential CSVs)
    sort_cols = []
    if "store_id" in df.columns:
        sort_cols.append("store_id")
    if "product_id" in df.columns:
        sort_cols.append("product_id")
    sort_cols.append("date")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def detect_optional_columns(df: pd.DataFrame) -> dict:
    """Inspect a validated DataFrame and return metadata about its contents.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that has already been processed by ``validate_dataset``.

    Returns
    -------
    dict
        Keys and their meanings:

        * ``has_product_id``  – True when a *product_id* column is present.
        * ``has_store_id``    – True when a *store_id* column is present.
        * ``has_price``       – True when a *price* column is present.
        * ``has_promotion``   – True when an *is_promotion* column is present.
        * ``has_holiday``     – True when a *holiday_flag* column is present.
        * ``date_range_days`` – Number of calendar days between the earliest
          and latest date in the dataset.
        * ``total_rows``      – Total number of rows in the DataFrame.
        * ``unique_products`` – Number of unique product_id values, or None.
        * ``unique_stores``   – Number of unique store_id values, or None.
    """
    cols = set(df.columns)

    has_product_id = "product_id" in cols
    has_store_id = "store_id" in cols

    date_range_days = int((df["date"].max() - df["date"].min()).days)

    return {
        "has_product_id": has_product_id,
        "has_store_id": has_store_id,
        "has_price": "price" in cols,
        "has_promotion": "is_promotion" in cols,
        "has_holiday": "holiday_flag" in cols,
        "date_range_days": date_range_days,
        "total_rows": len(df),
        "unique_products": df["product_id"].nunique() if has_product_id else None,
        "unique_stores": df["store_id"].nunique() if has_store_id else None,
    }
