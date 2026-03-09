"""
storage.py
----------
Handles persistence of uploaded datasets, processed data, trained models,
forecast outputs, and the upload log.
"""

import os
import json
import joblib
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Base paths — resolved relative to this file so they work regardless of the
# working directory from which the Streamlit app is launched.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

_RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
_OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")
_UPLOAD_LOG = os.path.join(_PROCESSED_DIR, "upload_log.json")


def _timestamp() -> str:
    """Return the current UTC timestamp as a compact string.

    Returns
    -------
    str
        Format: ``YYYYMMDD_HHMMSS``
    """
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_uploaded_dataset(df: pd.DataFrame, original_filename: str) -> str:
    """Save the raw uploaded DataFrame to the *data/raw/* directory.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    original_filename : str
        Original file name supplied by the user (used as-is for the saved
        file name).

    Returns
    -------
    str
        Absolute path of the saved CSV file.
    """
    os.makedirs(_RAW_DIR, exist_ok=True)
    safe_name = os.path.basename(original_filename)
    filepath = os.path.join(_RAW_DIR, safe_name)
    df.to_csv(filepath, index=False)
    return filepath


def save_processed_data(df: pd.DataFrame, original_filename: str) -> str:
    """Save a processed DataFrame and append an entry to the upload log.

    The saved file name is prefixed with ``processed_<timestamp>_`` so
    successive uploads of the same file do not overwrite each other.

    The upload log (``data/processed/upload_log.json``) records:

    * ``filename``   – the original upload file name.
    * ``saved_path`` – absolute path of the saved processed file.
    * ``row_count``  – number of rows in the processed DataFrame.
    * ``timestamp``  – UTC timestamp string.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame to save.
    original_filename : str
        Original file name supplied by the user.

    Returns
    -------
    str
        Absolute path of the saved processed CSV file.
    """
    os.makedirs(_PROCESSED_DIR, exist_ok=True)
    ts = _timestamp()
    safe_name = os.path.basename(original_filename)
    filename = f"processed_{ts}_{safe_name}"
    filepath = os.path.join(_PROCESSED_DIR, filename)
    df.to_csv(filepath, index=False)

    log_entry = {
        "filename": safe_name,
        "saved_path": filepath,
        "row_count": len(df),
        "timestamp": ts,
    }

    existing_log = load_upload_log()
    existing_log.append(log_entry)

    with open(_UPLOAD_LOG, "w", encoding="utf-8") as fh:
        json.dump(existing_log, fh, indent=2)

    return filepath


def save_model(model, model_name: str) -> str:
    """Serialise a trained model to the *models/* directory using joblib.

    Parameters
    ----------
    model : any sklearn-compatible or Prophet model
        Fitted model object.
    model_name : str
        Short human-readable name used in the file name (e.g. ``'Prophet'``
        or ``'XGBoost'``).

    Returns
    -------
    str
        Absolute path of the saved ``.pkl`` file.
    """
    os.makedirs(_MODELS_DIR, exist_ok=True)
    ts = _timestamp()
    filename = f"{model_name}_{ts}.pkl"
    filepath = os.path.join(_MODELS_DIR, filename)
    joblib.dump(model, filepath)
    return filepath


def save_forecast_output(forecast_df: pd.DataFrame, original_filename: str) -> str:
    """Save a forecast DataFrame to the *outputs/* directory.

    The saved file name is prefixed with ``forecast_<timestamp>_`` so
    multiple forecasts for the same dataset are preserved separately.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame containing forecast results (date, predicted_sales, etc.).
    original_filename : str
        Original upload file name (used as part of the output file name).

    Returns
    -------
    str
        Absolute path of the saved forecast CSV file.
    """
    os.makedirs(_OUTPUTS_DIR, exist_ok=True)
    ts = _timestamp()
    safe_name = os.path.basename(original_filename)
    filename = f"forecast_{ts}_{safe_name}"
    filepath = os.path.join(_OUTPUTS_DIR, filename)
    forecast_df.to_csv(filepath, index=False)
    return filepath


def load_upload_log() -> list:
    """Load and return the upload log as a list of dicts.

    Returns an empty list when the log file does not yet exist.

    Returns
    -------
    list[dict]
        Each element is a log entry with keys *filename*, *saved_path*,
        *row_count*, and *timestamp*.
    """
    if not os.path.isfile(_UPLOAD_LOG):
        return []
    with open(_UPLOAD_LOG, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return []
