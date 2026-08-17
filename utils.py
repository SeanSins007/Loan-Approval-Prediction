"""
Utility module for the Loan Approval Prediction project.

This module provides shared helper functions used across
the Flask application, including form data extraction,
input sanitization, dataset introspection for dynamic
form generation, and JSON metrics loading.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from config import METRICS_JSON_PATH, DATA_FILE


# ======================================================
# Form Data Extraction
# ======================================================

def extract_form_data(form_data: dict, feature_columns: list) -> dict:
    """
    Extract and sanitize relevant fields from a Flask request form.

    Only extracts keys that match known feature columns to avoid
    injecting unexpected data into the model.

    Args:
        form_data (dict): Raw form data from Flask request.form.
        feature_columns (list): List of expected feature column names.

    Returns:
        dict: Sanitized dictionary with only the relevant feature values.
    """
    extracted = {}

    for col in feature_columns:
        value = form_data.get(col, "").strip()
        extracted[col] = value if value != "" else None

    return extracted


# ======================================================
# Input Sanitization
# ======================================================

def sanitize_numeric(value, default=0.0) -> float:
    """
    Safely convert a value to float.

    Args:
        value: The raw value to convert.
        default (float): Fallback value if conversion fails.

    Returns:
        float: Converted float value or default.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sanitize_string(value: str, default: str = "") -> str:
    """
    Safely clean and return a string value.

    Args:
        value (str): The raw string value.
        default (str): Fallback value if input is None or empty.

    Returns:
        str: Cleaned string or default.
    """
    if value is None:
        return default
    return str(value).strip()


# ======================================================
# Dynamic Form Field Generation
# ======================================================

def get_form_field_metadata(
    numerical_cols: list,
    categorical_cols: list,
    df_sample: pd.DataFrame,
) -> dict:
    """
    Generate metadata for each form field to enable dynamic
    HTML form rendering in the Flask template.

    For numerical columns: provides min, max, and step values.
    For categorical columns: provides a list of unique options.

    Args:
        numerical_cols (list): List of numerical feature column names.
        categorical_cols (list): List of categorical feature column names.
        df_sample (pd.DataFrame): A sample of the dataset used to
                                   extract ranges and unique values.

    Returns:
        dict: A dictionary mapping each column name to its field metadata.
              Each entry contains:
                - "type": "number" or "select"
                - "label": Human-readable field label
                - For number: "min", "max", "step"
                - For select: "options" (list of unique string values)
    """
    field_metadata = {}

    for col in numerical_cols:
        col_data = pd.to_numeric(df_sample[col], errors="coerce").dropna()

        col_min = float(col_data.min()) if not col_data.empty else 0.0
        col_max = float(col_data.max()) if not col_data.empty else 100.0

        # Use integer step if column appears to contain whole numbers
        is_integer_col = col_data.apply(lambda x: x == int(x)).all()
        step = "1" if is_integer_col else "0.01"

        field_metadata[col] = {
            "type": "number",
            "label": _format_label(col),
            "min": col_min,
            "max": col_max,
            "step": step,
        }

    for col in categorical_cols:
        unique_values = (
            df_sample[col]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        unique_values = sorted([v for v in unique_values if v.lower() != "nan"])

        field_metadata[col] = {
            "type": "select",
            "label": _format_label(col),
            "options": unique_values,
        }

    return field_metadata


def _format_label(column_name: str) -> str:
    """
    Convert a raw column name into a human-readable label.

    Examples:
        "loan_amount"     -> "Loan Amount"
        "applicantIncome" -> "Applicant Income"
        "credit_history"  -> "Credit History"

    Args:
        column_name (str): Raw column name from the dataset.

    Returns:
        str: Human-readable label string.
    """
    # Replace underscores and hyphens with spaces
    label = column_name.replace("_", " ").replace("-", " ")

    # Insert space before uppercase letters (camelCase support)
    import re
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)

    return label.strip().title()


# ======================================================
# Load Dataset Sample for Form Generation
# ======================================================

def load_dataset_sample() -> pd.DataFrame:
    """
    Load the raw dataset from disk for use in form field generation.

    Returns:
        pd.DataFrame: The full dataset as a DataFrame.

    Raises:
        FileNotFoundError: If the dataset file is not found.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_FILE}\n"
            "Please place 'loan_data.csv' inside the 'data/' folder."
        )

    return pd.read_csv(DATA_FILE)


# ======================================================
# Load Metrics JSON
# ======================================================

def load_metrics() -> dict:
    """
    Load the saved metrics JSON file from the results folder.

    Returns:
        dict: Metrics data including best model name and scores.
              Returns an empty dict if the file does not exist.
    """
    if not METRICS_JSON_PATH.exists():
        return {}

    with open(METRICS_JSON_PATH, "r") as f:
        return json.load(f)


# ======================================================
# Validation Helpers
# ======================================================

def validate_user_input(
    user_input: dict,
    numerical_cols: list,
    categorical_cols: list,
) -> tuple[bool, list]:
    """
    Validate raw user input before passing it to the prediction pipeline.

    Checks:
        - No required field is empty or None
        - Numerical fields contain valid numeric values
        - Categorical fields are non-empty strings

    Args:
        user_input (dict): Extracted form data dictionary.
        numerical_cols (list): List of expected numerical column names.
        categorical_cols (list): List of expected categorical column names.

    Returns:
        tuple[bool, list]:
            - is_valid (bool): True if all fields pass validation.
            - errors (list): List of human-readable error messages.
    """
    errors = []

    for col in numerical_cols:
        value = user_input.get(col)

        if value is None or str(value).strip() == "":
            errors.append(f"'{_format_label(col)}' is required.")
            continue

        try:
            float(value)
        except (TypeError, ValueError):
            errors.append(f"'{_format_label(col)}' must be a valid number.")

    for col in categorical_cols:
        value = user_input.get(col)

        if value is None or str(value).strip() == "" or str(value).strip().lower() == "none":
            errors.append(f"'{_format_label(col)}' is required.")

    is_valid = len(errors) == 0
    return is_valid, errors


# ======================================================
# Numpy Type Converter (for JSON serialization)
# ======================================================

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types
    for safe JSON serialization.

    Args:
        obj: Any object that may contain NumPy types.

    Returns:
        Native Python equivalent of the input object.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj