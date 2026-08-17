"""
Data loading module for the Loan Approval Prediction project.

This module handles loading the dataset from disk, performing
basic validation, and returning a clean DataFrame ready for
further processing.
"""

import pandas as pd

from config import DATA_FILE


# ======================================================
# Constants
# ======================================================

MINIMUM_REQUIRED_ROWS = 50


# ======================================================
# Data Loading
# ======================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the loan dataset from the configured CSV file path.

    Returns:
        pd.DataFrame: Raw dataset loaded from disk.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or too small to be useful.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_FILE}\n"
            "Please place 'loan_data.csv' inside the 'data/' folder."
        )

    df = pd.read_csv(DATA_FILE)

    _validate_dataset(df)

    print(f"[Data Loader] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def _validate_dataset(df: pd.DataFrame) -> None:
    """
    Perform basic validation checks on the loaded dataset.

    Args:
        df (pd.DataFrame): The loaded DataFrame to validate.

    Raises:
        ValueError: If the dataset fails any validation check.
    """
    if df.empty:
        raise ValueError("The dataset is empty. Please provide a valid loan_data.csv file.")

    if df.shape[0] < MINIMUM_REQUIRED_ROWS:
        raise ValueError(
            f"Dataset has only {df.shape[0]} rows. "
            f"A minimum of {MINIMUM_REQUIRED_ROWS} rows is required for training."
        )

    if df.shape[1] < 2:
        raise ValueError(
            "Dataset must have at least one feature column and one target column."
        )


# ======================================================
# Dataset Inspection Utilities
# ======================================================

def get_basic_info(df: pd.DataFrame) -> dict:
    """
    Collect basic information about the dataset.

    Args:
        df (pd.DataFrame): The loaded DataFrame.

    Returns:
        dict: A dictionary containing basic dataset statistics.
    """
    return {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Attempt to automatically detect the target column in the dataset.

    The function looks for common target column name patterns used in
    loan approval datasets. If none are found, it defaults to the
    last column in the DataFrame.

    Args:
        df (pd.DataFrame): The loaded DataFrame.

    Returns:
        str: The detected target column name.
    """
    common_target_names = [
        "loan_status",
        "loan_approved",
        "approved",
        "status",
        "target",
        "label",
        "class",
        "default",
        "outcome",
    ]

    for col in df.columns:
        if col.strip().lower() in common_target_names:
            print(f"[Data Loader] Target column detected: '{col}'")
            return col

    # Fallback: use the last column
    last_col = df.columns[-1]
    print(f"[Data Loader] No known target column found. Defaulting to last column: '{last_col}'")
    return last_col


def get_feature_columns(df: pd.DataFrame, target_column: str) -> list:
    """
    Return all feature columns by excluding the target column.

    Args:
        df (pd.DataFrame): The full DataFrame.
        target_column (str): The name of the target column.

    Returns:
        list: List of feature column names.
    """
    features = [col for col in df.columns if col != target_column]
    print(f"[Data Loader] Feature columns identified: {features}")
    return features