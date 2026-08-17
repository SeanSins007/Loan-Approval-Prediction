"""
Feature engineering module for the Loan Approval Prediction project.

This module handles all feature engineering tasks including
detecting column types, handling missing values, removing duplicates,
encoding categorical variables, and scaling numerical features.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.data_loader import detect_target_column, get_feature_columns


# ======================================================
# Column Type Detection
# ======================================================

def detect_column_types(df: pd.DataFrame, target_column: str) -> tuple[list, list]:
    """
    Automatically detect numerical and categorical feature columns.

    Args:
        df (pd.DataFrame): The full DataFrame.
        target_column (str): The target column name to exclude.

    Returns:
        tuple[list, list]: A tuple of (numerical_columns, categorical_columns).
    """
    feature_cols = get_feature_columns(df, target_column)
    feature_df = df[feature_cols]

    numerical_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print(f"[Feature Engineering] Numerical columns : {numerical_cols}")
    print(f"[Feature Engineering] Categorical columns: {categorical_cols}")

    return numerical_cols, categorical_cols


# ======================================================
# Data Cleaning
# ======================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]

    removed = before - after
    if removed > 0:
        print(f"[Feature Engineering] Removed {removed} duplicate row(s).")
    else:
        print("[Feature Engineering] No duplicate rows found.")

    return df.reset_index(drop=True)


# ======================================================
# Target Encoding
# ======================================================

def encode_target(series: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode the target column using LabelEncoder.

    Handles both string labels (e.g., 'Y'/'N', 'Approved'/'Rejected')
    and numeric labels (e.g., 0/1).

    Args:
        series (pd.Series): The target column as a Pandas Series.

    Returns:
        tuple[np.ndarray, LabelEncoder]: Encoded target array and fitted encoder.
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(series.astype(str).str.strip())

    print(f"[Feature Engineering] Target classes: {list(encoder.classes_)}")
    print(f"[Feature Engineering] Encoded mapping: "
          f"{dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

    return encoded, encoder


# ======================================================
# Preprocessing Pipeline
# ======================================================

def build_preprocessing_pipeline(
    numerical_cols: list,
    categorical_cols: list
) -> ColumnTransformer:
    """
    Build a Scikit-Learn ColumnTransformer preprocessing pipeline.

    Numerical pipeline:
        - Impute missing values with the median
        - Scale features using StandardScaler

    Categorical pipeline:
        - Impute missing values with the most frequent value
        - Encode using OrdinalEncoder (compatible with all sklearn models)

    Args:
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical column names.

    Returns:
        ColumnTransformer: The assembled preprocessing pipeline.
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []

    if numerical_cols:
        transformers.append(("numerical", numerical_pipeline, numerical_cols))

    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    print("[Feature Engineering] Preprocessing pipeline built successfully.")

    return preprocessor


# ======================================================
# Full Feature Preparation
# ======================================================

def prepare_features(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list, list]:
    """
    Run the complete feature preparation workflow.

    Steps:
        1. Remove duplicate rows
        2. Separate features (X) and target (y)
        3. Detect numerical and categorical columns
        4. Drop rows where target is missing

    Args:
        df (pd.DataFrame): Raw input DataFrame.
        target_column (str): Name of the target column.

    Returns:
        tuple:
            - X (pd.DataFrame): Feature DataFrame
            - y (pd.Series): Target Series
            - numerical_cols (list): List of numerical column names
            - categorical_cols (list): List of categorical column names
    """
    # Step 1: Remove duplicates
    df = remove_duplicates(df)

    # Step 2: Drop rows with missing target values
    before = df.shape[0]
    df = df.dropna(subset=[target_column])
    dropped = before - df.shape[0]
    if dropped > 0:
        print(f"[Feature Engineering] Dropped {dropped} row(s) with missing target values.")

    # Step 3: Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Step 4: Detect column types
    numerical_cols, categorical_cols = detect_column_types(df, target_column)

    return X, y, numerical_cols, categorical_cols