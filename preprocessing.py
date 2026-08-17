"""
Preprocessing module for the Loan Approval Prediction project.

This module serves as the main entry point for the full
preprocessing workflow. It coordinates data loading, feature
engineering, pipeline building, and saving the fitted
preprocessor to disk.
"""

import joblib
import pandas as pd
import numpy as np

from config import PREPROCESSOR_PATH
from src.data_loader import load_dataset, detect_target_column
from src.feature_engineering import (
    prepare_features,
    build_preprocessing_pipeline,
    encode_target,
)


# ======================================================
# Save Preprocessor
# ======================================================

def save_preprocessor(preprocessor_bundle: dict) -> None:
    """
    Save the fitted preprocessor bundle to disk using joblib.

    The bundle includes the fitted ColumnTransformer, the
    LabelEncoder for the target, and column metadata needed
    during inference.

    Args:
        preprocessor_bundle (dict): Dictionary containing preprocessing artifacts.
    """
    joblib.dump(preprocessor_bundle, PREPROCESSOR_PATH)
    print(f"[Preprocessing] Preprocessor bundle saved to: {PREPROCESSOR_PATH}")


# ======================================================
# Load Preprocessor
# ======================================================

def load_preprocessor() -> dict:
    """
    Load the saved preprocessor bundle from disk.

    Returns:
        dict: The preprocessor bundle containing:
              - "preprocessor": Fitted ColumnTransformer
              - "label_encoder": Fitted LabelEncoder for target
              - "target_column": Name of the target column
              - "numerical_cols": List of numerical feature columns
              - "categorical_cols": List of categorical feature columns
              - "feature_columns": Ordered list of all feature columns

    Raises:
        FileNotFoundError: If the preprocessor file does not exist.
    """
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at: {PREPROCESSOR_PATH}\n"
            "Please run 'python train.py' first to train the model."
        )

    bundle = joblib.load(PREPROCESSOR_PATH)
    print(f"[Preprocessing] Preprocessor bundle loaded from: {PREPROCESSOR_PATH}")
    return bundle


# ======================================================
# Run Full Preprocessing Pipeline
# ======================================================

def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """
    Execute the complete preprocessing pipeline.

    Steps:
        1. Load raw dataset from disk
        2. Detect target column automatically
        3. Prepare features (clean, separate X and y, detect column types)
        4. Encode the target column
        5. Build the preprocessing pipeline (imputation + scaling + encoding)
        6. Bundle and save the preprocessor artifacts

    Returns:
        tuple:
            - X_train (pd.DataFrame): Training features (raw, before transform)
            - X_test (pd.DataFrame): Testing features (raw, before transform)
            - y_train (np.ndarray): Encoded training target
            - y_test (np.ndarray): Encoded testing target
            - preprocessor_bundle (dict): All preprocessing artifacts
    """
    from src.model_training import split_data

    print("\n[Preprocessing] Starting preprocessing pipeline ...\n")

    # Step 1: Load raw dataset
    df = load_dataset()

    # Step 2: Detect target column
    target_column = detect_target_column(df)

    # Step 3: Prepare features
    X, y, numerical_cols, categorical_cols = prepare_features(df, target_column)

    # Step 4: Encode target labels
    y_encoded, label_encoder = encode_target(y)

    # Step 5: Split data before fitting preprocessor
    # (prevents data leakage from test set into preprocessing)
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    # Step 6: Build preprocessing pipeline and fit ONLY on training data
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    preprocessor.fit(X_train)

    print("[Preprocessing] Preprocessor fitted on training data only (no leakage).")

    # Step 7: Bundle all artifacts needed for inference
    preprocessor_bundle = {
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "target_column": target_column,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "feature_columns": numerical_cols + categorical_cols,
    }

    # Step 8: Save bundle to disk
    save_preprocessor(preprocessor_bundle)

    print("\n[Preprocessing] Preprocessing pipeline complete.\n")

    return X_train, X_test, y_train, y_test, preprocessor_bundle