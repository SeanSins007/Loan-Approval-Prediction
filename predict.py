"""
Prediction module for the Loan Approval Prediction project.

This module handles loading the trained model and preprocessor,
accepting raw user input, transforming it through the preprocessing
pipeline, and returning a prediction with confidence score.

This module is used by the Flask application (app.py) to serve
predictions without reloading artifacts on every request.
"""

import joblib
import numpy as np
import pandas as pd

from config import BEST_MODEL_PATH
from preprocessing import load_preprocessor


# ======================================================
# Load Artifacts
# ======================================================

def load_model():
    """
    Load the best trained model pipeline from disk.

    Returns:
        sklearn.pipeline.Pipeline: The trained model pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {BEST_MODEL_PATH}\n"
            "Please run 'python train.py' first to train the model."
        )

    model = joblib.load(BEST_MODEL_PATH)
    print(f"[Predict] Model loaded from: {BEST_MODEL_PATH}")
    return model


def load_artifacts() -> tuple:
    """
    Load both the trained model and preprocessor bundle from disk.

    Returns:
        tuple:
            - model: Trained Scikit-Learn Pipeline
            - preprocessor_bundle (dict): Preprocessing artifacts

    Raises:
        FileNotFoundError: If either artifact is missing.
    """
    model = load_model()
    preprocessor_bundle = load_preprocessor()
    return model, preprocessor_bundle


# ======================================================
# Input Preparation
# ======================================================

def prepare_input(
    user_input: dict,
    preprocessor_bundle: dict,
) -> pd.DataFrame:
    """
    Convert raw user input dictionary into a properly ordered
    and typed DataFrame ready for model prediction.
    """
    numerical_cols = preprocessor_bundle["numerical_cols"]
    categorical_cols = preprocessor_bundle["categorical_cols"]
    feature_columns = preprocessor_bundle["feature_columns"]

    # Check for missing keys in user input
    missing_keys = [col for col in feature_columns if col not in user_input]
    if missing_keys:
        raise ValueError(
            f"Missing input fields: {missing_keys}\n"
            "Please provide values for all required features."
        )

    # Build a single-row DataFrame in the correct column order
    input_df = pd.DataFrame([user_input])

    # Cast numerical columns to float
    for col in numerical_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    # Cast categorical columns to string
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str).str.strip()

    # Reorder columns to match training order
    input_df = input_df[feature_columns]

    return input_df


# ======================================================
# Prediction
# ======================================================

def make_prediction(
    user_input: dict,
    model,
    preprocessor_bundle: dict,
) -> dict:
    """
    Generate a loan approval prediction from raw user input.
    """
    label_encoder = preprocessor_bundle["label_encoder"]

    # Step 1: Prepare input DataFrame
    input_df = prepare_input(user_input, preprocessor_bundle)

    # Step 2: Predict encoded label
    raw_label = model.predict(input_df)[0]

    # Step 3: Decode label to human-readable string
    prediction_label = label_encoder.inverse_transform([raw_label])[0]

    # Step 4: Get prediction confidence if model supports probabilities
    confidence = None
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = round(float(np.max(probabilities)) * 100, 2)

    # Step 5: Determine if the loan is approved
    approved = _is_approved(str(prediction_label))

    print(f"[Predict] Prediction: {prediction_label} | "
          f"Confidence: {confidence}% | "
          f"Approved: {approved}")

    return {
        "prediction": str(prediction_label),
        "confidence": confidence,
        "approved": approved,
        "raw_label": int(raw_label),
    }


# ======================================================
# Approval Detection Helper
# ======================================================

def _is_approved(label: str) -> bool:
    """
    Determine if a prediction label represents loan approval.
    """
    approval_indicators = [
        "y", "yes", "approved", "1", "true", "accept", "accepted", "grant", "granted"
    ]

    return label.strip().lower() in approval_indicators


# ======================================================
# Prediction Result Formatter
# ======================================================

def format_prediction_result(prediction_result: dict) -> dict:
    """
    Format the raw prediction result into a display-ready dictionary
    for the Flask template.
    """
    approved = prediction_result["approved"]
    confidence = prediction_result["confidence"]

    return {
        "prediction": prediction_result["prediction"],
        "approved": approved,
        "confidence": f"{confidence:.2f}%" if confidence is not None else "N/A",
        "result_label": "Approved" if approved else "Rejected",
        "result_class": "success" if approved else "danger",
        "result_icon": "✅" if approved else "❌",
        "message": (
            "Congratulations! Your loan application has been approved."
            if approved else
            "Sorry, your loan application has been rejected."
        ),
    }