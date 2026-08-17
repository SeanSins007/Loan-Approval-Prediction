"""
Model training module for the Loan Approval Prediction project.

This module handles defining classifiers, training them on the
preprocessed data, and selecting the best performing model
based on evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from config import TEST_SIZE, RANDOM_STATE


# ======================================================
# Model Definitions
# ======================================================

def get_models() -> dict:
    """
    Return a dictionary of classification models to train and compare.

    Returns:
        dict: A dictionary mapping model names to instantiated classifiers.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    print(f"[Model Training] Models registered: {list(models.keys())}")
    return models


# ======================================================
# Train/Test Split
# ======================================================

def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and testing sets.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (np.ndarray): Encoded target array.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"[Model Training] Training samples : {X_train.shape[0]}")
    print(f"[Model Training] Testing samples  : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


# ======================================================
# Build Full Pipeline
# ======================================================

def build_model_pipeline(
    preprocessor: ColumnTransformer,
    classifier,
) -> Pipeline:
    """
    Combine the preprocessing pipeline and a classifier into a
    single Scikit-Learn Pipeline.

    Args:
        preprocessor (ColumnTransformer): The fitted or unfitted preprocessor.
        classifier: A Scikit-Learn compatible classifier instance.

    Returns:
        Pipeline: A complete pipeline with preprocessing and classification steps.
    """
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return pipeline


# ======================================================
# Train All Models
# ======================================================

def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    preprocessor: ColumnTransformer,
) -> dict:
    """
    Train all registered models using the preprocessing pipeline
    and return their trained pipelines along with predictions.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (np.ndarray): Training target.
        y_test (np.ndarray): Testing target.
        preprocessor (ColumnTransformer): The preprocessing pipeline.

    Returns:
        dict: A dictionary mapping model names to a dict containing:
              - "pipeline": Trained sklearn Pipeline
              - "y_pred": Predictions on X_test
              - "y_prob": Prediction probabilities on X_test (if available)
    """
    models = get_models()
    trained_models = {}

    for name, classifier in models.items():
        print(f"\n[Model Training] Training: {name} ...")

        # Each model gets its own fresh copy of the preprocessor
        # to avoid fitting state leakage between models
        from sklearn.base import clone
        fresh_preprocessor = clone(preprocessor)

        pipeline = build_model_pipeline(fresh_preprocessor, classifier)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        # Get probabilities if the classifier supports it
        y_prob = None
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)

        trained_models[name] = {
            "pipeline": pipeline,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        print(f"[Model Training] {name} trained successfully.")

    return trained_models


# ======================================================
# Select Best Model
# ======================================================

def select_best_model(
    trained_models: dict,
    comparison_df: pd.DataFrame,
) -> tuple[str, Pipeline]:
    """
    Select the best performing model based on F1 Score.

    F1 Score is chosen as the primary metric because it balances
    precision and recall, which is important for imbalanced
    loan approval datasets.

    Args:
        trained_models (dict): Dictionary of trained model pipelines.
        comparison_df (pd.DataFrame): DataFrame containing model metrics.

    Returns:
        tuple[str, Pipeline]: The best model name and its trained pipeline.
    """
    best_model_name = comparison_df.loc[
        comparison_df["F1 Score"].idxmax(), "Model"
    ]

    best_pipeline = trained_models[best_model_name]["pipeline"]

    print(f"\n[Model Training] Best model selected: '{best_model_name}' "
          f"(F1 Score: {comparison_df.loc[comparison_df['Model'] == best_model_name, 'F1 Score'].values[0]:.4f})")

    return best_model_name, best_pipeline