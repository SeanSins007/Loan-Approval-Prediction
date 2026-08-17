"""
Model evaluation module for the Loan Approval Prediction project.

This module handles computing evaluation metrics, generating
comparison reports, saving results to disk, and producing
a confusion matrix visualization.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import (
    MODEL_COMPARISON_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    METRICS_JSON_PATH,
)

# Use non-interactive backend to avoid display issues on servers
matplotlib.use("Agg")


# ======================================================
# Compute Metrics for a Single Model
# ======================================================

def compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict:
    """
    Compute classification metrics for a single model.

    Args:
        y_test (np.ndarray): True target labels.
        y_pred (np.ndarray): Predicted labels.
        model_name (str): Name of the model being evaluated.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }

    return metrics


# ======================================================
# Build Model Comparison DataFrame
# ======================================================

def build_comparison_table(
    trained_models: dict,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Build a comparison table of all trained models using their metrics.

    Args:
        trained_models (dict): Dictionary of trained model results.
        y_test (np.ndarray): True target labels for the test set.

    Returns:
        pd.DataFrame: A DataFrame with one row per model and metric columns.
    """
    all_metrics = []

    for model_name, model_data in trained_models.items():
        y_pred = model_data["y_pred"]
        metrics = compute_metrics(y_test, y_pred, model_name)
        all_metrics.append(metrics)
        print(f"[Evaluation] {model_name} -> "
              f"Accuracy: {metrics['Accuracy']:.4f} | "
              f"Precision: {metrics['Precision']:.4f} | "
              f"Recall: {metrics['Recall']:.4f} | "
              f"F1 Score: {metrics['F1 Score']:.4f}")

    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)

    return comparison_df


# ======================================================
# Save Model Comparison CSV
# ======================================================

def save_comparison_csv(comparison_df: pd.DataFrame) -> None:
    """
    Save the model comparison DataFrame to a CSV file.

    Args:
        comparison_df (pd.DataFrame): The model comparison table.
    """
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"[Evaluation] Model comparison saved to: {MODEL_COMPARISON_PATH}")


# ======================================================
# Save Classification Report CSV
# ======================================================

def save_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    best_model_name: str,
    target_classes: list,
) -> None:
    """
    Generate and save the classification report for the best model.

    Args:
        y_test (np.ndarray): True target labels.
        y_pred (np.ndarray): Predicted labels from the best model.
        best_model_name (str): Name of the best model.
        target_classes (list): List of human-readable class labels.
    """
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=[str(c) for c in target_classes],
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df.rename(columns={"index": "Class"}, inplace=True)
    report_df.insert(0, "Model", best_model_name)

    report_df.to_csv(CLASSIFICATION_REPORT_PATH, index=False)
    print(f"[Evaluation] Classification report saved to: {CLASSIFICATION_REPORT_PATH}")


# ======================================================
# Save Confusion Matrix PNG
# ======================================================

def save_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    best_model_name: str,
    target_classes: list,
) -> None:
    """
    Generate and save a confusion matrix as a PNG image.

    Args:
        y_test (np.ndarray): True target labels.
        y_pred (np.ndarray): Predicted labels from the best model.
        best_model_name (str): Name of the best model.
        target_classes (list): List of human-readable class labels.
    """
    cm = confusion_matrix(y_test, y_pred)
    class_labels = [str(c) for c in target_classes]

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_labels,
        yticklabels=class_labels,
        title=f"Confusion Matrix — {best_model_name}",
        ylabel="Actual Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)

    print(f"[Evaluation] Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


# ======================================================
# Save Metrics JSON
# ======================================================

def save_metrics_json(
    comparison_df: pd.DataFrame,
    best_model_name: str,
) -> None:
    """
    Save the best model metrics and full comparison table to a JSON file.

    Args:
        comparison_df (pd.DataFrame): Full model comparison table.
        best_model_name (str): Name of the best model.
    """
    best_row = comparison_df[comparison_df["Model"] == best_model_name].iloc[0]

    metrics_data = {
        "best_model": best_model_name,
        "best_model_metrics": {
            "accuracy": best_row["Accuracy"],
            "precision": best_row["Precision"],
            "recall": best_row["Recall"],
            "f1_score": best_row["F1 Score"],
        },
        "all_models": comparison_df.to_dict(orient="records"),
    }

    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"[Evaluation] Metrics JSON saved to: {METRICS_JSON_PATH}")


# ======================================================
# Run Full Evaluation Pipeline
# ======================================================

def run_evaluation(
    trained_models: dict,
    y_test: np.ndarray,
    best_model_name: str,
    target_classes: list,
) -> pd.DataFrame:
    """
    Run the complete evaluation pipeline for all trained models.

    Steps:
        1. Build model comparison table
        2. Save comparison CSV
        3. Save classification report for best model
        4. Save confusion matrix PNG for best model
        5. Save metrics JSON

    Args:
        trained_models (dict): Dictionary of trained model results.
        y_test (np.ndarray): True target labels.
        best_model_name (str): Name of the best performing model.
        target_classes (list): Human-readable class labels from the encoder.

    Returns:
        pd.DataFrame: The model comparison DataFrame.
    """
    print("\n[Evaluation] Running full evaluation pipeline ...")

    # Step 1: Build comparison table
    comparison_df = build_comparison_table(trained_models, y_test)

    # Step 2: Save comparison CSV
    save_comparison_csv(comparison_df)

    # Step 3: Save classification report for best model
    best_y_pred = trained_models[best_model_name]["y_pred"]
    save_classification_report(y_test, best_y_pred, best_model_name, target_classes)

    # Step 4: Save confusion matrix for best model
    save_confusion_matrix(y_test, best_y_pred, best_model_name, target_classes)

    # Step 5: Save metrics JSON
    save_metrics_json(comparison_df, best_model_name)

    print("[Evaluation] Evaluation pipeline complete.\n")

    return comparison_df