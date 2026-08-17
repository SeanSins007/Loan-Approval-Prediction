"""
Training script for the Loan Approval Prediction project.

This script serves as the main entry point for the complete
training pipeline. Run this script to preprocess the data,
train all models, evaluate them, select the best model,
and save all artifacts to disk.

Usage:
    python train.py
"""

import joblib

from config import BEST_MODEL_PATH
from preprocessing import run_preprocessing
from src.model_training import train_all_models, select_best_model
from src.evaluation import run_evaluation


# ======================================================
# Save Best Model
# ======================================================

def save_best_model(best_pipeline) -> None:
    """
    Save the best trained model pipeline to disk using joblib.

    Args:
        best_pipeline: The best trained Scikit-Learn Pipeline object.
    """
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"[Train] Best model saved to: {BEST_MODEL_PATH}")


# ======================================================
# Main Training Pipeline
# ======================================================

def main() -> None:
    """
    Execute the full training pipeline in the correct order.

    Steps:
        1. Run preprocessing (load data, clean, encode, split, fit preprocessor)
        2. Train all registered classification models
        3. Evaluate all models and generate result files
        4. Select the best model based on F1 Score
        5. Save the best model pipeline to disk
    """
    print("=" * 60)
    print("   Loan Approval Prediction — Training Pipeline")
    print("=" * 60)

    # Step 1: Preprocessing
    X_train, X_test, y_train, y_test, preprocessor_bundle = run_preprocessing()

    preprocessor = preprocessor_bundle["preprocessor"]
    label_encoder = preprocessor_bundle["label_encoder"]
    target_classes = label_encoder.classes_

    # Step 2: Train all models
    print("\n[Train] Starting model training ...\n")
    trained_models = train_all_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
    )

    # Step 3: Evaluate all models and generate result files
    # We need the comparison table first to determine the best model
    from src.evaluation import build_comparison_table
    comparison_df = build_comparison_table(trained_models, y_test)

    # Step 4: Select the best model based on F1 Score
    best_model_name, best_pipeline = select_best_model(trained_models, comparison_df)

    # Step 5: Run full evaluation and save all result files
    run_evaluation(
        trained_models=trained_models,
        y_test=y_test,
        best_model_name=best_model_name,
        target_classes=target_classes,
    )

    # Step 6: Save the best model pipeline
    save_best_model(best_pipeline)

    # Final summary
    print("\n" + "=" * 60)
    print("   Training Complete!")
    print("=" * 60)
    print(f"  Best Model     : {best_model_name}")
    print(f"  Model saved to : {BEST_MODEL_PATH}")
    print(f"  Preprocessor   : {preprocessor_bundle['target_column']} (target)")
    print(f"  Features used  : {preprocessor_bundle['feature_columns']}")
    print("=" * 60)
    print("\n  Results saved to the 'results/' folder:")
    print("    - model_comparison.csv")
    print("    - classification_report.csv")
    print("    - confusion_matrix.png")
    print("    - metrics.json")
    print("\n  Run 'python app.py' to start the Flask application.")
    print("=" * 60 + "\n")


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    main()