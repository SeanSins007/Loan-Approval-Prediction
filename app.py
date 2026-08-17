"""
Flask application for the Loan Approval Prediction project.

This module creates and runs the Flask web server that serves
the prediction form, handles user input, validates it, runs
the model prediction pipeline, and displays the result.

Usage:
    python app.py
"""

from flask import Flask, render_template, request

from config import (
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    TEMPLATES_DIR,
    STATIC_DIR,
)
from predict import load_artifacts, make_prediction, format_prediction_result
from utils import (
    extract_form_data,
    get_form_field_metadata,
    load_dataset_sample,
    load_metrics,
    validate_user_input,
)
from src.data_loader import detect_target_column, get_feature_columns


# ======================================================
# Flask App Factory
# ======================================================

def create_app() -> Flask:
    """
    Create and configure the Flask application instance.

    Returns:
        Flask: Configured Flask application.
    """
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )

    # Load model and preprocessor once at startup
    model, preprocessor_bundle = load_artifacts()

    # Load dataset sample for dynamic form generation
    df_sample = load_dataset_sample()
    target_column = detect_target_column(df_sample)
    feature_columns = get_feature_columns(df_sample, target_column)

    numerical_cols = preprocessor_bundle["numerical_cols"]
    categorical_cols = preprocessor_bundle["categorical_cols"]

    # Generate form field metadata from dataset
    field_metadata = get_form_field_metadata(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        df_sample=df_sample,
    )

    # Load model metrics for display
    metrics = load_metrics()

    # --------------------------------------------------
    # Routes
    # --------------------------------------------------

    @app.route("/", methods=["GET", "POST"])
    def index():
        """
        Main route that renders the prediction form and handles
        form submissions for loan approval prediction.
        """
        prediction_result = None
        errors = []
        user_input = {}

        if request.method == "POST":
            # Step 1: Extract form data
            user_input = extract_form_data(
                form_data=request.form,
                feature_columns=feature_columns,
            )

            # Step 2: Validate user input
            is_valid, validation_errors = validate_user_input(
                user_input=user_input,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
            )

            if not is_valid:
                errors = validation_errors
            else:
                try:
                    # Step 3: Run prediction
                    raw_result = make_prediction(
                        user_input=user_input,
                        model=model,
                        preprocessor_bundle=preprocessor_bundle,
                    )

                    # Step 4: Format result for display
                    prediction_result = format_prediction_result(raw_result)

                except Exception as e:
                    errors = [f"Prediction failed: {str(e)}"]
                    print(f"[App] Prediction error: {e}")

        return render_template(
            "index.html",
            field_metadata=field_metadata,
            feature_columns=feature_columns,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            prediction_result=prediction_result,
            errors=errors,
            user_input=user_input,
            metrics=metrics,
        )

    @app.route("/health", methods=["GET"])
    def health():
        """
        Health check endpoint to verify the application is running.
        """
        return {"status": "ok", "message": "Loan Approval Prediction API is running."}

    return app


# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    app = create_app()

    print("\n" + "=" * 60)
    print("   Loan Approval Prediction — Flask Application")
    print("=" * 60)
    print(f"   Running at: http://localhost:{FLASK_PORT}")
    print(f"   Debug mode: {FLASK_DEBUG}")
    print("=" * 60 + "\n")

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )