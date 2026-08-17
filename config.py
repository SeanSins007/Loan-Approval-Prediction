"""
Configuration settings for the Loan Approval Prediction project.

This module centralizes all file paths and configurable constants
used across training and Flask application modules.
"""

from pathlib import Path


# ======================================================
# Base Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

SRC_DIR = BASE_DIR / "src"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# ======================================================
# Dataset Configuration
# ======================================================

DATA_FILE = DATA_DIR / "loan_data.csv"


# ======================================================
# Model & Preprocessing Artifacts
# ======================================================

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"


# ======================================================
# Results Files
# ======================================================

MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
METRICS_JSON_PATH = RESULTS_DIR / "metrics.json"


# ======================================================
# Training Configuration
# ======================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ======================================================
# Flask Configuration
# ======================================================

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True


# ======================================================
# Ensure Required Directories Exist
# ======================================================

def create_required_directories() -> None:
    """
    Create required project directories if they do not exist.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Automatically ensure required directories exist
create_required_directories()