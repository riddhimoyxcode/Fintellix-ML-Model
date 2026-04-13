"""
Fintellix Fraud Detection - Configuration
==========================================
Central configuration for all model parameters, file paths, and constants.
"""

import os
import logging

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

DATASET_PATH = os.path.join(DATA_DIR, "creditcard.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "amount_scaler.pkl")

# Ensure directories exist
for _d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Class"
DROP_COLUMNS = ["id"]                    # columns to drop before training
FEATURE_COLUMNS_PCA = [f"V{i}" for i in range(1, 29)]  # V1 … V28
EXTRA_FEATURES = ["Time", "Amount"]      # non-PCA features to keep
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS_PCA + EXTRA_FEATURES  # 30 features

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# SMOTE (class imbalance handling)
# ---------------------------------------------------------------------------
SMOTE_SAMPLING_STRATEGY = 0.5   # ratio of minority/majority after resampling
SMOTE_K_NEIGHBORS = 5
SMOTE_RANDOM_STATE = RANDOM_STATE

# ---------------------------------------------------------------------------
# XGBoost hyper-parameters
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",         # area under precision-recall curve
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# We also set scale_pos_weight dynamically during training

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
CLASSIFICATION_THRESHOLD = 0.5   # default; can be tuned for recall/precision

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE = os.path.join(LOG_DIR, "fraud_detection.log")


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console + file."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(LOG_LEVEL)
        ch.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(fh)

    return logger
