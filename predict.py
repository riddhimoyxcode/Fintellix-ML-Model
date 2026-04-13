"""
Fintellix Fraud Detection - Prediction Module
===============================================
Provides the `predict_transaction` function that accepts raw feature arrays
and returns fraud probability + classification.  Designed for direct use
and as the backend for the FastAPI service.
"""

import numpy as np
import joblib
import os

from config import MODEL_PATH, SCALER_PATH, CLASSIFICATION_THRESHOLD, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded model singleton
# ---------------------------------------------------------------------------
_model = None
_scaler = None
_optimal_threshold = CLASSIFICATION_THRESHOLD


def _ensure_model_loaded():
    """Load model and scaler on first call (lazy init)."""
    global _model, _scaler
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run train.py first to train and save the model."
            )
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)

        if os.path.exists(SCALER_PATH):
            _scaler = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded from %s", SCALER_PATH)


def set_threshold(threshold: float):
    """Override the global classification threshold at runtime."""
    global _optimal_threshold
    _optimal_threshold = threshold
    logger.info("Classification threshold set to %.4f", threshold)


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------
def predict_transaction(
    input_data,
    threshold: float | None = None,
) -> dict:
    """
    Predict whether a transaction is fraudulent.

    Parameters
    ----------
    input_data : list | np.ndarray
        A single transaction as a 1-D array of 30 numerical features:
        [V1, V2, … V28, Time, Amount]
        OR a 2-D array/list of shape (n_samples, 30).

    threshold : float, optional
        Custom probability threshold for this call.  Defaults to the
        globally configured value.

    Returns
    -------
    dict
        {
            "prediction": int,            # 0 = Normal, 1 = Fraud
            "label": str,                 # "Normal" or "Fraud"
            "fraud_probability": float,   # probability of fraud
            "threshold_used": float,
        }
        For batch inputs, lists are returned for prediction / label / probability.
    """
    _ensure_model_loaded()

    thr = threshold if threshold is not None else _optimal_threshold

    # --- Coerce to numpy 2-D array ----------------------------------------
    arr = np.asarray(input_data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        single = True
    else:
        single = False

    # --- Validate shape ----------------------------------------------------
    expected_features = _model.n_features_in_
    if arr.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, got {arr.shape[1]}. "
            "Input must contain [V1..V28, Time, Amount]."
        )

    # --- Handle missing values safely --------------------------------------
    if np.isnan(arr).any():
        logger.warning("NaN detected in input - replacing with 0.0")
        arr = np.nan_to_num(arr, nan=0.0)

    # --- Predict -----------------------------------------------------------
    probas = _model.predict_proba(arr)[:, 1]
    preds = (probas >= thr).astype(int)
    labels = np.where(preds == 1, "Fraud", "Normal")

    if single:
        result = {
            "prediction": int(preds[0]),
            "label": str(labels[0]),
            "fraud_probability": round(float(probas[0]), 6),
            "threshold_used": thr,
        }
        logger.info(
            "Prediction: %s (prob=%.4f, thr=%.4f)",
            result["label"], result["fraud_probability"], thr,
        )
    else:
        result = {
            "predictions": preds.tolist(),
            "labels": labels.tolist(),
            "fraud_probabilities": [round(float(p), 6) for p in probas],
            "threshold_used": thr,
            "total_transactions": len(preds),
            "flagged_fraud": int(preds.sum()),
        }
        logger.info(
            "Batch prediction: %d transactions, %d flagged as fraud",
            result["total_transactions"], result["flagged_fraud"],
        )

    return result
