"""
Fintellix Fraud Detection - Model Training & Evaluation
=========================================================
Train an XGBoost classifier, evaluate it with fraud-oriented metrics,
and export the trained model to disk.
"""

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
)

from config import (
    XGB_PARAMS,
    MODEL_PATH,
    SCALER_PATH,
    CLASSIFICATION_THRESHOLD,
    get_logger,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Build model
# ---------------------------------------------------------------------------
def build_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """
    Instantiate an XGBClassifier with tuned hyper-parameters.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio of negative to positive samples.  Helps the model pay more
        attention to the minority (fraud) class.

    Returns
    -------
    XGBClassifier
    """
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    model = XGBClassifier(**params)
    logger.info("XGBClassifier built | scale_pos_weight=%.2f", scale_pos_weight)
    return model


# ---------------------------------------------------------------------------
# 2. Train model
# ---------------------------------------------------------------------------
def train_model(model: XGBClassifier, X_train, y_train, X_val=None, y_val=None):
    """
    Fit the model on (resampled) training data.

    Parameters
    ----------
    model      : XGBClassifier
    X_train    : array-like  – training features
    y_train    : array-like  – training labels
    X_val      : array-like  – optional validation features (for early stopping)
    y_val      : array-like  – optional validation labels

    Returns
    -------
    XGBClassifier  (fitted)
    """
    logger.info("Training started ...")
    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)
    logger.info("Training complete.")
    return model


# ---------------------------------------------------------------------------
# 3. Evaluate model
# ---------------------------------------------------------------------------
def evaluate_model(
    model: XGBClassifier,
    X_test,
    y_test,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> dict:
    """
    Evaluate the trained model and print a detailed report.

    Parameters
    ----------
    model     : trained XGBClassifier
    X_test    : test features
    y_test    : true labels
    threshold : probability threshold for positive class

    Returns
    -------
    dict  – evaluation metrics
    """
    logger.info("Evaluating model ...")

    # Probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]

    # Apply custom threshold
    y_pred = (y_proba >= threshold).astype(int)

    # ---- Metrics ----
    report = classification_report(
        y_test, y_pred, target_names=["Normal", "Fraud"], output_dict=True
    )
    report_str = classification_report(
        y_test, y_pred, target_names=["Normal", "Fraud"]
    )
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    f1_fraud = f1_score(y_test, y_pred, pos_label=1)

    # ---- Optimal threshold (maximise F1 for fraud) ----
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else threshold

    # ---- Pretty print ----
    print("\n" + "=" * 60)
    print("           FRAUD DETECTION - EVALUATION REPORT")
    print("=" * 60)
    print(report_str)
    print(f"Confusion Matrix:\n{cm}\n")
    print(f"  ROC-AUC Score ........... {roc_auc:.4f}")
    print(f"  Average Precision (PR) .. {avg_precision:.4f}")
    print(f"  F1-Score (Fraud) ........ {f1_fraud:.4f}")
    print(f"  Optimal Threshold ....... {best_threshold:.4f}")
    print("=" * 60)

    # ---- Log summary ----
    tn, fp, fn, tp = cm.ravel()
    logger.info(
        "Confusion: TP=%d  FP=%d  FN=%d  TN=%d | "
        "ROC-AUC=%.4f | AP=%.4f | F1-Fraud=%.4f | Best-Thr=%.4f",
        tp, fp, fn, tn, roc_auc, avg_precision, f1_fraud, best_threshold,
    )

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "f1_fraud": f1_fraud,
        "optimal_threshold": float(best_threshold),
    }


# ---------------------------------------------------------------------------
# 4. Save model
# ---------------------------------------------------------------------------
def save_model(model, scaler=None):
    """
    Persist the trained model (and optional scaler) to disk using joblib.

    Parameters
    ----------
    model  : trained XGBClassifier
    scaler : fitted StandardScaler (or None)
    """
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved -> %s", MODEL_PATH)

    if scaler is not None:
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Scaler saved -> %s", SCALER_PATH)


# ---------------------------------------------------------------------------
# 5. Load model
# ---------------------------------------------------------------------------
def load_model():
    """
    Load a previously saved model (and scaler if available).

    Returns
    -------
    model  : XGBClassifier
    scaler : StandardScaler or None
    """
    import os
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded <- %s", MODEL_PATH)

    scaler = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded <- %s", SCALER_PATH)

    return model, scaler
