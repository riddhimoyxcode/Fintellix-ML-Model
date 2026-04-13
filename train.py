"""
Fintellix Fraud Detection - Training Orchestrator
====================================================
Run this script to execute the full pipeline:

    python train.py                         # with existing CSV in data/
    python train.py --data path/to/file.csv # custom dataset path
    python train.py --generate              # generate synthetic dataset

Usage
-----
    $ python train.py
    $ python train.py --data data/creditcard.csv
    $ python train.py --generate --samples 284807
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd

from config import DATASET_PATH, get_logger
from data_preprocessing import preprocess_pipeline
from model_training import build_model, train_model, evaluate_model, save_model

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing when real dataset is unavailable)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(
    n_samples: int = 284_807,
    fraud_ratio: float = 0.00172,
    save_path: str = DATASET_PATH,
) -> str:
    """
    Generate a synthetic credit card fraud dataset that mirrors the schema
    of the real Kaggle dataset.  Useful for development and CI/CD testing.

    Parameters
    ----------
    n_samples   : total number of transactions
    fraud_ratio : fraction of fraudulent transactions
    save_path   : where to write the CSV

    Returns
    -------
    str – path to the generated CSV
    """
    logger.info(
        "Generating synthetic dataset: %d samples (fraud_ratio=%.4f)",
        n_samples, fraud_ratio,
    )
    rng = np.random.default_rng(42)

    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # V1–V28: PCA-like features (different distributions for fraud vs normal)
    normal_V = rng.normal(loc=0, scale=1, size=(n_normal, 28))
    fraud_V = rng.normal(loc=2, scale=2, size=(n_fraud, 28))  # shifted

    # Time: seconds elapsed (simulated 48h window)
    normal_time = rng.uniform(0, 172_800, size=(n_normal, 1))
    fraud_time = rng.uniform(0, 172_800, size=(n_fraud, 1))

    # Amount: normal transactions mostly small; fraudulent skew higher
    normal_amount = rng.exponential(scale=80, size=(n_normal, 1))
    fraud_amount = rng.exponential(scale=500, size=(n_fraud, 1))

    # Assemble
    normal_data = np.hstack([normal_time, normal_V, normal_amount])
    fraud_data = np.hstack([fraud_time, fraud_V, fraud_amount])

    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    df_normal = pd.DataFrame(normal_data, columns=columns)
    df_normal["Class"] = 0

    df_fraud = pd.DataFrame(fraud_data, columns=columns)
    df_fraud["Class"] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))  # add id column

    df.to_csv(save_path, index=False)
    logger.info("Synthetic dataset saved -> %s (%d rows)", save_path, len(df))
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fintellix Fraud Detection - Train")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate a synthetic dataset for testing",
    )
    parser.add_argument(
        "--samples", type=int, default=284_807,
        help="Number of samples in synthetic dataset (with --generate)",
    )
    args = parser.parse_args()

    start = time.time()
    print("\n[>>] Fintellix Fraud Detection - Training Pipeline\n")

    # --- Resolve dataset path ---
    dataset_path = args.data or DATASET_PATH
    if args.generate:
        dataset_path = generate_synthetic_dataset(
            n_samples=args.samples, save_path=dataset_path
        )

    # --- Preprocessing ---
    print("[1/4] Data Preprocessing")
    data = preprocess_pipeline(dataset_path)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    scaler = data["scaler"]
    feature_names = data["feature_names"]

    print(f"   Features: {len(feature_names)} | Train: {len(y_train)} | Test: {len(y_test)}")

    # --- Build model ---
    print("\n[2/4] Building XGBoost Model")
    # Compute scale_pos_weight from the ORIGINAL (pre-SMOTE) distribution
    # This gives extra weight to fraud class during training
    neg_count = int((y_test == 0).sum() / 0.2 * 0.8)  # approx original train negatives
    pos_count = max(int((y_test == 1).sum() / 0.2 * 0.8), 1)
    scale_pos_weight = neg_count / pos_count
    logger.info("Estimated scale_pos_weight = %.2f", scale_pos_weight)

    model = build_model(scale_pos_weight=scale_pos_weight)

    # --- Train ---
    print("\n[3/4] Training")
    model = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test)

    # --- Evaluate ---
    print("\n[4/4] Evaluation")
    metrics = evaluate_model(model, X_test, y_test)

    # --- Summary ---
    elapsed = time.time() - start
    print(f"\n[TIME] Total training time: {elapsed:.1f}s")

    # --- Fraud detection capability summary ---
    print("\n[RESULTS] Fraud Detection Summary:")
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    print(f"   True Positives  (correctly caught fraud): {tp}")
    print(f"   False Negatives (missed fraud):           {fn}")
    print(f"   False Positives (false alarms):           {fp}")
    print(f"   Fraud Recall:    {tp / (tp + fn):.2%}")
    print(f"   Fraud Precision: {tp / (tp + fp):.2%}")
    print(f"   ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"   Optimal Threshold: {metrics['optimal_threshold']:.4f}")

    # --- Save ---
    print("\n[SAVE] Saving model ...")
    save_model(model, scaler)
    print("   Model saved -> fraud_model.pkl")
    print("   Ready for API deployment!\n")

    return model, metrics


if __name__ == "__main__":
    main()
