"""
Fintellix Fraud Detection - Data Preprocessing
================================================
Load, clean, resample, and split the credit card fraud dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from config import (
    DATASET_PATH,
    TARGET_COLUMN,
    DROP_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    SMOTE_SAMPLING_STRATEGY,
    SMOTE_K_NEIGHBORS,
    SMOTE_RANDOM_STATE,
    get_logger,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the credit card transaction CSV.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all columns.
    """
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Class distribution:\n%s", df[TARGET_COLUMN].value_counts())
    return df


# ---------------------------------------------------------------------------
# 2. Clean data
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unnecessary columns and handle missing / duplicate values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    logger.info("Cleaning data ...")

    # Drop columns that shouldn't be used for training (e.g., 'id')
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped columns: %s", cols_to_drop)

    # Handle missing values - fill numerics with median (safe default)
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning("Found %d missing values - filling with column median", missing_count)
        df = df.fillna(df.median(numeric_only=True))
    else:
        logger.info("No missing values detected")

    # Remove exact duplicates (keep first)
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logger.info("Removing %d duplicate rows", dup_count)
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

    logger.info("Cleaned dataset shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 3. Separate features / target
# ---------------------------------------------------------------------------
def split_features_target(df: pd.DataFrame):
    """
    Separate the feature matrix X and target vector y.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    logger.info("Features shape: %s | Target shape: %s", X.shape, y.shape)
    return X, y


# ---------------------------------------------------------------------------
# 4. Scale the Amount / Time columns
# ---------------------------------------------------------------------------
def scale_features(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler | None]:
    """
    Apply StandardScaler to 'Amount' and 'Time' columns (V1-V28 are already
    PCA-transformed and scaled).

    Returns
    -------
    X_scaled : pd.DataFrame
    scaler   : StandardScaler or None (if columns not present)
    """
    cols_to_scale = [c for c in ("Amount", "Time") if c in X.columns]
    if not cols_to_scale:
        logger.info("No columns to scale")
        return X, None

    scaler = StandardScaler()
    X = X.copy()
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    logger.info("Scaled columns: %s", cols_to_scale)
    return X, scaler


# ---------------------------------------------------------------------------
# 5. Train / test split
# ---------------------------------------------------------------------------
def split_train_test(X, y):
    """
    Stratified 80/20 train-test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Train: %d samples (fraud=%.2f%%) | Test: %d samples (fraud=%.2f%%)",
        len(y_train), y_train.mean() * 100,
        len(y_test), y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 6. SMOTE oversampling (training set only)
# ---------------------------------------------------------------------------
def apply_smote(X_train, y_train):
    """
    Apply SMOTE to oversample the minority class on the **training set only**.

    Returns
    -------
    X_resampled : np.ndarray
    y_resampled : np.ndarray
    """
    logger.info("Applying SMOTE (strategy=%.2f) ...", SMOTE_SAMPLING_STRATEGY)
    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=SMOTE_RANDOM_STATE,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(
        "After SMOTE - Total: %d | Fraud: %d (%.2f%%)",
        len(y_res),
        y_res.sum() if hasattr(y_res, "sum") else np.sum(y_res),
        (y_res.sum() if hasattr(y_res, "sum") else np.sum(y_res)) / len(y_res) * 100,
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# 7. Full preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_pipeline(dataset_path: str = DATASET_PATH):
    """
    Execute the full preprocessing pipeline end-to-end.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test   – after SMOTE
        scaler                              – fitted StandardScaler (or None)
        feature_names                       – list of feature column names
    """
    df = load_dataset(dataset_path)
    df = clean_data(df)
    X, y = split_features_target(df)
    X, scaler = scale_features(X)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    return {
        "X_train": X_train_res,
        "X_test": X_test,
        "y_train": y_train_res,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
    }
