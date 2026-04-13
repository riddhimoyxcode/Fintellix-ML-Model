"""
Fintellix Fraud Detection - FastAPI Service
=============================================
A production-ready REST API for real-time fraud detection.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /              -> health check
    GET  /health        -> detailed health status
    POST /predict       -> single transaction prediction
    POST /predict/batch -> batch prediction
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict_transaction, _ensure_model_loaded, set_threshold
from config import CLASSIFICATION_THRESHOLD, get_logger

logger = get_logger("api")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class TransactionInput(BaseModel):
    """Single transaction input - 30 numerical features."""
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Array of 30 features: [V1..V28, Time, Amount]",
        examples=[[0.0] * 30],
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom probability threshold (overrides default)",
    )


class BatchTransactionInput(BaseModel):
    """Batch of transactions."""
    transactions: list[list[float]] = Field(
        ...,
        min_length=1,
        description="List of feature arrays, each with 30 elements",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom probability threshold (overrides default)",
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    fraud_probability: float
    threshold_used: float
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: list[int]
    labels: list[str]
    fraud_probabilities: list[float]
    threshold_used: float
    total_transactions: int
    flagged_fraud: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    default_threshold: float
    version: str


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model on startup so the first request isn't slow."""
    logger.info("Starting Fintellix Fraud Detection API ...")
    try:
        _ensure_model_loaded()
        logger.info("Model pre-loaded successfully.")
    except FileNotFoundError as e:
        logger.error("Model not found: %s - run train.py first", e)
    yield
    logger.info("Shutting down ...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fintellix Fraud Detection API",
    description=(
        "Real-time credit card fraud detection powered by XGBoost. "
        "Submit transaction features and receive instant fraud probability scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - quick connectivity check."""
    return {
        "service": "Fintellix Fraud Detection API",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health / readiness probe."""
    from predict import _model
    return HealthResponse(
        status="healthy" if _model is not None else "model_not_loaded",
        model_loaded=_model is not None,
        default_threshold=CLASSIFICATION_THRESHOLD,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(txn: TransactionInput):
    """
    Predict fraud probability for a **single** transaction.

    Input: array of 30 features `[V1, V2, … V28, Time, Amount]`.
    """
    t0 = time.perf_counter()
    try:
        result = predict_transaction(txn.features, threshold=txn.threshold)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=str(e))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return PredictionResponse(
        prediction=result["prediction"],
        label=result["label"],
        fraud_probability=result["fraud_probability"],
        threshold_used=result["threshold_used"],
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
async def predict_batch(batch: BatchTransactionInput):
    """
    Predict fraud probability for a **batch** of transactions.

    Each inner list must contain exactly 30 features.
    """
    t0 = time.perf_counter()

    # Validate each row length
    for idx, row in enumerate(batch.transactions):
        if len(row) != 30:
            raise HTTPException(
                status_code=422,
                detail=f"Transaction at index {idx} has {len(row)} features, expected 30.",
            )

    try:
        arr = np.array(batch.transactions, dtype=np.float64)
        result = predict_transaction(arr, threshold=batch.threshold)
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=422, detail=str(e))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return BatchPredictionResponse(
        predictions=result["predictions"],
        labels=result["labels"],
        fraud_probabilities=result["fraud_probabilities"],
        threshold_used=result["threshold_used"],
        total_transactions=result["total_transactions"],
        flagged_fraud=result["flagged_fraud"],
        processing_time_ms=round(elapsed_ms, 2),
    )
