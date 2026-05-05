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

import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict_transaction, _ensure_model_loaded, set_threshold
from config import CLASSIFICATION_THRESHOLD, get_logger
from stock_predict import predict_stock
from market_data import get_live_quote, get_historical_prices

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


class StockPredictionInput(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    days: int = Field(default=7, ge=1, le=30, description="Number of days to forecast")


class StockPredictionResponse(BaseModel):
    symbol: str
    forecast: list[dict]
    processing_time_ms: float


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


@app.post("/predict/stock", response_model=StockPredictionResponse, tags=["Prediction"])
async def predict_stock_price(req: StockPredictionInput):
    """
    Dynamically trains an ML model on recent historical data and predicts future prices.
    """
    t0 = time.perf_counter()
    try:
        predictions = predict_stock(req.symbol, req.days)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.exception("Stock prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return StockPredictionResponse(
        symbol=req.symbol,
        forecast=predictions,
        processing_time_ms=round(elapsed_ms, 2),
    )

@app.get("/market/quote/{symbol}", tags=["Market Data"])
async def market_quote(symbol: str):
    """Fetch live quote for a symbol."""
    quote = get_live_quote(symbol)
    if not quote:
        raise HTTPException(status_code=404, detail="Quote not found or invalid symbol")
    return quote

@app.get("/market/history/{symbol}", tags=["Market Data"])
async def market_history(symbol: str):
    """Fetch 3-month historical data for a symbol."""
    history = get_historical_prices(symbol)
    if not history:
        raise HTTPException(status_code=404, detail="History not found or invalid symbol")
    return history


# ---------------------------------------------------------------------------
# Local development entry-point  (python api.py)
# In production on Render, gunicorn is used instead (see Procfile).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
