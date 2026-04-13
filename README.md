# Fintellix Fraud Detection System

A high-performance, production-ready fraud detection model for credit card transactions, powered by **XGBoost** and served via **FastAPI**.

## 🏗 Project Structure

```
Fintellix-ML-Model/
├── config.py               # Central configuration (paths, hyperparams, logging)
├── data_preprocessing.py   # Data loading, cleaning, SMOTE, train/test split
├── model_training.py       # XGBoost training, evaluation, model export
├── predict.py              # Prediction function (single & batch)
├── train.py                # Main training orchestrator
├── api.py                  # FastAPI REST service
├── requirements.txt        # Python dependencies
├── data/                   # Dataset directory (CSV files)
├── models/                 # Saved model artifacts (.pkl)
└── logs/                   # Training & API logs
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model

**With the real Kaggle dataset:**
Place `creditcard.csv` in the `data/` directory, then:
```bash
python train.py
```

**With a custom dataset path:**
```bash
python train.py --data /path/to/creditcard.csv
```

**With synthetic data (for testing):**
```bash
python train.py --generate
```

### 3. Launch the API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## 📡 API Endpoints

| Method | Endpoint         | Description                  |
|--------|------------------|------------------------------|
| GET    | `/`              | Service info                 |
| GET    | `/health`        | Health / readiness check     |
| GET    | `/docs`          | Interactive Swagger UI       |
| POST   | `/predict`       | Single transaction prediction|
| POST   | `/predict/batch` | Batch prediction             |

### Example — Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100.0]}'
```

### Example — Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100.0]], "threshold": 0.3}'
```

## 🧠 Model Details

- **Algorithm:** XGBoost (Gradient Boosted Trees)
- **Imbalance Handling:** SMOTE oversampling + `scale_pos_weight`
- **Features:** 30 (V1–V28 PCA features + Time + Amount)
- **Metrics Focus:** Recall for fraud detection class

## 📊 Key Evaluation Metrics

- **ROC-AUC** — overall discrimination ability
- **Average Precision (PR-AUC)** — performance on imbalanced data
- **F1-Score (Fraud class)** — harmonic mean of precision and recall
- **Confusion Matrix** — TP, FP, FN, TN breakdown
- **Optimal Threshold** — threshold that maximizes fraud F1-score

## ⚙️ Configuration

All parameters are centralized in `config.py`:
- File paths
- Train/test split ratio
- SMOTE parameters
- XGBoost hyperparameters
- Logging configuration
