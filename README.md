# Fintellix Intelligence Service — Machine Learning API

A specialized high-performance microservice that powers the AI-driven features of the Fintellix ecosystem. Built with **FastAPI** and **Python**, it provides real-time fraud detection and predictive stock market analysis.

## 🏗 Key Capabilities

### 🛡️ Fraud Detection (XGBoost)
- **Model**: Gradient Boosted Trees (XGBoost) trained on 280,000+ transactions.
- **Accuracy**: Optimized for high recall to identify fraudulent patterns in real-time.
- **Features**: Analyzes 30 transaction features including Time, Amount, and V1-V28 PCA components.

### 📈 Stock Market Intelligence
- **AI Price Forecast**: Dynamically trains models on recent historical data to predict future price movements (7-day window).
- **Market Data Proxy**: Acts as a resilient data bridge using `yfinance` to bypass datacenter IP restrictions, ensuring reliable data delivery for the Fintellix Client.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Fraud Model
```bash
python train.py --generate # Or use your own creditcard.csv in data/
```

### 3. Launch the API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single transaction fraud prediction |
| POST | `/predict/stock` | AI-based stock price forecast (7 days) |
| GET | `/market/quote/{symbol}` | Resilient live stock quote (yfinance proxy) |
| GET | `/market/history/{symbol}` | 3-month historical data (yfinance proxy) |
| GET | `/health` | Service health and model status check |

## 🛠 Tech Stack
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/)
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [Scikit-learn](https://scikit-learn.org/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Finance Data**: [yfinance](https://github.com/ranaroussi/yfinance)

## ☁️ Deployment
This service is designed to be deployed on platforms like **Render** or **AWS App Runner**.
- **Procfile**: Included for Gunicorn deployment.
- **Port**: Defaults to `8000` or the `$PORT` environment variable.
