import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import logging

logger = logging.getLogger("api")

def create_features(df, n_lags=5):
    """Create lagged features for time series prediction."""
    for i in range(1, n_lags + 1):
        df[f'close_lag_{i}'] = df['Close'].shift(i)
    # Moving averages based on past closing prices (shifted by 1 to prevent target leak)
    df['ma_7'] = df['Close'].shift(1).rolling(window=7).mean()
    df['ma_14'] = df['Close'].shift(1).rolling(window=14).mean()
    # Volatility based on past closing prices (shifted by 1 to prevent target leak)
    df['volatility_7'] = df['Close'].shift(1).rolling(window=7).std()
    
    return df

def predict_stock(symbol: str, days: int = 7) -> list[dict]:
    """
    Fetches historical data for `symbol`, trains a lightweight XGBoost model,
    and autoregressively predicts the next `days` prices.
    """
    logger.info(f"Starting stock prediction for {symbol} for {days} days")
    
    # Fetch data (last 1 year is enough for a fast, responsive model)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1y")
    
    if df.empty or len(df) < 30:
        raise ValueError(f"Not enough historical data found for symbol: {symbol}")
        
    df = df[['Close', 'Volume']].copy()
    
    # Prepare features
    n_lags = 5
    df = create_features(df, n_lags)
    df.dropna(inplace=True)
    
    if len(df) < 10:
         raise ValueError(f"Not enough valid data after feature engineering for {symbol}")
         
    # Define features and target (predicting next day's close)
    features = [col for col in df.columns if col != 'Close' and col != 'Volume']
    # For a simple autoregressive model, we use current row's features to predict current row's Close
    # Wait, actually we want to use t-1 features to predict t.
    # The way create_features is written, close_lag_1 is yesterday's price.
    # So if we predict today's 'Close' using today's 'close_lag_1', that's correct.
    
    X = df[features]
    y = df['Close']
    
    # Train XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X, y)
    
    # Autoregressive prediction
    predictions = []
    last_date = df.index[-1]
    
    # Track the exact rolling window of the last 14 closing prices
    last_prices = df['Close'].tail(14).tolist()
    
    for i in range(1, days + 1):
        # Determine the next trading day (skip weekends for simplicity)
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() > 4: # 5=Sat, 6=Sun
            next_date += timedelta(days=1)
            
        last_date = next_date
        
        # Prepare exact features for the next day using the sliding window
        features_dict = {}
        for lag in range(1, n_lags + 1):
            features_dict[f'close_lag_{lag}'] = last_prices[-lag]
            
        features_dict['ma_7'] = float(np.mean(last_prices[-7:]))
        features_dict['ma_14'] = float(np.mean(last_prices[-14:]))
        features_dict['volatility_7'] = float(np.std(last_prices[-7:], ddof=1))
        
        # Convert to DataFrame with columns matching training features
        input_row = pd.DataFrame([features_dict])[features]
        
        # Predict next close
        pred_price = float(model.predict(input_row)[0])
        
        # Save prediction
        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "price": round(pred_price, 2)
        })
        
        # Append predicted price to historical window for subsequent steps
        last_prices.append(pred_price)

    logger.info(f"Successfully predicted {days} days for {symbol}")
    return predictions
