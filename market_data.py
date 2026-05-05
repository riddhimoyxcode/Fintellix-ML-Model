import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config import get_logger

logger = get_logger("market_data")

def get_live_quote(symbol: str) -> dict:
    """
    Fetches the live quote data for a symbol using yfinance.
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Fastinfo contains basic quote information without full history
        fast_info = ticker.fast_info
        
        if fast_info is None or 'lastPrice' not in fast_info:
            # Fallback to history if fast_info fails
            hist = ticker.history(period="5d")
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            previous = hist.iloc[-2] if len(hist) > 1 else latest
            
            price = latest['Close']
            previous_close = previous['Close']
            change = price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close else 0
            
            return {
                "symbol": symbol,
                "price": float(price),
                "previousClose": float(previous_close),
                "change": float(change),
                "changePercent": float(change_percent),
                "latestTradingDay": latest.name.strftime("%Y-%m-%d") if hasattr(latest, 'name') else datetime.now().strftime("%Y-%m-%d"),
                "source": "yfinance-fallback"
            }
            
        price = fast_info.last_price
        previous_close = fast_info.previous_close if hasattr(fast_info, 'previous_close') else price
        change = price - previous_close
        change_percent = (change / previous_close) * 100 if previous_close else 0
        
        return {
            "symbol": symbol,
            "price": float(price),
            "previousClose": float(previous_close),
            "change": float(change),
            "changePercent": float(change_percent),
            "latestTradingDay": datetime.now().strftime("%Y-%m-%d"),
            "source": "yfinance"
        }
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {e}")
        return None

def get_historical_prices(symbol: str) -> list:
    """
    Fetches the last 3 months of historical data (max 30 trading days for the UI).
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo", interval="1d")
        
        if hist.empty:
            return []
            
        # Keep only the last 30 rows to match the UI expectation
        hist = hist.tail(30)
        
        points = []
        for index, row in hist.iterrows():
            # Check for NaNs
            if pd.isna(row['Close']):
                continue
                
            points.append({
                "date": index.strftime("%Y-%m-%d") if hasattr(index, 'strftime') else str(index),
                "open": float(row.get('Open', 0)),
                "high": float(row.get('High', 0)),
                "low": float(row.get('Low', 0)),
                "close": float(row['Close']),
                "volume": int(row.get('Volume', 0))
            })
            
        return points
    except Exception as e:
        logger.error(f"Error fetching history for {symbol}: {e}")
        return []
