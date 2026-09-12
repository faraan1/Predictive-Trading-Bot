import yfinance as yf
import pandas as pd
import numpy as np

def fetch_live_feature_matrix(ticker):
    """
    Downloads live market data from Yahoo Finance and calculates 
    technical indicators required by the PyTorch model.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period="60d", interval="1d")
    
    if df.empty:
        return None

    df = df.reset_index()
    
    # Calculate live technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)  # Prevent division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Clean up warm-up NA rows
    df = df.dropna().reset_index(drop=True)
    return df

def get_latest_live_price(ticker):
    """
    Fetches latest price for ticker with fallback for PSX / limited-data assets.
    """
    try:
        stock = yf.Ticker(ticker)
        # 1-minute interval often fails on PSX (.KA) tickers on Yahoo Finance
        interval = "1d" if ticker.endswith(".KA") else "1m"
        period = "5d" if ticker.endswith(".KA") else "1d"
        
        live_data = stock.history(period=period, interval=interval)
        if not live_data.empty and "Close" in live_data.columns:
            return round(float(live_data["Close"].iloc[-1]), 2)
    except Exception as e:
        print(f"[Warning] Unable to fetch live price for {ticker}: {e}")
        
    return None