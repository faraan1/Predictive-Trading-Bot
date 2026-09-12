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
    Fetches the most recent live market price for a given ticker.
    """
    stock = yf.Ticker(ticker)
    live_data = stock.history(period="1d", interval="1m")
    
    if not live_data.empty:
        return round(float(live_data["Close"].iloc[-1]), 2)
    return None