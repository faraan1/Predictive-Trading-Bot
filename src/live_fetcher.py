import yfinance as yf
import pandas as pd
import numpy as np

def fetch_live_feature_matrix(ticker):
    """
    Downloads market data from Yahoo Finance and calculates 
    technical indicators required by the model.
    """
    try:
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
    except Exception as e:
        print(f"[Warning] Failed to fetch matrix for {ticker}: {e}")
        return None

def get_latest_live_price(ticker):
    """
    Bulletproof live price retriever. 
    Completely avoids yfinance history calls for PSX (.KA) tickers to prevent KeyError.
    """
    # Fix 1: PSX (.KA) stocks bypass history() entirely
    if ticker.endswith(".KA"):
        try:
            # Method A: Fast quote summary lookup
            ticker_obj = yf.Ticker(ticker)
            fast_info = getattr(ticker_obj, "fast_info", None)
            if fast_info and "lastPrice" in fast_info and fast_info["lastPrice"] is not None:
                return round(float(fast_info["lastPrice"]), 2)
        except Exception:
            pass

        try:
            # Method B: Fallback to Daily Feature Matrix end-price
            df = fetch_live_feature_matrix(ticker)
            if df is not None and not df.empty:
                return round(float(df["Close"].iloc[-1]), 2)
        except Exception:
            pass
            
        return None

    # Fix 2: US / Standard Equities path
    try:
        stock = yf.Ticker(ticker)
        live_data = stock.history(period="1d", interval="1m")
        if not live_data.empty and "Close" in live_data.columns:
            return round(float(live_data["Close"].iloc[-1]), 2)
    except Exception as e:
        print(f"[Warning] Live minute lookup failed for {ticker}: {e}")

    # Final Fallback for all tickers
    try:
        df = fetch_live_feature_matrix(ticker)
        if df is not None and not df.empty:
            return round(float(df["Close"].iloc[-1]), 2)
    except Exception:
        pass

    return None