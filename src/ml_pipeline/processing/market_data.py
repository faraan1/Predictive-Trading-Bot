import os
import pandas as pd
import yfinance as yf


def fetch_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical OHLCV data for a given ticker from Yahoo Finance."""
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}.")

    # Flatten multi-index columns if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic technical indicators: SMA_20, SMA_50, RSI, and MACD."""
    df = df.copy()

    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Relative Strength Index (RSI - 14 period)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12-period EMA - 26-period EMA)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df


def save_processed_data(df: pd.DataFrame, ticker: str, output_dir: str = "data/raw"):
    """Save processed DataFrame to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{ticker}_historical.csv")
    df.to_csv(file_path, index=False)
    print(f"Successfully saved market data to {file_path}")


def fetch_market_data(ticker: str = "AAPL", start_date: str = "2023-01-01", end_date: str = "2026-01-01") -> str:
    """Wrapper function for main pipeline orchestrator."""
    data = fetch_historical_data(ticker, start_date, end_date)
    data_with_indicators = add_technical_indicators(data)
    save_processed_data(data_with_indicators, ticker)
    return f"data/raw/{ticker}_historical.csv"


if __name__ == "__main__":
    fetch_market_data("AAPL")