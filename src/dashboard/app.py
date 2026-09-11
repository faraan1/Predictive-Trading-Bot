import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import streamlit as st

# Append root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ml_pipeline.processing.market_data import fetch_market_data
from src.ml_pipeline.scrapers.news_scraper import fetch_latest_news
from src.ml_pipeline.models.sentiment_analyzer import analyze_news_sentiment
from src.ml_pipeline.processing.feature_builder import build_unified_dataset
from src.ml_pipeline.models.price_predictor import train_and_predict_ticker
from src.execution_engine.trade_executor import PaperTradingEngine

SUPPORTED_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

st.set_page_config(page_title="AI Predictive Trading Bot", layout="wide")

st.title("📈 Multi-Asset AI Predictive Trading Bot")
st.caption("Real-time price analytics, FinBERT news sentiment, PyTorch LSTM inference, and automated paper trading.")

st.sidebar.header("Configuration")
selected_ticker = st.sidebar.selectbox("Select Asset Ticker", SUPPORTED_TICKERS)


def execute_selected_ticker_pipeline(ticker_symbol):
    """Runs data collection, PyTorch ML inference, and paper trades exclusively for the selected ticker."""
    engine = PaperTradingEngine(initial_capital=10000.0, max_risk_per_trade=0.10)

    with st.spinner(f"Running ML pipeline & executing paper trades for {ticker_symbol}..."):
        # 1. Fetch market and news data
        price_file = fetch_market_data(ticker=ticker_symbol)
        news_file = fetch_latest_news(ticker=ticker_symbol)

        # 2. Sentiment analysis
        sentiment_file = analyze_news_sentiment(news_file, ticker=ticker_symbol)

        # 3. Build unified feature matrix
        feature_file = f"data/processed/{ticker_symbol}_feature_matrix.csv"
        build_unified_dataset(price_file, sentiment_file, feature_file)

        # 4. PyTorch LSTM Inference
        prediction_prob = train_and_predict_ticker(feature_file, ticker_symbol)

        # 5. Execute paper trades across recent price history
        df = pd.read_csv(feature_file)
        recent_rows = df.tail(15).reset_index(drop=True)

        for idx, row in recent_rows.iterrows():
            trade_price = float(row["Close"])
            sentiment_val = float(row.get("sentiment_score", 0.0))

            step_variation = float(np.sin(idx) * 0.07)
            combined_prob = round(float(prediction_prob + step_variation + (sentiment_val * 0.05)), 4)

            engine.execute_signal(
                ticker=ticker_symbol,
                current_price=trade_price,
                prediction_probability=combined_prob,
                threshold=0.50
            )

        engine.save_trade_logs()
        gc.collect()

    st.sidebar.success(f"Pipeline & execution completed for {ticker_symbol}!")


# Sidebar execution button
if st.sidebar.button(f"🚀 Run Pipeline for {selected_ticker}"):
    execute_selected_ticker_pipeline(selected_ticker)

trade_logs_path = "data/processed/trade_logs.json"
feature_matrix_path = f"data/processed/{selected_ticker}_feature_matrix.csv"
sentiment_path = f"data/processed/{selected_ticker}_news_sentiment.csv"

# Auto-run if pre-computed data is missing for the chosen ticker
if not os.path.exists(feature_matrix_path) or not os.path.exists(sentiment_path):
    st.info(f"Initializing data and model predictions for {selected_ticker}...")
    execute_selected_ticker_pipeline(selected_ticker)

# ---------------------------------------------------------
# Display Dashboard Sections
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Market Price & Technical Indicators ({selected_ticker})")
    if os.path.exists(feature_matrix_path):
        df_features = pd.read_csv(feature_matrix_path)
        st.dataframe(df_features.tail(10), width="stretch")
        st.line_chart(df_features.set_index("Date")["Close"])
    else:
        st.error(f"Market data file missing for {selected_ticker}.")

with col2:
    st.subheader(f"Financial News Sentiment ({selected_ticker})")
    if os.path.exists(sentiment_path):
        df_sentiment = pd.read_csv(sentiment_path)
        st.dataframe(df_sentiment.tail(10), width="stretch")
    else:
        st.error(f"Sentiment data file missing for {selected_ticker}.")

st.markdown("---")
st.subheader("Automated Execution Engine - Account Status")

if os.path.exists(trade_logs_path):
    with open(trade_logs_path, "r") as f:
        logs = json.load(f)

    st.metric(label="Current Cash Balance", value=f"${logs.get('account_balance', 10000.0):,.2f}")

    if "history" in logs and logs["history"]:
        df_history = pd.DataFrame(logs["history"])
        
        # Filter trades specifically for the selected ticker
        filtered_df = df_history[df_history["ticker"] == selected_ticker]

        if not filtered_df.empty:
            st.write(f"**Executed Trades for {selected_ticker}:**")
            st.dataframe(filtered_df, width="stretch")
        else:
            st.info(f"No executed trades found for {selected_ticker} yet. Click '🚀 Run Pipeline for {selected_ticker}' to generate signals.")
    else:
        st.info("No trades executed yet.")
else:
    st.error("No trade logs found.")