import os
import sys
import json
import pandas as pd
import streamlit as st

# Append root directory to sys.path to allow imports from src/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ml_pipeline.processing.market_data import fetch_market_data
from src.ml_pipeline.scrapers.news_scraper import fetch_latest_news
from src.ml_pipeline.models.sentiment_analyzer import analyze_news_sentiment
from src.ml_pipeline.processing.feature_builder import build_unified_dataset
from src.ml_pipeline.models.price_predictor import train_and_predict_ticker
from src.execution_engine.trade_executor import PaperTradingEngine

# Page Config
st.set_page_config(page_title="AI Predictive Trading Bot", layout="wide")

st.title("📈 Multi-Asset AI Predictive Trading Bot")
st.caption("Real-time price analytics, FinBERT news sentiment, PyTorch LSTM inference, and automated paper trading.")

# Sidebar Configuration
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Asset Ticker", ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"])

def execute_full_pipeline(selected_ticker):
    """Runs the end-to-end data pipeline & PyTorch inference dynamically."""
    with st.spinner(f"Running ML pipeline & training PyTorch LSTM for {selected_ticker}..."):
        # 1. Fetch market and news data
        price_file = fetch_market_data(ticker=selected_ticker)
        news_file = fetch_latest_news(ticker=selected_ticker)

        # 2. Sentiment analysis with explicit ticker
        sentiment_file = analyze_news_sentiment(news_file, ticker=selected_ticker)

        # 3. Build unified feature matrix
        feature_file = f"data/processed/{selected_ticker}_feature_matrix.csv"
        build_unified_dataset(price_file, sentiment_file, feature_file)

        # 4. PyTorch LSTM Inference
        prediction_prob = train_and_predict_ticker(feature_file, selected_ticker)

        # 5. Signal Execution
        df = pd.read_csv(feature_file)
        latest_price = float(df["Close"].iloc[-1])

        engine = PaperTradingEngine(initial_capital=10000.0)
        engine.execute_signal(
            ticker=selected_ticker,
            current_price=latest_price,
            prediction_probability=prediction_prob
        )
        engine.save_trade_logs()

    st.sidebar.success(f"Pipeline & Model Inference completed for {selected_ticker}!")

# Sidebar execution button
if st.sidebar.button("🚀 Run Pipeline Now"):
    execute_full_pipeline(ticker)

# Check file paths
feature_matrix_path = f"data/processed/{ticker}_feature_matrix.csv"
sentiment_path = f"data/processed/{ticker}_news_sentiment.csv"
trade_logs_path = "data/processed/trade_logs.json"

# Auto-run if files do not exist for chosen ticker
if not os.path.exists(feature_matrix_path) or not os.path.exists(sentiment_path):
    st.info(f"No pre-computed data found for {ticker}. Automatically executing initial pipeline run...")
    execute_full_pipeline(ticker)

# ---------------------------------------------------------
# Display Dashboard Sections
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Market Price & Technical Indicators ({ticker})")
    if os.path.exists(feature_matrix_path):
        df_features = pd.read_csv(feature_matrix_path)
        st.dataframe(df_features.tail(10), width="stretch")
        st.line_chart(df_features.set_index("Date")["Close"])
    else:
        st.error(f"Market data file missing for {ticker}.")

with col2:
    st.subheader(f"Financial News Sentiment ({ticker})")
    if os.path.exists(sentiment_path):
        df_sentiment = pd.read_csv(sentiment_path)
        st.dataframe(df_sentiment.tail(10), width="stretch")
    else:
        st.error(f"Sentiment data file missing for {ticker}.")

st.markdown("---")
st.subheader("Automated Execution Engine - Account Status")

if os.path.exists(trade_logs_path):
    with open(trade_logs_path, "r") as f:
        logs = json.load(f)
    
    st.metric(label="Current Balance", value=f"${logs.get('account_balance', 10000.0):,.2f}")
    
    if "history" in logs and logs["history"]:
        st.write("**Recent Executed Trades:**")
        st.dataframe(pd.DataFrame(logs["history"]), width="stretch")
    else:
        st.info("No trades executed yet.")
else:
    st.error("No trade logs found.")