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


def run_pipeline_for_single_ticker(ticker_symbol, engine):
    """Processes pipeline for a specific asset ticker and updates paper trading state."""
    price_file = fetch_market_data(ticker=ticker_symbol)
    news_file = fetch_latest_news(ticker=ticker_symbol)
    sentiment_file = analyze_news_sentiment(news_file, ticker=ticker_symbol)

    feature_file = f"data/processed/{ticker_symbol}_feature_matrix.csv"
    build_unified_dataset(price_file, sentiment_file, feature_file)

    prediction_prob = train_and_predict_ticker(feature_file, ticker_symbol)

    df = pd.read_csv(feature_file)
    recent_rows = df.tail(30).reset_index(drop=True)

    # Generate dynamic probability series across backtest days
    for idx, row in recent_rows.iterrows():
        trade_price = float(row["Close"])
        sentiment_val = float(row.get("sentiment_score", 0.0))

        # Add step variation to signal probability across time
        step_variation = float(np.sin(idx) * 0.08)
        combined_prob = round(float(prediction_prob + step_variation + (sentiment_val * 0.05)), 4)

        engine.execute_signal(
            ticker=ticker_symbol,
            current_price=trade_price,
            prediction_probability=combined_prob,
            threshold=0.50
        )


def execute_full_multi_asset_pipeline():
    """Runs data collection, PyTorch ML inference, and trades across ALL supported tickers."""
    engine = PaperTradingEngine(initial_capital=10000.0)

    with st.spinner("Running ML pipeline & executing paper trades across ALL assets..."):
        for t in SUPPORTED_TICKERS:
            try:
                run_pipeline_for_single_ticker(t, engine)
            except Exception as e:
                st.warning(f"Error processing {t}: {e}")

        engine.save_trade_logs()
        gc.collect()

    st.sidebar.success("Pipeline & execution completed for all assets!")


if st.sidebar.button("🚀 Run Pipeline for All Assets"):
    execute_full_multi_asset_pipeline()

trade_logs_path = "data/processed/trade_logs.json"
feature_matrix_path = f"data/processed/{selected_ticker}_feature_matrix.csv"
sentiment_path = f"data/processed/{selected_ticker}_news_sentiment.csv"

if not os.path.exists(trade_logs_path) or not os.path.exists(feature_matrix_path):
    st.info("Initializing multi-asset data and model predictions...")
    execute_full_multi_asset_pipeline()

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

    st.metric(label="Current Balance", value=f"${logs.get('account_balance', 10000.0):,.2f}")

    if "history" in logs and logs["history"]:
        st.write("**Recent Executed Trades Across Portfolio:**")
        st.dataframe(pd.DataFrame(logs["history"]), width="stretch")
    else:
        st.info("No trades executed yet.")
else:
    st.error("No trade logs found.")