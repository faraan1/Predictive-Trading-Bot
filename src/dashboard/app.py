import os
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Trading Bot Dashboard", layout="wide")

st.title("📈 Predictive Financial Analytics & Automated Trading Bot")
st.markdown("Real-time price analytics, FinBERT news sentiment, and automated paper trading logs.")

# Sidebar Controls
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Asset Ticker", ["AAPL"])

# 1. Load Market & Sentiment Data
price_file = f"data/raw/{ticker}_historical.csv"
sentiment_file = "data/processed/AAPL_news_sentiment.csv"
trade_logs_file = "data/processed/trade_logs.json"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Market Price & Technical Indicators")
    if os.path.exists(price_file):
        df_price = pd.read_csv(price_file)
        st.line_chart(df_price.set_index("Date")[["Close", "SMA_20", "SMA_50"]])
    else:
        st.info("Market data file not found. Run market_data.py first.")

with col2:
    st.subheader("Financial News Sentiment (FinBERT)")
    if os.path.exists(sentiment_file):
        df_sent = pd.read_csv(sentiment_file)
        st.dataframe(df_sent[["headline", "sentiment_score", "positive", "negative"]].head(10))
    else:
        st.info("Sentiment data file not found. Run sentiment_analyzer.py first.")

st.divider()

# 2. Load Account Portfolio & Trade Logs
st.subheader("Automated Execution Engine - Account Status")

if os.path.exists(trade_logs_file):
    with open(trade_logs_file, "r") as f:
        logs = json.load(f)

    m1, m2 = st.columns(2)
    m1.metric("Available Cash Balance", f"${logs.get('account_balance', 0.0):,.2f}")
    m2.metric("Portfolio Holdings", str(logs.get("portfolio", {})))

    st.subheader("Trade Execution History")
    history_df = pd.DataFrame(logs.get("history", []))
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.write("No trades executed yet.")
else:
    st.info("No trade logs found. Run trade_executor.py first.")

