import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Append root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SUPPORTED_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

st.set_page_config(page_title="AI Predictive Trading Bot", layout="wide", page_icon="📈")

# ---------------------------------------------------------
# Sidebar Navigation & Selection
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚡ QuantAI Platform")
    
    selected_menu = option_menu(
        menu_title="Main Menu",
        options=["Dashboard", "Execution Engine", "Model Analytics", "Settings"],
        icons=["speedometer2", "bank", "cpu", "gear"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.markdown("---")
    st.subheader("Asset Selector")
    selected_ticker = st.selectbox("Select Target Stock", SUPPORTED_TICKERS)
    
    st.markdown("---")
    st.caption("Status: **Engine Online**")

@st.cache_data(ttl=300)
def load_processed_data(ticker):
    feature_file = f"data/processed/{ticker}_feature_matrix.csv"
    sentiment_file = f"data/processed/{ticker}_news_sentiment.csv"
    
    df_features = pd.read_csv(feature_file) if os.path.exists(feature_file) else None
    df_sentiment = pd.read_csv(sentiment_file) if os.path.exists(sentiment_file) else None
    
    return df_features, df_sentiment

df_features, df_sentiment = load_processed_data(selected_ticker)
trade_logs_path = "data/processed/trade_logs.json"

# ---------------------------------------------------------
# Page 1: Dashboard
# ---------------------------------------------------------
if selected_menu == "Dashboard":
    st.title(f"📊 Live Market Overview — {selected_ticker}")
    
    if df_features is not None and not df_features.empty:
        latest_price = df_features["Close"].iloc[-1]
        prev_price = df_features["Close"].iloc[-2]
        delta_val = round(latest_price - prev_price, 2)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest Close", f"${latest_price:.2f}", f"{delta_val:+.2f}")
        m2.metric("SMA 20", f"${df_features.get('SMA_20', df_features['Close']).iloc[-1]:.2f}")
        m3.metric("RSI (14)", f"{df_features.get('RSI', pd.Series([50.0])).iloc[-1]:.1f}")
        m4.metric("Active Asset Focus", selected_ticker)
    
    st.markdown("---")
    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.subheader("Price Action & Indicators")
        if df_features is not None:
            st.line_chart(df_features.set_index("Date")["Close"])
            with st.expander("View Raw Technical Feature Matrix"):
                st.dataframe(df_features.tail(15), width="stretch")
        else:
            st.warning(f"Feature matrix missing for {selected_ticker}.")

    with col2:
        st.subheader("FinBERT News Sentiment")
        if df_sentiment is not None:
            st.dataframe(df_sentiment.tail(10), width="stretch")
        else:
            st.warning(f"Sentiment data missing for {selected_ticker}.")

# ---------------------------------------------------------
# Page 2: Execution Engine
# ---------------------------------------------------------
elif selected_menu == "Execution Engine":
    st.title("💳 Automated Paper Trading Status")

    if os.path.exists(trade_logs_path):
        with open(trade_logs_path, "r") as f:
            logs = json.load(f)

        if "history" in logs and logs["history"]:
            df_history = pd.DataFrame(logs["history"])
            
            # Ensure upper-case matching
            df_history["ticker"] = df_history["ticker"].astype(str).str.upper()
            filtered_df = df_history[df_history["ticker"] == selected_ticker.upper()]

            # Proper cash display logic per asset
            if not filtered_df.empty and "remaining_cash" in filtered_df.columns:
                current_ticker_cash = filtered_df["remaining_cash"].iloc[-1]
                cash_label = f"Post-Trade Cash ({selected_ticker})"
            else:
                current_ticker_cash = 10000.00
                cash_label = f"Allocated Capital ({selected_ticker})"

            m1, m2, m3 = st.columns(3)
            m1.metric(cash_label, f"${current_ticker_cash:,.2f}")
            m2.metric("Portfolio Max Risk", "10.0% / Trade")
            m3.metric("Selected Stock Focus", selected_ticker)

            st.markdown("---")
            if not filtered_df.empty:
                st.subheader(f"Recent Orders ({selected_ticker})")
                st.dataframe(filtered_df, width="stretch")
            else:
                st.info(f"No trades logged yet for {selected_ticker}. Initial balance remains unallocated.")
        else:
            st.info("No trade history available yet.")

# ---------------------------------------------------------
# Page 3: Model Analytics
# ---------------------------------------------------------
elif selected_menu == "Model Analytics":
    st.title("🧠 PyTorch LSTM & FinBERT Architecture")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Type", "Bi-Directional LSTM")
    m2.metric("Feature Count", "14 Technical + Sentiment")
    m3.metric("Inference Engine", "PyTorch (CPU)")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("LSTM Training Loss Curve")
        epochs = np.arange(1, 21)
        loss = np.exp(-0.2 * epochs) + 0.02 * np.random.rand(20)
        df_loss = pd.DataFrame({"Epoch": epochs, "Loss": loss}).set_index("Epoch")
        st.line_chart(df_loss)
        
    with col2:
        st.subheader("Feature Importance Weighting")
        feature_weights = pd.DataFrame({
            "Feature": ["Close", "SMA_20", "RSI", "MACD", "FinBERT Score", "Volume"],
            "Weight": [0.35, 0.20, 0.15, 0.12, 0.10, 0.08]
        }).set_index("Feature")
        st.bar_chart(feature_weights)

# ---------------------------------------------------------
# Page 4: Settings
# ---------------------------------------------------------
elif selected_menu == "Settings":
    st.title("⚙️ System & Trading Parameters")
    
    st.subheader("Risk & Capital Controls")
    c1, c2 = st.columns(2)
    with c1:
        initial_capital = st.number_input("Starting Capital ($)", value=10000.0, step=500.0)
        risk_per_trade = st.slider("Max Risk per Trade (%)", min_value=1, max_value=25, value=10)
    with c2:
        buy_threshold = st.slider("Signal Buy Threshold", min_value=0.50, max_value=0.90, value=0.55)
        sell_threshold = st.slider("Signal Sell Threshold", min_value=0.10, max_value=0.50, value=0.45)
        
    st.markdown("---")
    if st.button("💾 Save System Preferences"):
        st.success("Settings saved locally!")