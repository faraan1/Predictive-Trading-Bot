# QuantAI Platform: Predictive Trading & Automated Risk Engine

An end-to-end quantitative trading and analytics platform that combines deep neural sequential forecasting (Bidirectional LSTM) with real-time financial news sentiment analysis (FinBERT) and dynamic risk management guards.

---

## 🌟 System Overview

QuantAI bridges the gap between raw market signals and quantitative decision-making. The system fetches live market features, runs deep learning inference to forecast directional price movements, monitors technical overbought/oversold indicators (RSI, SMA), and enforces account-level risk management rules (stop-loss, position sizing) before allowing trade execution.

```text
                  ┌───────────────────────────────┐
                  │    Live Financial Market Data │
                  └──────────────┬────────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
┌──────────────────────┐                    ┌─────────────────────┐
│ Technical Indicators │                    │ FinBERT News Scraper│
│   (RSI, SMA, OHLC)   │                    │ (Sentiment Metrics) │
└──────────┬───────────┘                    └──────────┬──────────┘
           │                                           │
           └─────────────────────┬─────────────────────┘
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │ Neural Network Pipeline   │
                   │ (Bidirectional PyTorch    │
                   │         Bi-LSTM)          │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │  Live Decision & Risk     │
                   │        Advisor            │
                   └─────────────┬─────────────┘
                                 │
                        [ Risk Check Pass? ]
                       /                    \
                     YES                     NO
                     /                        \
                    ▼                          ▼
     ┌────────────────────────────┐  ┌──────────────────┐
     │ Executed Trade Log Engine  │  │ Trade Blocked    │
     │ (Slippage/Fees Accounting) │  │ (Capital Guard)  │
     └────────────────────────────┘  └──────────────────┘