# AI Predictive Trading Bot 📈 🤖

An end-to-end multi-asset predictive trading application powered by PyTorch LSTM time-series modeling, FinBERT news sentiment analysis, and Streamlit visualization.

---

## 🌟 Key Features

* **Market Indicator Ingestion:** Real-time ticker price and moving average technical indicators via `yfinance`.
* **Financial News Sentiment Analysis:** Fine-tuned FinBERT NLP pipeline evaluating headline sentiment.
* **LSTM Price Prediction:** PyTorch sequence model trained on merged market indicators and sentiment metrics.
* **Automated Paper Trading:** Simulated confidence-based trade execution engine with logging.
* **Streamlit Interactive Dashboard:** Real-time deployment interface displaying live indicators, news sentiment, and trade signals.

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Deep Learning & NLP** | PyTorch, Hugging Face Transformers (FinBERT) |
| **Data & Analytics** | Pandas, NumPy, yfinance |
| **Frontend & Plotting** | Streamlit, Plotly |

---

## 📂 Project Structure

```text
```text
predictive-trading-bot/
|-- models/                   # Saved PyTorch (.pth) model weights
|-- src/
|   |-- dashboard/            # Streamlit app frontend
|   |-- ml_pipeline/          # PyTorch LSTM & FinBERT modules
|   |-- processing/           # Data fetchers & feature matrices
|-- requirements.txt          # Production dependencies
|-- README.md                 # Project documentation