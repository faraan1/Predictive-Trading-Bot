from pathlib import Path

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Data Fetching Defaults
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"

# Technical Indicator Parameters
SMA_FAST = 20
SMA_SLOW = 50
RSI_PERIOD = 14

# LSTM Model Hyperparameters
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2

# Trading & Risk Rules
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
INITIAL_CAPITAL = 10000.0
MAX_POSITION_SIZE = 0.20  # Max 20% of capital per allocation