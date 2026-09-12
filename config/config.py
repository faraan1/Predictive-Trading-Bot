from pathlib import Path

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Data Fetching Defaults
# Expanded universe: 5 US Tech + Top 10 Pakistan Heavyweights (PSX)
DEFAULT_TICKERS = [
    # US Equities
    "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",
    
    # Pakistan Stock Exchange (PSX) Top 10
    "HBL.KA",   # Habib Bank Limited
    "MCB.KA",   # MCB Bank Limited
    "UBL.KA",   # United Bank Limited
    "MEBL.KA",  # Meezan Bank Limited
    "OGDC.KA",  # Oil & Gas Development Company
    "PPL.KA",   # Pakistan Petroleum Limited
    "ENGRO.KA", # Engro Corporation
    "FFC.KA",   # Fauji Fertilizer Company
    "LUCK.KA",  # Lucky Cement
    "SYS.KA"    # Systems Limited
]

# Export TICKERS alias to ensure backward compatibility with app.py imports
TICKERS = DEFAULT_TICKERS

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