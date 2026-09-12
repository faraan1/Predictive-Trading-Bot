import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from config.config import (
    SEQUENCE_LENGTH, 
    BATCH_SIZE, 
    LEARNING_RATE, 
    EPOCHS, 
    LSTM_HIDDEN_SIZE, 
    LSTM_NUM_LAYERS, 
    MODELS_DIR,
    DATA_DIR
)


class TimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int = SEQUENCE_LENGTH):
        self.seq_length = seq_length
        self.X, self.y = [], []
        
        for i in range(len(features) - seq_length):
            self.X.append(features[i : i + seq_length])
            self.y.append(targets[i + seq_length])
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = LSTM_HIDDEN_SIZE, num_layers: int = LSTM_NUM_LAYERS):
        super(LSTMPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Output from last timestep
        return self.sigmoid(out)


def train_and_predict_ticker(
    feature_file: Union[str, Path], 
    ticker: str, 
    epochs: int = EPOCHS, 
    seq_length: int = SEQUENCE_LENGTH
) -> float:
    """Trains the LSTM model for a specific ticker and returns the probability for the next trading day."""
    feature_file = Path(feature_file)
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature matrix missing: {feature_file}")

    df = pd.read_csv(feature_file)
    feature_cols = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'sentiment_score']
    
    # Fill missing columns/NAs gracefully
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Directional Target (1 if tomorrow > today)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna().reset_index(drop=True)

    if len(df) < seq_length + 5:
        return 0.50  # Fallback probability if dataset is too small

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    targets = df['Target'].values

    # Dataset & Loader setup
    dataset = TimeSeriesDataset(scaled_features, targets, seq_length)
    if len(dataset) == 0:
        return 0.50

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMPricePredictor(input_size=len(feature_cols))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    # Save model checkpoint dynamically per ticker
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / f"lstm_{ticker.lower()}.pth"
    torch.save(model.state_dict(), model_save_path)

    # Predict probability for the latest window sequence
    model.eval()
    with torch.no_grad():
        latest_seq = torch.tensor(scaled_features[-seq_length:], dtype=torch.float32).unsqueeze(0)
        prob = model(latest_seq).item()

    return float(prob)


if __name__ == "__main__":
    sample_feature_matrix = DATA_DIR / "processed" / "AAPL_feature_matrix.csv"
    test_prob = train_and_predict_ticker(sample_feature_matrix, "AAPL")
    print(f"AAPL Next Day Directional Probability: {test_prob:.4f}")