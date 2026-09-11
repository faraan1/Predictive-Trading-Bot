import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, seq_length=10):
        self.seq_length = seq_length
        self.X, self.y = [], []
        
        for i in range(len(features) - seq_length):
            self.X.append(features[i : i + seq_length])
            self.y.append(targets[i + seq_length])
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Output from last timestep
        return self.sigmoid(out)


def train_and_predict_ticker(feature_file: str, ticker: str, epochs=15, seq_length=10) -> float:
    """Trains the LSTM model for a specific ticker and returns the probability for the next trading day."""
    if not os.path.exists(feature_file):
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

    train_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = LSTMPricePredictor(input_size=len(feature_cols))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    # Save model checkpoint dynamically per ticker
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, f"lstm_{ticker.lower()}.pth")
    torch.save(model.state_dict(), model_save_path)

    # Predict probability for the latest window sequence
    model.eval()
    with torch.no_grad():
        latest_seq = torch.tensor(scaled_features[-seq_length:], dtype=torch.float32).unsqueeze(0)
        prob = model(latest_seq).item()

    return float(prob)


if __name__ == "__main__":
    test_prob = train_and_predict_ticker("data/processed/AAPL_feature_matrix.csv", "AAPL")
    print(f"AAPL Next Day Directional Probability: {test_prob:.4f}")