import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# 1. Dataset Preparation with Sliding Windows
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


# 2. PyTorch LSTM Architecture
class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take output from last time step
        return self.sigmoid(out)


# 3. Model Training Pipeline
def train_model(feature_file: str, epochs=15, seq_length=10):
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature matrix missing: {feature_file}")

    df = pd.read_csv(feature_file)
    
    # Create target: 1 if Close price tomorrow > Close price today, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna().reset_index(drop=True)

    feature_cols = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'sentiment_score']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    targets = df['Target'].values

    # Train/Test Split (80/20 sequential split)
    split_idx = int(len(scaled_features) * 0.8)
    train_X, test_X = scaled_features[:split_idx], scaled_features[split_idx:]
    train_y, test_y = targets[:split_idx], targets[split_idx:]

    train_dataset = TimeSeriesDataset(train_X, train_y, seq_length)
    test_dataset = TimeSeriesDataset(test_X, test_y, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize Network
    model = LSTMPricePredictor(input_size=len(feature_cols))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training LSTM Price Predictor...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            preds = model(batch_X)
            predicted_labels = (preds >= 0.5).float()
            correct += (predicted_labels == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = (correct / total) * 100
    print(f"\nModel Test Directional Accuracy: {accuracy:.2f}%")

    # Save Model Weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_aapl.pth")
    print("Saved trained model to models/lstm_aapl.pth")


if __name__ == "__main__":
    train_model("data/processed/AAPL_feature_matrix.csv")

