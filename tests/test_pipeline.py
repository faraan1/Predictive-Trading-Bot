import sys
from pathlib import Path

# Explicitly ensure project root is at the top of sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest
import numpy as np
import torch
import pandas as pd

from src.ml_pipeline.models.price_predictor import TimeSeriesDataset, LSTMPricePredictor
from config.config import SEQUENCE_LENGTH, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS


def test_time_series_dataset_shapes():
    """Verify sequence windowing logic and tensor output shapes."""
    num_samples = 100
    num_features = 7
    seq_length = 10

    mock_features = np.random.rand(num_samples, num_features)
    mock_targets = np.random.randint(0, 2, size=num_samples)

    dataset = TimeSeriesDataset(mock_features, mock_targets, seq_length=seq_length)

    assert len(dataset) == num_samples - seq_length
    
    sample_x, sample_y = dataset[0]
    assert sample_x.shape == (seq_length, num_features)
    assert sample_y.shape == (1,)
    assert isinstance(sample_x, torch.Tensor)


def test_lstm_model_forward_pass():
    """Verify PyTorch LSTM model input/output tensor dimensions and activation range."""
    batch_size = 16
    input_size = 7
    seq_length = SEQUENCE_LENGTH

    model = LSTMPricePredictor(
        input_size=input_size, 
        hidden_size=LSTM_HIDDEN_SIZE, 
        num_layers=LSTM_NUM_LAYERS
    )
    mock_input = torch.randn(batch_size, seq_length, input_size)

    output = model(mock_input)

    assert output.shape == (batch_size, 1)
    # Sigmoid output must be bounded between 0.0 and 1.0
    assert (output >= 0.0).all() and (output <= 1.0).all()