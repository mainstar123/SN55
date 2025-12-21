#!/usr/bin/env python3
"""
Simple model for real market data training
Works with raw OHLCV features (5 features per timestep)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMarketPredictor(nn.Module):
    """Simple LSTM-based predictor for OHLCV data"""

    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.1)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq, hidden)

        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention
        attended = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden)

        # Output prediction
        prediction = self.output(attended)

        # Uncertainty estimation
        uncertainty = self.uncertainty_head(attended)

        return prediction, uncertainty


def create_simple_model():
    """Factory function for simple model"""
    return SimpleMarketPredictor()
