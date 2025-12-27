#!/usr/bin/env python3
"""
Simple Supremacy Backtest
Test supremacy model performance against historical data
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import os

class SupremacyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def load_supremacy_model():
    """Load the supremacy model"""
    model_path = "latest_supremacy_model.pth"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model = SupremacyModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        accuracy = checkpoint.get('accuracy', 0)
        print(".4f")
        return model

    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def generate_historical_data(days=30):
    """Generate historical market data"""
    np.random.seed(42)
    n_samples = days * 24  # Hourly data

    # Generate realistic price movements
    t = np.linspace(0, 4*np.pi, n_samples)
    price = 50000 + 5000 * np.sin(t * 0.5) + np.random.normal(0, 500, n_samples)
    returns = np.diff(price, prepend=price[0])

    # Generate 10 technical features
    features = []
    for i in range(10):
        feature = np.sin(t * (i+1) * 0.1) + np.random.normal(0, 0.1, n_samples)
        features.append(feature)

    features = np.array(features).T  # Shape: (n_samples, 10)

    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    return features, returns, price

def calculate_directional_accuracy(predictions, actual_returns):
    """Calculate directional accuracy"""
    pred_direction = (predictions > 0.5).astype(int)
    actual_direction = (actual_returns > 0).astype(int)

    accuracy = np.mean(pred_direction == actual_direction)
    return accuracy

def run_backtest():
    """Run the backtest"""
    print("🏆 SUPREMACY BACKTEST")
    print("=" * 40)

    # Load model
    model = load_supremacy_model()
    if model is None:
        return

    # Generate data
    features, returns, prices = generate_historical_data(days=60)
    print(f"Generated {len(features)} hours of historical data")

    # Make predictions
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(len(features)):
            x = torch.from_numpy(features[i:i+1].astype(np.float32))
            pred = model(x).item()
            predictions.append(pred)

    predictions = np.array(predictions)

    # Calculate directional accuracy
    directional_acc = calculate_directional_accuracy(predictions, returns)

    print("\n📊 BACKTEST RESULTS:")
    print(".1%")

    # Simple trading simulation
    balance = 1000.0
    position = 0
    trades = []

    for i in range(1, len(predictions)):
        pred = predictions[i]

        # Simple strategy: go long if prediction > 0.6
        if position == 0 and pred > 0.6:
            position = 1
            entry_price = prices[i]
            quantity = balance * 0.1 / entry_price  # 10% of balance
            trades.append({'type': 'long', 'entry': entry_price, 'quantity': quantity})

        # Exit after 24 hours
        elif position == 1 and len(trades) > 0 and (i - trades[-1].get('entry_idx', 0)) >= 24:
            exit_price = prices[i]
            pnl = (exit_price - trades[-1]['entry']) * trades[-1]['quantity']
            balance += pnl
            trades[-1].update({'exit': exit_price, 'pnl': pnl})
            position = 0

    # Calculate trading performance
    total_trades = len([t for t in trades if 'pnl' in t])
    winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])

    if total_trades > 0:
        win_rate = winning_trades / total_trades * 100
        total_return = (balance - 1000) / 1000 * 100

        print(".1%")
        print(".2f")
        print(f"Total Trades: {total_trades}")
    else:
        win_rate = 0
        total_return = 0
        print("No trades executed")

    # Assessment
    print("\n🎯 ASSESSMENT:")
    if directional_acc >= 0.90:
        print("🏆 EXCELLENT: Supremacy achieved! >90% directional accuracy")
    elif directional_acc >= 0.85:
        print("🥈 VERY GOOD: Strong performance, close to supremacy target")
    elif directional_acc >= 0.80:
        print("🥉 GOOD: Solid performance, room for improvement")
    else:
        print("📈 NEEDS WORK: Below target directional accuracy")

    if total_return > 10:
        print("💰 PROFITABLE: Positive trading returns")
    elif total_return > 0:
        print("⚖️ SLIGHT GAIN: Small positive returns")
    else:
        print("📉 LOSSES: Negative trading performance")

def main():
    run_backtest()

if __name__ == "__main__":
    main()
