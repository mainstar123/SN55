#!/usr/bin/env python3
"""
Quick training on real market data
"""

import sys
import os
import csv
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_model import create_simple_model

def load_data(csv_file):
    """Load CSV data"""
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
    return data

def prepare_data(data, seq_len=60):
    """Prepare sequences"""
    sequences = []
    targets = []

    for i in range(seq_len, len(data)):
        seq = []
        for j in range(i - seq_len, i):
            row = data[j]
            seq.extend([row['open'], row['high'], row['low'], row['close'], row['volume']])

        current_price = data[i-1]['close']
        next_price = data[i]['close']
        target = (next_price - current_price) / current_price

        sequences.append(seq)
        targets.append(target)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    # Reshape
    n_samples, total_features = X.shape
    features_per_timestep = 5
    seq_len = total_features // features_per_timestep
    X = X.reshape(n_samples, seq_len, features_per_timestep)

    return torch.from_numpy(X), torch.from_numpy(y)

def train_on_real_data(csv_file='crypto_training_data.csv'):
    """Train on real data"""

    print("Training on real market data...")

    # Load and prepare data
    data = load_data(csv_file)
    print(f"Loaded {len(data)} data points")

    X, y = prepare_data(data)
    print(f"Prepared {len(X)} training samples")

    # Split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_simple_model()
    model.to(device)
    model.train()

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    batch_size = 16

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("Starting training...")

    for epoch in range(10):
        total_loss = 0
        n_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)
            if isinstance(outputs, tuple):
                predictions, _ = outputs
            else:
                predictions = outputs

            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)

            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(".4f")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'timestamp': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
    }, 'real_trained_model.pth')

    print("Model saved!")

    # Quick evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test[:10].to(device))
        if isinstance(test_outputs, tuple):
            test_preds, _ = test_outputs
        else:
            test_preds = test_outputs

        if test_preds.dim() > 1:
            test_preds = test_preds.squeeze(-1)

        predictions = test_preds.cpu().numpy()
        actuals = y_test[:10].numpy()

        mape = np.mean(np.abs((predictions - actuals) / (np.abs(actuals) + 1e-6)))
        print(".4f")

    return mape

if __name__ == "__main__":
    mape = train_on_real_data()
    print(f"Training completed! MAPE: {mape:.4f}")
    if mape < 0.8:
        print("SUCCESS: Model improved significantly!")
    else:
        print("Model needs more training or data")
