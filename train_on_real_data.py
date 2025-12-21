#!/usr/bin/env python3
"""
Train Precog models on real market data
Simple and robust training script
"""

import sys
import os
import csv
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timezone
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_ensemble_model import create_advanced_ensemble
from advanced_attention_mechanisms import create_enhanced_attention_ensemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_data(csv_file):
    """Load CSV data without pandas"""
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'timestamp': row['timestamp'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'symbol': row['symbol']
            })
    return data


def prepare_sequences(data, seq_len=60):
    """Prepare sequences for training"""
    # Use all data for training (simplified)
    sequences = []
    targets = []

    for i in range(seq_len, len(data)):
        # Simple feature vector: OHLCV values
        seq = []
        for j in range(i - seq_len, i):
            row = data[j]
            seq.extend([row['open'], row['high'], row['low'], row['close'], row['volume']])

        # Target: next price change percentage
        current_price = data[i-1]['close']
        next_price = data[i]['close']
        target = (next_price - current_price) / current_price

        sequences.append(seq)
        targets.append(target)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    return X, y


def train_model(model_type='attention', csv_file='crypto_training_data.csv', epochs=10):
    """Train model on real data"""

    print("🚀 TRAINING PRECOG MODEL ON REAL MARKET DATA")
    print("=" * 50)

    # Load data
    print(f"Loading data from {csv_file}...")
    data = load_csv_data(csv_file)
    print(f"Loaded {len(data)} data points")

    # Prepare sequences
    print("Preparing training sequences...")
    X, y = prepare_sequences(data, seq_len=60)

    if len(X) < 10:
        print("❌ Not enough data for training")
        return None

    print(f"Created {len(X)} training samples")

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape for model input
    # X is (n_samples, seq_len * features), convert to (n_samples, seq_len, features)
    n_samples, total_features = X_train.shape
    features_per_timestep = 5  # OHLCV
    seq_len = total_features // features_per_timestep

    X_train = X_train.reshape(n_samples, seq_len, features_per_timestep)
    X_test = X_test.reshape(len(X_test), seq_len, features_per_timestep)

    # Convert to tensors
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if model_type == 'attention':
        model = create_enhanced_attention_ensemble()
    else:
        model = create_advanced_ensemble()

    model.to(device)
    model.train()

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    batch_size = 16

    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Starting training for {epochs} epochs...")

    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            if isinstance(outputs, tuple):
                predictions, _ = outputs
            else:
                predictions = outputs

            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)

            loss = criterion(predictions, batch_y)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 2 == 0:
            print(".4f")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'best_loss': best_loss
            }, f"real_trained_{model_type}_model.pth")
            print(f"  💾 Saved best model (loss: {best_loss:.6f})")

    print("✅ Training completed!")

    # Evaluate
    print("Evaluating model...")
    model.eval()

    with torch.no_grad():
        test_predictions = []
        test_actuals = []

        for i in range(len(X_test)):
            x = X_test[i:i+1].to(device)
            actual = y_test[i].item()

            output = model(x)
            if isinstance(output, tuple):
                pred, _ = output
            else:
                pred = output

            pred_val = pred.cpu().numpy().flatten()[0]
            test_predictions.append(pred_val)
            test_actuals.append(actual)

    # Calculate metrics
    predictions = np.array(test_predictions)
    actuals = np.array(test_actuals)

    mape = np.mean(np.abs((predictions - actuals) / (np.abs(actuals) + 1e-6)))
    mae = np.mean(np.abs(predictions - actuals))

    print("\n📊 EVALUATION RESULTS:")
    print(".4f")
    print(".4f")

    # Compare with baseline
    baseline_mape = 1.025  # From synthetic data
    improvement = (baseline_mape - mape) / baseline_mape * 100
    print(".1f")

    # Estimate earnings
    miner52_reward = 0.180276  # TAO per prediction
    our_estimated_reward = max(0, (1 - mape) * 0.2)
    daily_estimate = our_estimated_reward * 24 * 6

    print("
💰 EARNINGS ESTIMATE:"    print(".6f")
    print(".3f")

    if our_estimated_reward > miner52_reward * 0.7:
        print("🎯 COMPETITIVE - Ready for deployment!")
    elif our_estimated_reward > miner52_reward * 0.4:
        print("💪 GETTING CLOSE - More training needed")
    else:
        print("🔄 MORE TRAINING REQUIRED")

    return {
        'mape': mape,
        'mae': mae,
        'improvement_percent': improvement,
        'estimated_daily_tao': daily_estimate,
        'model_saved': True
    }


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train on Real Market Data')
    parser.add_argument('--model', type=str, default='attention',
                       choices=['original', 'attention'])
    parser.add_argument('--data', type=str, default='crypto_training_data.csv')
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    # Check if data exists
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        print("Run: python3 simple_data_fetch.py --symbols BTC ETH")
        return 1

    # Train model
    results = train_model(args.model, args.data, args.epochs)

    if results:
        print("
✅ TRAINING SUCCESSFUL!"        print(f"Model saved as: real_trained_{args.model}_model.pth")
        print("
🎯 READY FOR DEPLOYMENT TESTING!"        return 0
    else:
        print("❌ Training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
