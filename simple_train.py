#!/usr/bin/env python3
"""
Simple model training on real market data
Trains Precog models without external dependencies
"""

import sys
import os
import csv
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timezone
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our models
from advanced_ensemble_model import create_advanced_ensemble
from advanced_attention_mechanisms import create_enhanced_attention_ensemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDataLoader:
    """Load and preprocess CSV data without pandas"""

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = []

    def load_data(self):
        """Load CSV data"""
        logger.info(f"Loading data from {self.csv_file}")

        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert to proper types
                processed_row = {
                    'timestamp': row['timestamp'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'symbol': row['symbol']
                }
                self.data.append(processed_row)

        logger.info(f"Loaded {len(self.data)} data points")
        return self.data

    def get_price_sequences(self, symbol=None, seq_len=60):
        """Extract price sequences for training"""

        # Filter by symbol if specified
        if symbol:
            filtered_data = [row for row in self.data if row['symbol'] == symbol]
        else:
            filtered_data = self.data

        if len(filtered_data) < seq_len + 1:
            logger.error(f"Not enough data: {len(filtered_data)} < {seq_len + 1}")
            return None, None

        sequences = []
        targets = []

        for i in range(seq_len, len(filtered_data)):
            # Input sequence (OHLCV)
            seq_data = []
            for j in range(i - seq_len, i):
                row = filtered_data[j]
                seq_data.extend([
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume']
                ])

            # Target: next close price change
            current_close = filtered_data[i-1]['close']
            next_close = filtered_data[i]['close']
            target = (next_close - current_close) / current_close

            sequences.append(seq_data)
            targets.append(target)

        # Convert to numpy arrays
        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        logger.info(f"Created {len(X)} sequences with {X.shape[1]} features each")
        return X, y


class SimpleTrainer:
    """Simple trainer for Precog models"""

    def __init__(self, model_type='attention'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model = None

        logger.info(f"Simple Trainer initialized on {self.device}")

    def load_model(self):
        """Load the appropriate model"""
        if self.model_type == 'attention':
            self.model = create_enhanced_attention_ensemble()
        else:
            self.model = create_advanced_ensemble()

        self.model.to(self.device)
        self.model.train()

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Loaded {self.model_type} model with {total_params:,} parameters")

    def prepare_data(self, X, y):
        """Prepare data for training"""
        # Convert to torch tensors
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        # Reshape for model input
        # X is (n_samples, seq_len * features), we need (n_samples, seq_len, features)
        n_samples, total_features = X.shape

        # Assume 5 features per timestep (OHLCV)
        features_per_timestep = 5
        seq_len = total_features // features_per_timestep

        if total_features % features_per_timestep != 0:
            logger.error(f"Feature count {total_features} not divisible by {features_per_timestep}")
            return None, None

        X_reshaped = X_tensor.view(n_samples, seq_len, features_per_timestep)

        logger.info(f"Reshaped data: {X_reshaped.shape} (samples, seq_len, features)")
        return X_reshaped, y_tensor

    def train(self, X, y, epochs=20, batch_size=32, learning_rate=0.001):
        """Train the model"""

        X_tensor, y_tensor = self.prepare_data(X, y)
        if X_tensor is None:
            return None

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        best_loss = float('inf')
        training_history = []

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                if hasattr(self.model, 'forward'):
                    outputs = self.model(batch_X)
                    if isinstance(outputs, tuple):
                        predictions, _ = outputs
                    else:
                        predictions = outputs

                    # Ensure predictions match targets shape
                    if predictions.dim() > 1:
                        predictions = predictions.squeeze(-1)

                    loss = criterion(predictions, batch_y)
                else:
                    logger.error("Model has no forward method")
                    continue

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / max(n_batches, 1)

            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("2d")
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_epoch_loss
            })

            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.save_model(f"best_{self.model_type}_model.pth")

        logger.info("Training completed!")
        logger.info(".6f")
        return {
            'final_loss': avg_epoch_loss,
            'best_loss': best_loss,
            'training_history': training_history
        }

    def evaluate(self, X, y):
        """Evaluate model performance"""

        X_tensor, y_tensor = self.prepare_data(X, y)
        if X_tensor is None:
            return None

        self.model.eval()

        with torch.no_grad():
            predictions = []
            actuals = []

            for i in range(len(X_tensor)):
                x = X_tensor[i:i+1].to(self.device)
                actual = y_tensor[i].item()

                output = self.model(x)
                if isinstance(output, tuple):
                    pred, _ = output
                else:
                    pred = output

                pred_val = pred.cpu().numpy().flatten()[0]
                predictions.append(pred_val)
                actuals.append(actual)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        mape = np.mean(np.abs((predictions - actuals) / (np.abs(actuals) + 1e-6)))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))

        # Directional accuracy
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        directional_acc = np.mean(pred_direction == actual_direction)

        results = {
            'mape': mape,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_acc,
            'total_predictions': len(predictions),
            'predictions_sample': predictions[:10].tolist(),
            'actuals_sample': actuals[:10].tolist()
        }

        logger.info(".4f"        logger.info(".4f"        logger.info(".4f"
        return results

    def save_model(self, filename):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }, filename)
        logger.info(f"Model saved to {filename}")


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Simple Model Training on Real Data')
    parser.add_argument('--data', type=str, default='crypto_training_data.csv',
                       help='CSV data file')
    parser.add_argument('--model', type=str, default='attention',
                       choices=['original', 'attention'], help='Model type')
    parser.add_argument('--symbol', type=str, help='Train on specific symbol only')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')

    args = parser.parse_args()

    print("🎯 SIMPLE MODEL TRAINING ON REAL DATA")
    print("=" * 50)

    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        print("Run: python3 simple_data_fetch.py --symbols BTC ETH")
        return 1

    # Load data
        print("\n📊 Loading training data...")
    data_loader = SimpleDataLoader(args.data)
    data_loader.load_data()

    # Get training data
    X, y = data_loader.get_price_sequences(args.symbol, args.seq_len)

    if X is None or len(X) < 10:
        print("❌ Not enough data for training")
        return 1

    print(f"Training samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Target range: {y.min():.6f} to {y.max():.6f}")

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train model
    print(f"\n🎯 Training {args.model} model...")
    trainer = SimpleTrainer(args.model)
    trainer.load_model()

    training_results = trainer.train(X_train, y_train,
                                   epochs=args.epochs,
                                   batch_size=args.batch_size)

    if training_results is None:
        print("❌ Training failed")
        return 1

    # Evaluate
        print("\n🔍 Evaluating model...")
    test_results = trainer.evaluate(X_test, y_test)

    if test_results is None:
        print("❌ Evaluation failed")
        return 1

    # Results summary
        print("\n" + "=" * 50)
    print("🏆 TRAINING RESULTS")
    print("=" * 50)

    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".1%")
    # Compare with previous baseline
    print("\n📊 IMPROVEMENT ANALYSIS:")
    print("Previous baseline (synthetic data): MAPE = 1.025")
    print(".4f")
    improvement = (1.025 - test_results['mape']) / 1.025 * 100
    print(".1f")
    if test_results['mape'] < 0.3:
        status = "🚀 EXCELLENT - Ready for deployment!"
    elif test_results['mape'] < 0.5:
        status = "✅ VERY GOOD - Competitive performance!"
    elif test_results['mape'] < 0.8:
        status = "⚠️ GOOD - More training recommended"
    else:
        status = "🔄 NEEDS MORE TRAINING"

    print(f"Status: {status}")

    # Estimate TAO earnings potential
    print("\n💰 EARNINGS POTENTIAL:")
    miner52_reward = 0.180276  # From our earlier analysis
    our_estimated_reward = max(0, (1 - test_results['mape']) * 0.2)
    potential_daily = our_estimated_reward * 24 * 6  # 6 predictions/hour during peak

    print(".6f")
    print(".3f")
    print(".3f")
    if our_estimated_reward > miner52_reward * 0.8:
        print("🎯 You're in the competitive range!")
    elif our_estimated_reward > miner52_reward * 0.5:
        print("💪 Getting close - more optimization needed!")
    else:
        print("🔧 More training required for competitiveness")

    print("\n✅ MODEL SAVED: best_{args.model}_model.pth")
    print("\n🎯 READY FOR DEPLOYMENT TEST!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
