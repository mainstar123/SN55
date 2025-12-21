#!/usr/bin/env python3
"""
Enhanced Training Pipeline for Top 1 Miner Domination
Generate advanced features from real market data and train superior models
"""

import sys
import os
import csv
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timezone, timedelta
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_feature_engineering import create_comprehensive_feature_set
from advanced_attention_mechanisms import create_enhanced_attention_ensemble
from simple_data_fetch import SimpleCryptoDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedFeatureGenerator:
    """Generate advanced features from real market data"""

    def __init__(self):
        self.feature_names = None

    def generate_advanced_features(self, csv_file='crypto_training_data.csv',
                                 output_file='enhanced_training_data.csv'):
        """Generate 24+ advanced features from raw OHLCV data"""

        logger.info("🔧 Generating advanced features from real market data...")

        # Load raw data
        raw_data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_data.append({
                    'timestamp': row['timestamp'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'symbol': row['symbol']
                })

        logger.info(f"Loaded {len(raw_data)} raw data points")

        # Group by symbol for feature generation
        symbols = set(row['symbol'] for row in raw_data)
        enhanced_data = []

        for symbol in symbols:
            logger.info(f"Processing {symbol}...")

            # Filter data for this symbol
            symbol_data = [row for row in raw_data if row['symbol'] == symbol]

            if len(symbol_data) < 100:
                logger.warning(f"Skipping {symbol} - insufficient data ({len(symbol_data)} points)")
                continue

            # Convert to price dict format expected by feature engineering
            price_dict = {
                'open': np.array([row['open'] for row in symbol_data]),
                'high': np.array([row['high'] for row in symbol_data]),
                'low': np.array([row['low'] for row in symbol_data]),
                'close': np.array([row['close'] for row in symbol_data]),
                'volume': np.array([row['volume'] for row in symbol_data])
            }

            # Generate advanced features
            try:
                features_array, feature_names = create_comprehensive_feature_set(price_dict)
                self.feature_names = feature_names

                logger.info(f"Generated {features_array.shape[1]} advanced features for {symbol}")

                # Create enhanced data points
                for i, row in enumerate(symbol_data):
                    if i < features_array.shape[0]:  # Ensure we don't go out of bounds
                        enhanced_row = row.copy()
                        enhanced_row['features'] = features_array[i].tolist()
                        enhanced_row['feature_count'] = len(features_array[i])
                        enhanced_data.append(enhanced_row)

            except Exception as e:
                logger.error(f"Failed to generate features for {symbol}: {e}")
                continue

        # Save enhanced data
        if enhanced_data:
            # Write to CSV with feature columns
            fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'feature_count'] + \
                        [f'feature_{i}' for i in range(enhanced_data[0]['feature_count'])]

            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in enhanced_data:
                    csv_row = {
                        'timestamp': row['timestamp'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume'],
                        'symbol': row['symbol'],
                        'feature_count': row['feature_count']
                    }

                    # Add feature columns
                    for i, feature_val in enumerate(row['features']):
                        csv_row[f'feature_{i}'] = feature_val

                    writer.writerow(csv_row)

            logger.info(f"✅ Saved {len(enhanced_data)} enhanced data points to {output_file}")
            logger.info(f"   Features generated: {len(feature_names)} - {feature_names[:5]}...")

        return enhanced_data, self.feature_names


class EnhancedTrainer:
    """Enhanced trainer for advanced models with proper features"""

    def __init__(self, model_type='attention'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model = None
        self.feature_names = None

        logger.info(f"Enhanced Trainer initialized on {self.device}")

    def load_model(self):
        """Load the appropriate advanced model"""
        if self.model_type == 'attention':
            self.model = create_enhanced_attention_ensemble()
        else:
            # Could add other advanced models here
            self.model = create_enhanced_attention_ensemble()

        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Loaded {self.model_type} model with {total_params:,} parameters")

    def load_enhanced_data(self, csv_file='enhanced_training_data.csv'):
        """Load enhanced feature data"""
        logger.info(f"Loading enhanced data from {csv_file}")

        data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Extract features
                feature_count = int(row['feature_count'])
                features = [float(row[f'feature_{i}']) for i in range(feature_count)]

                data.append({
                    'timestamp': row['timestamp'],
                    'close': float(row['close']),
                    'features': features,
                    'symbol': row['symbol']
                })

        logger.info(f"Loaded {len(data)} enhanced data points")
        return data

    def prepare_sequences(self, data, seq_len=60):
        """Prepare sequences with advanced features"""

        # Convert to sequences
        sequences = []
        targets = []

        for i in range(seq_len, len(data)):
            # Input sequence: features only
            seq_features = [data[j]['features'] for j in range(i - seq_len, i)]
            sequences.append(seq_features)

            # Target: next price change
            current_price = data[i-1]['close']
            next_price = data[i]['close']
            target = (next_price - current_price) / current_price
            targets.append(target)

        # Convert to tensors
        X = np.array(sequences, dtype=np.float32)  # (n_samples, seq_len, n_features)
        y = np.array(targets, dtype=np.float32)

        logger.info(f"Prepared {len(X)} sequences with shape {X.shape}")
        return torch.from_numpy(X), torch.from_numpy(y)

    def train_enhanced_model(self, X_train, y_train, epochs=30, batch_size=16, learning_rate=0.001):
        """Train the enhanced model"""

        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        logger.info(f"🚀 Starting enhanced training for {epochs} epochs...")

        best_loss = float('inf')
        training_history = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
                if isinstance(outputs, tuple):
                    predictions, _ = outputs
                else:
                    predictions = outputs

                if predictions.dim() > 1:
                    predictions = predictions.squeeze(-1)

                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            scheduler.step()

            training_history.append(avg_epoch_loss)

            if (epoch + 1) % 5 == 0:
                logger.info("2d")
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.save_model(f"enhanced_{self.model_type}_model.pth")
                logger.info(f"  💾 Saved best model (loss: {best_loss:.6f})")

        logger.info("✅ Enhanced training completed!")
        return best_loss, training_history

    def evaluate_enhanced_model(self, X_test, y_test):
        """Evaluate the enhanced model"""

        self.model.eval()
        test_predictions = []
        test_actuals = []

        with torch.no_grad():
            for i in range(len(X_test)):
                x = X_test[i:i+1].to(self.device)
                actual = y_test[i].item()

                output = self.model(x)
                if isinstance(output, tuple):
                    pred, _ = output
                else:
                    pred = output

                pred_val = pred.cpu().numpy().flatten()[0]
                test_predictions.append(pred_val)
                test_actuals.append(actual)

        predictions = np.array(test_predictions)
        actuals = np.array(test_actuals)

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
            'total_predictions': len(predictions)
        }

        logger.info(".4f")
        logger.info(".4f")
        logger.info(".4f")
        logger.info(".1%")

        return results

    def save_model(self, filename):
        """Save the enhanced model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }, filename)
        logger.info(f"Enhanced model saved to {filename}")


def run_enhanced_training_pipeline():
    """Run the complete enhanced training pipeline"""

    print("🎯 ENHANCED TRAINING PIPELINE FOR TOP 1 MINER")
    print("=" * 60)

    # Step 1: Get more comprehensive real data
    print("\n📊 Step 1: Collecting Enhanced Real Market Data...")
    fetcher = SimpleCryptoDataFetcher()
    fetcher.create_training_csv(['BTC', 'ETH', 'ADA', 'SOL'], 'comprehensive_crypto_data.csv')

    # Step 2: Generate advanced features
    print("\n🔧 Step 2: Generating Advanced Features...")
    feature_generator = EnhancedFeatureGenerator()
    enhanced_data, feature_names = feature_generator.generate_advanced_features(
        'comprehensive_crypto_data.csv',
        'enhanced_training_data.csv'
    )

    if not enhanced_data:
        print("❌ Failed to generate enhanced features")
        return None

    print(f"✅ Generated {len(feature_names)} advanced features")

    # Step 3: Train enhanced model
    print("\n🚀 Step 3: Training Enhanced Model...")
    trainer = EnhancedTrainer('attention')
    trainer.feature_names = feature_names
    trainer.load_model()

    # Load and prepare enhanced data
    enhanced_data = trainer.load_enhanced_data('enhanced_training_data.csv')
    X, y = trainer.prepare_sequences(enhanced_data)

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train
    best_loss, history = trainer.train_enhanced_model(X_train, y_train, epochs=50, batch_size=16)

    # Evaluate
    print("\n📊 Step 4: Evaluating Enhanced Model...")
    test_results = trainer.evaluate_enhanced_model(X_test, y_test)

    # Compare with previous performance
    print("\n🏆 PERFORMANCE COMPARISON")
    print("-" * 40)

    # Previous baseline (from synthetic data)
    baseline_mape = 1.025
    improvement = (baseline_mape - test_results['mape']) / baseline_mape * 100

    print(".4f")
    print(".4f")
    print(".1f")

    # Compare with Miner 52
    miner52_reward = 0.180276  # TAO per prediction
    our_estimated_reward = max(0, (1 - test_results['mape']) * 0.2)
    competitiveness = our_estimated_reward / miner52_reward

    print(".6f")
    print(".6f")
    print(".2f")

    if test_results['mape'] < 0.3 and competitiveness > 0.9:
        status = "🚀 EXCELLENT - Ready for Top 1 domination!"
    elif test_results['mape'] < 0.5 and competitiveness > 0.7:
        status = "✅ VERY GOOD - Competitive with top miners!"
    elif test_results['mape'] < 0.8:
        status = "⚠️ GOOD - More optimization needed"
    else:
        status = "🔄 NEEDS MORE TRAINING"

    print(f"Status: {status}")

    # Save results
    results = {
        'training_loss': best_loss,
        'test_results': test_results,
        'improvement_percent': improvement,
        'competitiveness_ratio': competitiveness,
        'status': status,
        'model_saved': 'enhanced_attention_model.pth'
    }

    with open('enhanced_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n✅ ENHANCED TRAINING COMPLETED!")
    print(f"Results saved to: enhanced_training_results.json")
    print(f"Best model saved as: enhanced_attention_model.pth")

    return results


if __name__ == "__main__":
    results = run_enhanced_training_pipeline()
    if results and 'status' in results:
        print(f"\n🎯 FINAL STATUS: {results['status']}")
        if 'EXCELLENT' in results['status'] or 'VERY GOOD' in results['status']:
            print("🎉 Ready to deploy and dominate as Top 1 miner!")
        else:
            print("🔧 More work needed for top performance")
