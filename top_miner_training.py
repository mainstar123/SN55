#!/usr/bin/env python3
"""
Top Miner Training - Simplified but Effective
Train on real data with enhanced features for domination
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

from advanced_attention_mechanisms import create_enhanced_attention_ensemble
from simple_data_fetch import SimpleCryptoDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_enhanced_features(price_data):
    """Create enhanced features from OHLCV data"""
    opens = np.array([row['open'] for row in price_data])
    highs = np.array([row['high'] for row in price_data])
    lows = np.array([row['low'] for row in price_data])
    closes = np.array([row['close'] for row in price_data])
    volumes = np.array([row['volume'] for row in price_data])

    features = []
    feature_names = []

    # Basic price features
    features.append(opens); feature_names.append('open')
    features.append(highs); feature_names.append('high')
    features.append(lows); feature_names.append('low')
    features.append(closes); feature_names.append('close')
    features.append(volumes); feature_names.append('volume')

    # Returns
    returns = np.diff(closes, prepend=closes[0])
    features.append(returns); feature_names.append('returns')

    # Moving averages
    for window in [5, 10, 20]:
        if len(closes) >= window:
            ma = np.convolve(closes, np.ones(window)/window, mode='valid')
            ma_padded = np.concatenate([np.full(window-1, closes[0]), ma])
            features.append(ma_padded); feature_names.append(f'ma_{window}')

    # Volatility (rolling std of returns)
    for window in [5, 10, 20]:
        if len(returns) >= window:
            vol = np.array([np.std(returns[max(0, i-window+1):i+1]) for i in range(len(returns))])
            features.append(vol); feature_names.append(f'volatility_{window}')

    # RSI
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')

        rs = np.concatenate([np.full(period, 1), avg_gains / (avg_losses + 1e-10)])
        rsi = 100 - (100 / (1 + rs))
        return rsi

    if len(closes) >= 14:
        rsi = calculate_rsi(closes)
        features.append(rsi); feature_names.append('rsi')

    # MACD
    if len(closes) >= 26:
        ema12 = np.convolve(closes, np.ones(12)/12, mode='valid')
        ema26 = np.convolve(closes, np.ones(26)/26, mode='valid')

        macd_line = np.concatenate([np.full(25, 0), ema12[-len(ema26):] - ema26])
        signal_line = np.convolve(macd_line, np.ones(9)/9, mode='valid')
        signal_line = np.concatenate([np.full(8, 0), signal_line])

        features.append(macd_line); feature_names.append('macd_line')
        features.append(signal_line); feature_names.append('macd_signal')

    # Bollinger Bands
    if len(closes) >= 20:
        ma20 = np.convolve(closes, np.ones(20)/20, mode='valid')
        ma20_padded = np.concatenate([np.full(19, closes[0]), ma20])

        rolling_std = np.array([np.std(closes[max(0, i-19):i+1]) for i in range(len(closes))])

        upper_band = ma20_padded + 2 * rolling_std
        lower_band = ma20_padded - 2 * rolling_std

        features.append(upper_band); feature_names.append('bb_upper')
        features.append(lower_band); feature_names.append('bb_lower')

    # Volume indicators
    volume_ma5 = np.convolve(volumes, np.ones(5)/5, mode='valid')
    volume_ma5 = np.concatenate([np.full(4, volumes[0]), volume_ma5])
    features.append(volume_ma5); feature_names.append('volume_ma5')

    # Price momentum
    for period in [1, 3, 5]:
        momentum = np.diff(closes, n=period, prepend=np.full(period, closes[0]))
        features.append(momentum); feature_names.append(f'momentum_{period}')

    # Combine all features
    feature_matrix = np.column_stack([f[:len(closes)] for f in features])

    return feature_matrix, feature_names


def prepare_top_miner_data(csv_file='crypto_training_data.csv'):
    """Prepare enhanced data for top miner training"""
    logger.info("Preparing enhanced training data for top miner...")

    # Load data
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

    # Group by symbol and create enhanced features
    symbols = set(row['symbol'] for row in data)
    all_sequences = []
    all_targets = []

    for symbol in symbols:
        symbol_data = [row for row in data if row['symbol'] == symbol]
        if len(symbol_data) < 100:
            continue

        # Create enhanced features
        feature_matrix, feature_names = create_enhanced_features(symbol_data)
        logger.info(f"Created {feature_matrix.shape[1]} features for {symbol}")

        # Create sequences
        seq_len = 60
        for i in range(seq_len, len(symbol_data)):
            # Input sequence
            seq = feature_matrix[i-seq_len:i]  # (seq_len, n_features)
            all_sequences.append(seq)

            # Target: next price change
            current_price = symbol_data[i-1]['close']
            next_price = symbol_data[i]['close']
            target = (next_price - current_price) / current_price
            all_targets.append(target)

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)

    logger.info(f"Prepared {len(X)} enhanced sequences")
    return X, y, feature_names


def train_top_miner_model(X_train, y_train, model_type='attention', input_size=None):
    """Train the top miner model"""
    logger.info(f"Training top miner {model_type} model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model with correct input size
    if input_size is None:
        input_size = X_train.shape[2]  # Infer from data
    model = create_enhanced_attention_ensemble(input_size=input_size)
    model.to(device)
    model.train()

    # Convert to tensors
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    batch_size = 16
    epochs = 30

    # Data loader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, 'top_miner_model.pth')
            logger.info(f"Saved best model with loss: {best_loss:.6f}")

    logger.info("Top miner training completed!")
    return best_loss


def run_top_miner_training():
    """Run the complete top miner training pipeline"""
    print("🎯 TOP MINER TRAINING PIPELINE")
    print("=" * 50)

    # Step 1: Get comprehensive data
    print("\n📊 Step 1: Collecting Comprehensive Market Data...")
    fetcher = SimpleCryptoDataFetcher()
    fetcher.create_training_csv(['BTC', 'ETH', 'ADA', 'SOL', 'DOT'], 'top_miner_data.csv')

    # Step 2: Prepare enhanced features
    print("\n🔧 Step 2: Creating Enhanced Features...")
    X, y, feature_names = prepare_top_miner_data('top_miner_data.csv')
    print(f"✅ Created {X.shape[1]} features per timestep")
    print(f"   Features: {feature_names[:5]}...")

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Step 3: Train top miner model
    print("\n🚀 Step 3: Training Top Miner Model...")
    input_size = X_train.shape[2]  # Number of features
    best_loss = train_top_miner_model(X_train, y_train, input_size=input_size)

    # Step 4: Evaluate
    print("\n📊 Step 4: Evaluating Top Miner Performance...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_test.shape[2]
    model = create_enhanced_attention_ensemble(input_size=input_size)
    checkpoint = torch.load('top_miner_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Evaluate on test set
    test_predictions = []
    test_actuals = []

    with torch.no_grad():
        for i in range(min(100, len(X_test))):  # Test on first 100 samples
            x = torch.from_numpy(X_test[i:i+1]).to(device)
            actual = y_test[i]

            output = model(x)
            if isinstance(output, tuple):
                pred, _ = output
            else:
                pred = output

            pred_val = pred.cpu().numpy().flatten()[0]
            test_predictions.append(pred_val)
            test_actuals.append(actual)

    predictions = np.array(test_predictions)
    actuals = np.array(test_actuals)

    mape = np.mean(np.abs((predictions - actuals) / (np.abs(actuals) + 1e-6)))
    mae = np.mean(np.abs(predictions - actuals))

    directional_acc = np.mean(np.sign(predictions) == np.sign(actuals))

    print(".4f")
    print(".4f")
    print(".1%")

    # Compare with benchmarks
    baseline_mape = 1.025  # Previous synthetic baseline
    miner52_reward = 0.180276  # Top miner TAO per prediction

    improvement = (baseline_mape - mape) / baseline_mape * 100
    estimated_reward = max(0, (1 - mape) * 0.2)
    competitiveness = estimated_reward / miner52_reward

    print("\n🏆 PERFORMANCE ANALYSIS:")
    print(".1f")
    print(".6f")
    print(".2f")

    if mape < 0.4 and competitiveness > 0.8:
        status = "🚀 EXCELLENT - Ready for Top 1 domination!"
        recommendation = "Deploy immediately - you will dominate!"
    elif mape < 0.6 and competitiveness > 0.6:
        status = "✅ VERY GOOD - Competitive with top miners!"
        recommendation = "Deploy and monitor - strong contender!"
    elif mape < 0.8:
        status = "⚠️ GOOD - Above average performance"
        recommendation = "Deploy with additional optimizations"
    else:
        status = "🔄 NEEDS IMPROVEMENT"
        recommendation = "More training data needed"

    print(f"Status: {status}")
    print(f"Recommendation: {recommendation}")

    # Save results
    results = {
        'mape': mape,
        'mae': mae,
        'directional_accuracy': directional_acc,
        'improvement_percent': improvement,
        'estimated_tao_per_prediction': estimated_reward,
        'competitiveness_vs_miner52': competitiveness,
        'status': status,
        'recommendation': recommendation,
        'model_saved': 'top_miner_model.pth',
        'features_used': len(feature_names),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }

    with open('top_miner_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)

    print("\n✅ TOP MINER TRAINING COMPLETED!")
    print("Results saved to: top_miner_results.json")
    print("Model saved to: top_miner_model.pth")

    return results


if __name__ == "__main__":
    results = run_top_miner_training()

    print(f"\n🎯 FINAL ASSESSMENT: {results['status']}")
    if results['competitiveness_vs_miner52'] > 0.8:
        print("🏆 You are ready to become the #1 miner!")
    elif results['competitiveness_vs_miner52'] > 0.5:
        print("💪 You are competitive - deploy and climb rankings!")
    else:
        print("🔧 More optimization needed for top performance")
