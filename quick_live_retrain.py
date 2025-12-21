#!/usr/bin/env python3
"""
Quick Live Data Retraining Before Deployment

This script fetches the most recent live market data and retrains your model
to improve performance before deploying to mainnet.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import csv
from datetime import datetime, timezone
from collections import defaultdict

sys.path.append('.')
from simple_data_fetch import SimpleCryptoDataFetcher
from advanced_attention_mechanisms import create_enhanced_attention_ensemble


def main():
    """Quick retraining on live data"""
    print("🚀 QUICK LIVE DATA RETRAINING - BEFORE DEPLOYMENT")
    print("=" * 60)

    # Step 1: Get fresh live data
    print("📊 Step 1: Fetching fresh live market data...")

    fetcher = SimpleCryptoDataFetcher()
    symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'BNB', 'LINK', 'XRP']
    live_data = []

    for symbol in symbols:
        try:
            print(f"Getting {symbol}...")
            fetcher.create_training_csv([symbol], f'quick_{symbol}.csv')

            with open(f'quick_{symbol}.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    live_data.append({
                        'symbol': symbol,
                        'close': float(row['close']),
                        'volume': float(row['volume'])
                    })
        except Exception as e:
            print(f"Skipping {symbol}: {e}")

    if len(live_data) < 1000:
        print("❌ Insufficient data")
        return 1

    print(f"✅ Collected {len(live_data)} live data points")

    # Step 2: Load your best model
    print("\\n🤖 Step 2: Loading your current model...")

    model_files = ['robust_deployment_model.pth']
    model_file = None

    for mf in model_files:
        if os.path.exists(mf):
            model_file = mf
            break

    if not model_file:
        print("❌ No model found")
        return 1

    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    model = create_enhanced_attention_ensemble(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✅ Loaded model from {model_file}")

    # Step 3: Quick retraining
    print("\\n🔄 Step 3: Quick retraining on live data...")

    # Group data by symbol
    symbol_groups = defaultdict(list)
    for row in live_data:
        symbol_groups[row['symbol']].append(row)

    # Use same normalization
    norm_params = checkpoint['normalization']
    close_mean = norm_params['close_mean']
    close_std = norm_params['close_std']
    vol_mean = norm_params['vol_mean']
    vol_std = norm_params['vol_std']

    all_features = []
    all_targets = []

    for symbol, data in symbol_groups.items():
        if len(data) < 30:
            continue

        closes = np.array([row['close'] for row in data])
        volumes = np.array([row['volume'] for row in data])

        closes_norm = (closes - close_mean) / (close_std + 1e-8)
        volumes_norm = (volumes - vol_mean) / (vol_std + 1e-8)

        # Add dummy data for missing OHLC features (use close for all)
        highs_norm = closes_norm  # Use close as proxy
        lows_norm = closes_norm   # Use close as proxy

        # Same features as original training
        features = [closes_norm, volumes_norm, highs_norm, lows_norm]

        # Momentum features
        momentum1 = np.diff(closes_norm, prepend=closes_norm[0])
        momentum5 = np.diff(closes_norm, n=5, prepend=np.full(5, closes_norm[0]))
        momentum10 = np.diff(closes_norm, n=10, prepend=np.full(10, closes_norm[0]))
        features.extend([momentum1, momentum5, momentum10])

        # Volatility
        volatility = np.zeros_like(closes_norm)
        for i in range(5, len(closes_norm)):
            volatility[i] = np.std(closes_norm[i-5:i])
        features.append(volatility)

        # RSI
        def robust_rsi(prices, period=14):
            deltas = np.diff(prices)
            gains = np.maximum(deltas, 0)
            losses = np.maximum(-deltas, 0)

            rsi_values = np.full_like(prices, 50.0)
            for i in range(period, len(prices)):
                avg_gain = np.mean(gains[i-period:i])
                avg_loss = np.mean(losses[i-period:i])

                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                rsi_values[i] = rsi
            return rsi_values / 100

        rsi = robust_rsi(closes_norm)
        features.append(rsi)

        # Volume momentum
        volume_momentum = np.diff(volumes_norm, n=3, prepend=np.full(3, volumes_norm[0]))
        features.append(volume_momentum)

        feature_matrix = np.column_stack(features)

        # Targets
        targets = []
        for i in range(1, len(closes_norm)):
            current = closes_norm[i-1]
            next_val = closes_norm[i]
            if current != 0:
                pct_change = (next_val - current) / abs(current)
                target = np.clip(pct_change * 50, -10, 10)
            else:
                target = 0
            targets.append(target)
        targets.insert(0, 0.0)
        targets = np.array(targets, dtype=np.float32)

        # Sequences
        seq_len = 25
        for i in range(seq_len, len(feature_matrix)):
            all_features.append(feature_matrix[i-seq_len:i])
            all_targets.append(targets[i])

    if not all_features:
        print("❌ No sequences created")
        return 1

    X_live = np.array(all_features, dtype=np.float32)
    y_live = np.array(all_targets, dtype=np.float32)

    print(f"Created {len(X_live)} training sequences")

    # Quick fine-tuning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.5)

    batch_size = 16
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_live), torch.from_numpy(y_live)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Fine-tuning...")

    for epoch in range(5):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_X, batch_y in loader:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    # Quick validation
    model.eval()
    val_predictions = []
    val_actuals = []

    val_size = min(100, len(X_live) // 3)
    with torch.no_grad():
        for i in range(val_size):
            x = torch.from_numpy(X_live[-val_size + i:i - val_size + i + 1]).to(device)
            if len(x) == 0:
                continue

            output = model(x)
            if isinstance(output, tuple):
                pred, _ = output
            else:
                pred = output
            pred_val = pred.cpu().numpy().flatten()[0]
            val_predictions.append(pred_val)
            val_actuals.append(y_live[-val_size + i])

    if val_predictions:
        directional_acc = np.mean(np.sign(np.array(val_predictions)) == np.sign(np.array(val_actuals)))
        print(f"\\n📊 Validation: {directional_acc:.1%} directional accuracy")

        # Calculate earnings
        miner221_reward = 0.239756
        estimated_reward = max(0.001, (directional_acc - 0.5) * 0.2)
        competitiveness = estimated_reward / miner221_reward
        daily_tao = estimated_reward * 24 * 6

        print(f"Estimated TAO/prediction: {estimated_reward:.6f}")
        print(f"vs Miner 221: {competitiveness:.2f}x")
        print(f"Daily TAO: {daily_tao:.1f}")

    # Save improved model
    quick_checkpoint = checkpoint.copy()
    quick_checkpoint['model_state_dict'] = model.state_dict()
    quick_checkpoint['quick_retrain_timestamp'] = datetime.now(timezone.utc).isoformat()
    quick_checkpoint['live_data_points'] = len(X_live)
    quick_checkpoint['directional_accuracy'] = directional_acc if 'directional_acc' in locals() else None

    torch.save(quick_checkpoint, 'quick_live_improved_model.pth')

    print("\\n✅ QUICK RETRAINING COMPLETED!")
    print("📁 Model saved as: quick_live_improved_model.pth")

    # Deployment recommendation
    if 'directional_acc' in locals() and directional_acc >= 0.6:
        print("\\n🚀 MODEL IMPROVED - READY FOR DEPLOYMENT!")
        print("Deploy command:")
        print("python3 start_domination_miner.py --model quick_live_improved_model.pth --deploy")
    else:
        print("\\n⚠️ Model performance similar - still deploy the robust model")
        print("python3 start_domination_miner.py --model robust_deployment_model.pth --deploy")

    return 0


if __name__ == "__main__":
    sys.exit(main())
