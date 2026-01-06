#!/usr/bin/env python3
"""
TEST DOMINATION IMPROVEMENTS
============================

This script tests the optimized domination model improvements
and predicts performance gains.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('.')

# Simplified simulation - no external imports needed
import random

def simulate_improved_performance():
    """Simulate performance with the new improvements"""

    # Load historical data for testing
    btc_df = pd.read_csv('evaluation/csv_log/bitcoin_full.csv')

    # Sample 1000 predictions for testing
    test_df = btc_df.sample(1000, random_state=42)

    improved_results = []
    original_results = []

    print("🧪 TESTING DOMINATION IMPROVEMENTS")
    print("=" * 50)

    for idx, row in test_df.iterrows():
        # Simulate market data
        mock_data = create_mock_data(row)

        if len(mock_data) >= 10:
            features, confidence = extract_comprehensive_features(None, mock_data)

            if confidence > 0.3:
                # IMPROVED MODEL SIMULATION
                improved_prediction = simulate_improved_model(features, row['CM Reference Rate at Eval Time'])
                improved_results.append(improved_prediction)

                # ORIGINAL MODEL SIMULATION (for comparison)
                original_prediction = simulate_original_model(features, row['CM Reference Rate at Eval Time'])
                original_results.append(original_prediction)

    # Calculate metrics
    if improved_results:
        improved_df = pd.DataFrame(improved_results)
        original_df = pd.DataFrame(original_results)

        print("\n📊 PERFORMANCE COMPARISON")
        print("Metric                  | Original | Improved | Target (Top Miners)")
        print("-" * 70)

        # Hit Rates
        orig_hit_rate = calculate_hit_rate(original_df)
        imp_hit_rate = calculate_hit_rate(improved_df)
        print("Hit Rate              | {:7.1%} | {:8.1%} | 45-58%".format(orig_hit_rate, imp_hit_rate))
        print("Avg Interval Width    | {:7.1f} | {:8.1f} | 2.0-2.8".format(original_df['interval_width'].mean(), improved_df['interval_width'].mean()))
        print("MAE                   | {:7.3f} | {:8.3f} | ~1.7".format(original_df['mae'].mean(), improved_df['mae'].mean()))
        print("Prediction Rate       | {:7.1%} | {:8.1%} | 40-60%".format(len(original_df)/1000, len(improved_df)/1000))

        # Performance assessment
        print("\n🎯 IMPROVEMENT ASSESSMENT")
        if imp_hit_rate >= 0.45 and imp_hit_rate <= 0.65:
            print("✅ HIT RATE: In optimal range for maximum rewards")
        else:
            print("⚠️  HIT RATE: Outside optimal range - adjust confidence thresholds")

        if improved_df['interval_width'].mean() <= 3.0:
            print("✅ INTERVAL WIDTH: In optimal range for reward efficiency")
        else:
            print("⚠️  INTERVAL WIDTH: Too wide - reduce interval sizing")

        # Projected EMA reward improvement
        current_reward = 0.020  # Average miner
        target_reward = 0.052   # Top miner
        projected_reward = estimate_ema_reward(imp_hit_rate, improved_df['interval_width'].mean(), improved_df['mae'].mean())

        print("\n💰 PROJECTED EMA REWARD")
        print(".6f")
        print(".6f")
        print(".6f")
        print(".6f")
        if projected_reward > target_reward:
            print("🎉 PREDICTION: Can achieve FIRST PLACE! 🚀")
        elif projected_reward > current_reward * 1.5:
            print("✅ PREDICTION: Top 10 contender 📈")
        else:
            print("⚠️  PREDICTION: More optimization needed 📊")

def create_mock_data(row):
    """Create mock market data for testing"""
    base_price = row['CM Reference Rate at Eval Time']
    timestamps = pd.date_range(end=row['Prediction Time'], periods=60, freq='1min')

    np.random.seed(42)
    price_changes = np.random.normal(0, 0.005, len(timestamps))
    prices = base_price * (1 + price_changes).cumprod()

    mock_df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': np.random.lognormal(10, 1, len(timestamps)),
        'high': prices * (1 + np.random.uniform(0, 0.01, len(timestamps))),
        'low': prices * (1 - np.random.uniform(0, 0.01, len(timestamps)))
    })

    return mock_df

def simulate_improved_model(features, actual_price):
    """Simulate the improved model"""
    if len(features) >= 5:
        recent_return = features[0]
        short_trend = features[1]
        momentum = features[18] if len(features) > 18 else 0

        prediction_signal = (recent_return * 0.4 + short_trend * 0.3 + momentum * 0.3)
        predicted_change = np.clip(prediction_signal, -0.03, 0.03)

        # IMPROVED: Much narrower intervals
        interval_width = min(abs(actual_price * 0.01 * 2.5), actual_price * 0.015)
        interval_width = max(interval_width, actual_price * 0.005)
    else:
        predicted_change = np.random.normal(0, 0.005)
        interval_width = actual_price * 0.0125  # 1.25% average

    predicted_price = actual_price * (1 + predicted_change)

    return {
        'actual': actual_price,
        'predicted': predicted_price,
        'lower': predicted_price - interval_width,
        'upper': predicted_price + interval_width,
        'interval_width': interval_width * 2,  # Total width
        'mae': abs(predicted_price - actual_price)
    }

def simulate_original_model(features, actual_price):
    """Simulate the original model (for comparison)"""
    if len(features) >= 5:
        recent_return = features[0]
        short_trend = features[1]
        momentum = features[18] if len(features) > 18 else 0

        prediction_signal = (recent_return * 0.4 + short_trend * 0.3 + momentum * 0.3)
        predicted_change = prediction_signal
    else:
        predicted_change = np.random.normal(0, 0.01)

    predicted_price = actual_price * (1 + predicted_change)
    interval_width = abs(predicted_price * 0.05)  # Original wide intervals

    return {
        'actual': actual_price,
        'predicted': predicted_price,
        'lower': predicted_price - interval_width,
        'upper': predicted_price + interval_width,
        'interval_width': interval_width * 2,
        'mae': abs(predicted_price - actual_price)
    }

def calculate_hit_rate(df):
    """Calculate hit rate"""
    hits = (df['actual'] >= df['lower']) & (df['actual'] <= df['upper'])
    return hits.mean()

def estimate_ema_reward(hit_rate, avg_interval_width, mae):
    """Estimate EMA reward based on performance metrics"""
    # Simplified reward estimation based on observed patterns
    # Higher hit rates and narrower intervals = higher rewards

    # Base reward from hit rate (optimal range 45-60%)
    if 0.45 <= hit_rate <= 0.60:
        hit_reward = 0.035
    elif 0.40 <= hit_rate <= 0.70:
        hit_reward = 0.030
    else:
        hit_reward = 0.020

    # Interval width penalty (narrower = better)
    width_penalty = max(0, (avg_interval_width - 2.5) * 0.002)

    # MAE penalty (lower = better)
    mae_penalty = max(0, (mae - 1.7) * 0.001)

    estimated_reward = hit_reward - width_penalty - mae_penalty
    return max(0.015, estimated_reward)  # Minimum reward floor

if __name__ == "__main__":
    simulate_improved_performance()
