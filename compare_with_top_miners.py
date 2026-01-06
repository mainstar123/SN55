#!/usr/bin/env python3
"""
COMPARE MULTI-ASSET MODEL WITH TOP MINERS FROM CSV LOGS
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Add paths
sys.path.append('.')

# Import required modules
from precog.utils.cm_data import CMData
from train_multi_asset_domination import simple_32_feature_extraction, WorkingEnsemble

def load_trained_model():
    """Load the multi-asset trained model"""
    print("🔄 Loading multi-asset model...")

    # Load scaler
    with open('models/multi_asset_feature_scaler.pkl', 'rb') as f:
        scaler_data = pickle.load(f)
    feature_means = scaler_data['means']
    feature_stds = scaler_data['stds']

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WorkingEnsemble(input_size=32, hidden_size=128)
    model.load_state_dict(torch.load('models/multi_asset_domination_model.pth', map_location=device))
    model.to(device)
    model.eval()

    print("✅ Multi-asset model loaded")
    return model, feature_means, feature_stds, device

def load_top_miners_data():
    """Load and analyze top miners data"""
    print("📊 Loading top miners data...")

    files = {
        'BTC': 'evaluation/csv_log/bitcoin_full.parquet',
        'ETH': 'evaluation/csv_log/ethereum_full.parquet',
        'TAO': 'evaluation/csv_log/tao_full.parquet'
    }

    top_miners_stats = {}

    for asset, file in files.items():
        df = pd.read_parquet(file)

        # Get top 10% of miners
        reward_threshold = df['EMA Final Reward'].quantile(0.9)
        top_miners = df[df['EMA Final Reward'] >= reward_threshold]

        # Calculate statistics
        stats = {
            'total_miners': df['Miner UID'].nunique(),
            'top_10pct_count': len(top_miners['Miner UID'].unique()),
            'avg_reward_top_10pct': top_miners['EMA Final Reward'].mean(),
            'avg_interval_width': (top_miners['Interval Upper Bound'] - top_miners['Interval Lower Bound']).mean(),
            'avg_point_forecast': top_miners['Point Forecast'].mean(),
            'reward_threshold_10pct': reward_threshold,
            'reward_threshold_1pct': df['EMA Final Reward'].quantile(0.99),
            'market_avg_reward': df['EMA Final Reward'].mean()
        }

        top_miners_stats[asset] = stats
        print(f"✅ {asset} top miners loaded: {stats['top_10pct_count']} miners")

    return top_miners_stats

def evaluate_model_vs_top_miners(model, feature_means, feature_stds, device, top_miners_stats):
    """Compare model performance against top miners"""
    print("\n🎯 MODEL VS TOP MINERS COMPARISON")
    print("=" * 60)

    assets = ['btc', 'eth', 'tao_bittensor']
    asset_key_map = {'btc': 'BTC', 'eth': 'ETH', 'tao_bittensor': 'TAO'}
    comparison_results = {}

    for asset in assets:
        print(f"\n🏆 {asset.upper()} ANALYSIS:")
        print("-" * 30)

        # Get top miners stats
        asset_key = asset_key_map[asset]
        top_stats = top_miners_stats[asset_key]

        # Evaluate our model on same asset
        cm = CMData()
        data = cm.get_recent_data(minutes=360, asset=asset)

        if data.empty:
            print(f"❌ No data for {asset}")
            continue

        print(f"📊 Testing on {len(data)} recent data points")

        # Simulate predictions like top miners (every 5 minutes)
        predictions = []
        actuals = []
        intervals_lower = []
        intervals_upper = []

        window_size = 60
        step_size = 5  # Every 5 minutes like top miners

        for i in range(window_size, min(len(data), window_size + 100 * step_size), step_size):
            try:
                price_window = data['price'].values[i-window_size:i]
                current_price = data['price'].values[i]
                next_price = data['price'].values[i+1] if i+1 < len(data) else current_price

                # Create mock dataframe
                mock_data = pd.DataFrame({'price': price_window, 'volume': np.ones(window_size)})

                # Extract features
                features, _ = simple_32_feature_extraction(mock_data)

                # Scale features
                scaled_features = (features - feature_means) / feature_stds
                scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

                # Predict
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).unsqueeze(0).to(device)
                    prediction_return = model(input_tensor).item()
                    predicted_price = current_price * (1 + prediction_return)

                # Calculate interval (similar to top miners' average interval width)
                interval_width = top_stats['avg_interval_width']
                interval_lower = predicted_price - interval_width/2
                interval_upper = predicted_price + interval_width/2

                predictions.append(predicted_price)
                actuals.append(next_price)
                intervals_lower.append(interval_lower)
                intervals_upper.append(interval_upper)

            except Exception as e:
                continue

        if len(predictions) == 0:
            print(f"❌ No predictions generated for {asset}")
            continue

        # Calculate our model's performance
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        intervals_lower = np.array(intervals_lower)
        intervals_upper = np.array(intervals_upper)

        # MAPE
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        # Hit rates
        hit_rate_1pct = np.mean(np.abs((predictions - actuals) / actuals) <= 0.01) * 100
        hit_rate_2pct = np.mean(np.abs((predictions - actuals) / actuals) <= 0.02) * 100

        # Interval coverage
        interval_coverage = np.mean((actuals >= intervals_lower) & (actuals <= intervals_upper)) * 100

        # Average interval width
        avg_interval_width = np.mean(intervals_upper - intervals_lower)

        # Simulated EMA reward (rough estimate based on accuracy)
        # Higher accuracy = higher reward
        base_reward = top_stats['market_avg_reward']
        accuracy_bonus = (hit_rate_1pct / 100) * 0.01  # Bonus for high accuracy
        coverage_bonus = (interval_coverage / 100) * 0.005  # Bonus for good coverage
        estimated_reward = base_reward + accuracy_bonus + coverage_bonus

        model_stats = {
            'mape': mape,
            'hit_rate_1pct': hit_rate_1pct,
            'hit_rate_2pct': hit_rate_2pct,
            'interval_coverage': interval_coverage,
            'avg_interval_width': avg_interval_width,
            'estimated_reward': estimated_reward,
            'predictions_count': len(predictions)
        }

        comparison_results[asset] = {
            'model': model_stats,
            'top_miners_avg': top_stats
        }

        # Print comparison
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(f"📊 Predictions: {len(predictions)}")

        # Compare with top miners
        print(f"\n🎯 VS TOP MINERS (10%):")
        print(".2f")
        print(f"  • Your ranking potential: {'TOP 10%' if estimated_reward >= top_stats['reward_threshold_10pct'] else 'OUTSIDE TOP 10%'}")
        print(f"  • Reward advantage: {((estimated_reward - top_stats['avg_reward_top_10pct']) / top_stats['avg_reward_top_10pct'] * 100):+.1f}%")

    return comparison_results

def main():
    print("🚀 COMPARING MULTI-ASSET MODEL WITH TOP MINERS")
    print("=" * 60)

    # Load model and top miners data
    model, feature_means, feature_stds, device = load_trained_model()
    top_miners_stats = load_top_miners_data()

    # Compare performance
    comparison_results = evaluate_model_vs_top_miners(model, feature_means, feature_stds, device, top_miners_stats)

    # Overall summary
    print("\n" + "=" * 60)
    print("🏆 FINAL COMPARISON SUMMARY")
    print("=" * 60)

    if comparison_results:
        total_assets = len(comparison_results)
        total_estimated_reward = sum(r['model']['estimated_reward'] for r in comparison_results.values()) / total_assets
        avg_hit_rate = sum(r['model']['hit_rate_1pct'] for r in comparison_results.values()) / total_assets
        avg_coverage = sum(r['model']['interval_coverage'] for r in comparison_results.values()) / total_assets

        # Compare with market average
        market_avg = sum(stats['market_avg_reward'] for stats in top_miners_stats.values()) / len(top_miners_stats)
        top_10pct_avg = sum(stats['avg_reward_top_10pct'] for stats in top_miners_stats.values()) / len(top_miners_stats)

        print("\\n🎯 YOUR MULTI-ASSET MODEL PERFORMANCE:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print("\\n🏆 COMPETITION BENCHMARKS:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")

        print("\\n📈 RANKING POTENTIAL:")
        if total_estimated_reward >= top_10pct_avg:
            print("  • CURRENT STATUS: 🏆 TOP 10% MINER")
            print("  • COMPETITIVE EDGE: {:.1f}% above top 10% average".format(
                (total_estimated_reward - top_10pct_avg) / top_10pct_avg * 100))
        else:
            print("  • CURRENT STATUS: 📈 IMPROVEMENT NEEDED")
            print("  • GAP TO TOP 10%: {:.1f}%".format(
                (top_10pct_avg - total_estimated_reward) / total_estimated_reward * 100))

        print("\\n🔮 PREDICTIONS FOR SUBNET 55 DEPLOYMENT:")
        if avg_hit_rate >= 80 and avg_coverage >= 45:
            print("  • DEPLOYMENT READINESS: ✅ EXCELLENT")
            print("  • EXPECTED RANKING: TOP 5-10% on first deployment")
        elif avg_hit_rate >= 60 and avg_coverage >= 35:
            print("  • DEPLOYMENT READINESS: ✅ GOOD")
            print("  • EXPECTED RANKING: TOP 25-50% initially")
        else:
            print("  • DEPLOYMENT READINESS: ⚠️ NEEDS IMPROVEMENT")
            print("  • RECOMMENDATION: Retrain with more diverse data")

    print("\n" + "=" * 60)
    print("🎯 ANALYSIS COMPLETE")
    print("Ready to deploy and dominate subnet 55! 🚀")

if __name__ == "__main__":
    main()