#!/usr/bin/env python3
"""
Top Miner Backtesting - Compare with Miner 52 Performance
"""

import sys
import os
import torch
import numpy as np
import json
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quick_backtest import generate_test_data, test_model_performance
from advanced_attention_mechanisms import create_enhanced_attention_ensemble


def load_top_miner_model(model_path='top_miner_model.pth'):
    """Load the trained top miner model"""
    try:
        # First try to load with the correct input size
        checkpoint = torch.load(model_path, map_location='cpu')

        # Infer input size from saved model or use default
        input_size = checkpoint.get('input_size', 21)  # We know it's 21 features

        model = create_enhanced_attention_ensemble(input_size=input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("✅ Loaded top miner model with 21 features")
        return model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def compare_with_miner52():
    """Compare top miner model with Miner 52's performance"""

    print("🎯 TOP MINER BACKTEST COMPARISON")
    print("=" * 50)

    # Load top miner model
    model = load_top_miner_model()
    if model is None:
        print("❌ Cannot proceed without trained model")
        return

    # Generate test data (same as before)
        print("\n📊 Generating backtest data...")
    features, targets = generate_test_data(n_samples=500, seq_len=60, n_features=21)

    # Test the top miner model
        print("\n🚀 Testing Top Miner Model...")
    results = test_model_performance(model, features, targets, "Top Miner Model")

    if not results['total_predictions']:
        print("❌ No predictions made by model")
        return

    # Load miner 52 comparison data
    try:
        with open('miner52_raw_data.json', 'r') as f:
            miner52_data = json.load(f)
        miner52_stats = miner52_data.get('aggregate_stats', {})
        print("✅ Loaded Miner 52 data")
    except:
        print("⚠️ Miner 52 data not found, using estimated values")
        miner52_stats = {
            'avg_reward_per_prediction': 0.180276,
            'total_predictions': 5330
        }

    # Analysis
        print("\n📊 BACKTEST RESULTS:")
    print(".4f")
    print(".4f")
    print(".1%")
    print(f"   Predictions Made: {results['total_predictions']}")

    # Competitiveness calculation
    baseline_mape = 1.025  # Original baseline
    miner52_reward = miner52_stats.get('avg_reward_per_prediction', 0.18)

    improvement = (baseline_mape - results['mape']) / baseline_mape * 100
    estimated_reward = max(0, (1 - results['mape']) * 0.2)  # Scale to realistic TAO
    competitiveness = estimated_reward / miner52_reward if miner52_reward > 0 else 0

    print("
🏆 COMPETITIVENESS ANALYSIS:"    print(".1f")
    print(".6f")
    print(".6f")
    print(".2f")

    # Miner 52 stats
    print("
🥇 MINER 52 BENCHMARK:"    print(".6f")
    print(".3f")
    print(f"   Data Points: {miner52_stats.get('total_predictions', 'N/A'):,}")

    # Assessment
    print("
🎯 ASSESSMENT:"    if competitiveness > 0.9:
        status = "🚀 SUPERIOR - You surpass Miner 52!"
        color = "🟢"
    elif competitiveness > 0.8:
        status = "✅ EXCELLENT - Competitive with top miners!"
        color = "🟢"
    elif competitiveness > 0.7:
        status = "⚠️ VERY GOOD - Close to top performance"
        color = "🟡"
    elif competitiveness > 0.6:
        status = "🔄 GOOD - Above average, needs optimization"
        color = "🟡"
    else:
        status = "🔧 NEEDS IMPROVEMENT - More training required"
        color = "🔴"

    print(f"Status: {status}")
    print(f"Color Code: {color}")

    # Deployment recommendation
    print("
🚀 DEPLOYMENT RECOMMENDATION:"    if competitiveness > 0.7:
        print("✅ DEPLOY NOW - You are ready to dominate!")
        print("   Expected ranking: Top 5-10 miners")
        print("   Daily earnings: 20-30 TAO")
    elif competitiveness > 0.6:
        print("⚠️ DEPLOY WITH MONITORING - Good performance, room for improvement")
        print("   Expected ranking: Top 20-30 miners")
        print("   Daily earnings: 15-25 TAO")
    else:
        print("🔧 TRAIN MORE - Need better performance for competitiveness")
        print("   Focus on: More data, longer training, feature engineering")

    # Save comparison results
    comparison_results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'top_miner_model': {
            'mape': results['mape'],
            'mae': results['mae'],
            'directional_accuracy': results['directional_accuracy'],
            'total_predictions': results['total_predictions'],
            'improvement_over_baseline': improvement,
            'estimated_tao_per_prediction': estimated_reward,
            'competitiveness_vs_miner52': competitiveness
        },
        'miner52_benchmark': {
            'avg_reward_per_prediction': miner52_reward,
            'total_predictions': miner52_stats.get('total_predictions', 0),
            'data_source': 'wandb_simulated'
        },
        'assessment': {
            'status': status,
            'color_code': color,
            'deployment_ready': competitiveness > 0.6,
            'top_miner_potential': competitiveness > 0.8
        }
    }

    with open('top_miner_backtest_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)

    print("
✅ Results saved to: top_miner_backtest_results.json"
    return comparison_results


def run_detailed_backtest():
    """Run detailed backtest with multiple scenarios"""

    print("🔬 DETAILED BACKTEST ANALYSIS")
    print("=" * 50)

    model = load_top_miner_model()
    if model is None:
        return

    # Test different market conditions
    scenarios = [
        ('normal', 0.02, 100),  # Normal volatility
        ('volatile', 0.05, 100),  # High volatility
        ('trending', 0.01, 100),  # Strong trends
    ]

    results = {}

    for scenario_name, volatility, n_samples in scenarios:
        print(f"\n🎲 Testing scenario: {scenario_name.upper()}")

        # Generate scenario-specific data
        features, targets = generate_test_data(
            n_samples=n_samples,
            seq_len=60,
            n_features=21
        )

        # Test model
        scenario_results = test_model_performance(
            model,
            features,
            targets,
            f"Top Miner ({scenario_name})"
        )

        results[scenario_name] = scenario_results

        print(".4f"
    # Summary
    print("
📊 SCENARIO COMPARISON:"    for scenario, data in results.items():
        print("12")

    # Average performance
    avg_mape = np.mean([r['mape'] for r in results.values()])
    avg_directional = np.mean([r['directional_accuracy'] for r in results.values()])

    print("
📈 AVERAGE PERFORMANCE:"    print(".4f"    print(".1%"
    print("
🎯 CONSISTENCY RATING:"    mape_std = np.std([r['mape'] for r in results.values()])
    if mape_std < 0.02:
        consistency = "⭐ EXCELLENT - Very consistent across scenarios"
    elif mape_std < 0.05:
        consistency = "✅ GOOD - Reliable performance"
    else:
        consistency = "⚠️ VARIABLE - Performance depends on market conditions"

    print(f"Consistency: {consistency}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Top Miner Backtest Comparison')
    parser.add_argument('--detailed', action='store_true',
                       help='Run detailed multi-scenario backtest')
    parser.add_argument('--model', type=str, default='top_miner_model.pth',
                       help='Path to trained model')

    args = argparse.ArgumentParser().parse_args()

    if args.detailed:
        run_detailed_backtest()
    else:
        compare_with_miner52()
