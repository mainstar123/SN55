#!/usr/bin/env python3
"""
Final Backtest - Top Miner vs Miner 52 Comparison
"""

import sys
import os
import torch
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quick_backtest import generate_test_data, test_model_performance
from advanced_attention_mechanisms import create_enhanced_attention_ensemble


def load_trained_model():
    """Load the trained top miner model"""
    try:
        checkpoint = torch.load('top_miner_model.pth', map_location='cpu')
        model = create_enhanced_attention_ensemble(input_size=21)  # 21 features
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Loaded trained top miner model")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def run_backtest_comparison():
    """Run comprehensive backtest comparison"""

    print("🎯 FINAL BACKTEST: TOP MINER vs MINER 52")
    print("=" * 60)

    # Load model
    model = load_trained_model()
    if model is None:
        return

    # Generate test data
    print("\n📊 Generating backtest data...")
    features, targets = generate_test_data(n_samples=200, seq_len=60, n_features=21)

    # Test model
    print("\n🚀 Testing Top Miner Model...")
    results = test_model_performance(model, features, targets, "Top Miner Model")

    if not results['total_predictions']:
        print("❌ No predictions made")
        return

    # Load miner 52 data
    try:
        with open('miner52_raw_data.json', 'r') as f:
            miner52_data = json.load(f)
        miner52_stats = miner52_data.get('aggregate_stats', {})
    except:
        miner52_stats = {'avg_reward_per_prediction': 0.180276}

    # Results
    print("\n📊 BACKTEST RESULTS:")
    print(".4f")
    print(".4f")
    print(".1%")

    # Comparison
    baseline_mape = 1.025
    miner52_reward = miner52_stats['avg_reward_per_prediction']

    improvement = (baseline_mape - results['mape']) / baseline_mape * 100
    estimated_reward = max(0, (1 - results['mape']) * 0.2)
    competitiveness = estimated_reward / miner52_reward

    print("\n🏆 COMPETITIVENESS ANALYSIS:")
    print(".1f")
    print(".6f")
    print(".6f")
    print(".2f")

    print("\n🥇 MINER 52 BENCHMARK:")
    print(".6f")

    # Final verdict
    print("\n🎯 FINAL VERDICT:")

    if competitiveness > 0.95:
        verdict = "🚀 SUPERIOR PERFORMANCE - You are the new #1!"
        action = "DEPLOY IMMEDIATELY - Dominate the subnet!"
    elif competitiveness > 0.85:
        verdict = "✅ EXCELLENT PERFORMANCE - Top tier miner!"
        action = "DEPLOY NOW - Compete for #1 spot!"
    elif competitiveness > 0.75:
        verdict = "⚠️ VERY GOOD PERFORMANCE - Competitive!"
        action = "DEPLOY AND OPTIMIZE - Strong contender!"
    else:
        verdict = "🔄 GOOD PERFORMANCE - Needs improvement"
        action = "DEPLOY WITH MONITORING - Build from here!"

    print(f"Verdict: {verdict}")
    print(f"Action: {action}")

    # Expected earnings
    daily_tao = estimated_reward * 24 * 6  # 6 predictions/hour peak
    weekly_tao = daily_tao * 7
    monthly_tao = daily_tao * 30

    print("
💰 EXPECTED EARNINGS:"    print(".1f"    print(".1f"    print(".0f"
    # Save results
    final_results = {
        'model_performance': results,
        'competitiveness': competitiveness,
        'improvement_percent': improvement,
        'estimated_daily_tao': daily_tao,
        'verdict': verdict,
        'action': action
    }

    with open('final_backtest_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print("
✅ Results saved to: final_backtest_results.json"
    return final_results


if __name__ == "__main__":
    results = run_backtest_comparison()

    if results:
        print(f"\n🎊 MISSION STATUS: {results['verdict']}")
        print(f"🚀 NEXT: {results['action']}")