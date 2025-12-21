#!/usr/bin/env python3
"""
Miner 31 vs Your Model Comparison
"""

import json
import os

def main():
    print("🎯 MINER 31 PERFORMANCE COMPARISON")
    print("=" * 60)

    # Based on the W&B run data you provided, here's what we know about miner 31:
    print("📊 MINER 31 BASELINE PERFORMANCE:")
    print("   • Position: Top 10-15 on Precog subnet 55")
    print("   • Average Reward: ~0.08-0.12 TAO per prediction")
    print("   • Consistency: Moderate (varies by market conditions)")
    print("   • Specialization: General crypto predictions")

    # Load your model performance
    print("\n🏆 YOUR ELITE DOMINATION MODEL:")

    if os.path.exists('elite_domination_results.json'):
        with open('elite_domination_results.json', 'r') as f:
            data = json.load(f)

        if 'model_performance' in data:
            perf = data['model_performance']
            print("   ✅ Model loaded successfully")
            if 'mape' in perf:
                print(".4f")
            if 'directional_accuracy' in perf:
                print(".1%")
            if 'estimated_tao_per_prediction' in perf:
                reward = perf['estimated_tao_per_prediction']
                print(".6f")

                # Compare with miner 31
                miner31_avg = 0.10  # Conservative estimate based on subnet performance
                if reward > miner31_avg:
                    improvement = (reward - miner31_avg) / miner31_avg * 100
                    print(f"   📈 Improvement: +{improvement:.1f}%")
                else:
                    deficit = (miner31_avg - reward) / miner31_avg * 100
                    print(f"   📉 Deficit: -{deficit:.1f}%")
    else:
        print("   ❌ elite_domination_results.json not found")
        print("   💡 Run your training script first")

    # Load backtest results
    import glob
    backtest_files = glob.glob('backtest_results_*.json')
    if backtest_files:
        latest = max(backtest_files, key=os.path.getctime)
        try:
            with open(latest, 'r') as f:
                backtest_data = json.load(f)

            print(f"\n📈 YOUR BACKTEST PERFORMANCE ({os.path.basename(latest)}):")
            if 'performance' in backtest_data:
                perf = backtest_data['performance']
                if 'mape' in perf:
                    print(".4f")
                if 'directional_accuracy' in perf:
                    print(".1%")

        except Exception as e:
            print(f"   ❌ Error loading backtest: {e}")

    print("\n🎯 DOMINATION STRATEGY:")
    print("   🎯 Target: Surpass miner 31's 0.10 TAO average")
    print("   📊 Focus: Achieve MAPE < 1.0% and directional accuracy > 75%")
    print("   🚀 Method: Deploy elite_domination_model.pth to mainnet")
    print("   ⏰ Timeline: Become #1 within 48 hours of deployment")

    print("\n💡 KEY ADVANTAGES OVER MINER 31:")
    print("   • Advanced ensemble architecture (GRU + Transformer)")
    print("   • 24 technical indicators vs basic predictions")
    print("   • Market regime adaptation (bull/bear/volatile)")
    print("   • Peak hour optimization (9-11, 13-15 UTC)")
    print("   • Continuous learning and adaptation")

if __name__ == "__main__":
    main()
