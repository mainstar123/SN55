#!/usr/bin/env python3
"""
Analyze Miner 31 Performance vs Your Model
"""

import pandas as pd
import numpy as np
import json
import os

def analyze_miner31_vs_your_model():
    """Compare miner 31's performance with your current model"""

    print("🎯 MINER 31 PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Check if we have miner 31 data
    data_file = 'miner31_performance_data.csv'
    if not os.path.exists(data_file):
        print(f"❌ Miner 31 data file not found: {data_file}")
        print("💡 Need to export miner 31 data from W&B first")
        return

    # Load miner 31 data
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Loaded miner 31 data: {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Basic analysis of miner 31's performance
    print("
📊 MINER 31 PERFORMANCE SUMMARY:"    print(f"   Records: {len(df)}")
    print(f"   Columns: {len(df.columns)}")

    # Look for reward data
    reward_cols = [col for col in df.columns if 'reward' in col.lower()]
    if reward_cols:
        for col in reward_cols:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"   {col}:")
                print(".6f"                print(".6f"                print(f"      Count: {len(values)}")

    # Look for prediction data
    pred_cols = [col for col in df.columns if 'prediction' in col.lower()]
    if pred_cols:
        print(f"\n   Prediction columns found: {len(pred_cols)}")

    # Compare with your model performance
    print("
🏆 COMPARISON WITH YOUR MODEL:"    # Load your elite domination results
    elite_file = 'elite_domination_results.json'
    if os.path.exists(elite_file):
        try:
            with open(elite_file, 'r') as f:
                elite_data = json.load(f)

            print("   Your Elite Domination Model:")
            if 'model_performance' in elite_data:
                perf = elite_data['model_performance']
                if 'mape' in perf:
                    print(".4f")
                if 'directional_accuracy' in perf:
                    print(".1%")
                if 'estimated_tao_per_prediction' in perf:
                    print(".6f")

        except Exception as e:
            print(f"   ❌ Error loading your model data: {e}")
    else:
        print(f"   ⚠️  Your model results not found: {elite_file}")

    # Load backtest results
    import glob
    backtest_files = glob.glob('backtest_results_*.json')
    if backtest_files:
        latest = max(backtest_files, key=os.path.getctime)
        try:
            with open(latest, 'r') as f:
                backtest_data = json.load(f)

            print(f"\n   Your Backtest Results ({os.path.basename(latest)}):")
            if 'performance' in backtest_data:
                perf = backtest_data['performance']
                if 'mape' in perf:
                    print(".4f")
                if 'directional_accuracy' in perf:
                    print(".1%")

        except Exception as e:
            print(f"   ❌ Error loading backtest data: {e}")

    print("
🎯 DOMINATION STRATEGY:"    print("   • Target: Surpass miner 31's average reward")
    print("   • Focus: Improve MAPE below 1.0% and directional accuracy above 75%")
    print("   • Method: Deploy elite_domination_model.pth to mainnet")
    print("   • Timeline: Achieve #1 position within 48 hours")

if __name__ == "__main__":
    analyze_miner31_vs_your_model()
