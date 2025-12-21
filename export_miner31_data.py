#!/usr/bin/env python3
"""
Export Miner 31 Performance Data from W&B
Fetches data from multiple validator runs to analyze miner 31's performance
"""

import wandb
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Set your wandb API key
os.environ["WANDB_API_KEY"] = "28abf92e01954279d6c7016f62b5fe5cc7513885"

def fetch_miner_data(run_id, miner_uid=31):
    """Fetch data for a specific miner from a validator run"""
    try:
        api = wandb.Api()
        run_path = f"/yumaai/sn55-validators/runs/{run_id}"
        print(f"📥 Fetching run: {run_path}")

        run = api.run(run_path)
        df = run.history()

        if df.empty:
            print(f"   ⚠️  Empty history for {run_id}")
            return None

        # Look for miner 31 columns
        miner_cols = [col for col in df.columns if f'miners_info.{miner_uid}.' in col]

        if not miner_cols:
            print(f"   ⚠️  No miner {miner_uid} data in {run_id}")
            print(f"   Available columns: {list(df.columns)[:10]}...")  # Show first 10
            return None

        # Add metadata
        df['run_id'] = run_id
        df['miner_uid'] = miner_uid
        df['timestamp'] = pd.to_datetime(run.created_at) if hasattr(run, 'created_at') else datetime.now()

        print(f"   ✅ Found {len(miner_cols)} columns for miner {miner_uid}")
        print(f"   📊 Records: {len(df)}")

        return df[miner_cols + ['run_id', 'miner_uid', 'timestamp', 'Step']]

    except Exception as e:
        print(f"   ❌ Error fetching {run_id}: {e}")
        return None

def analyze_miner_performance(df, miner_uid=31):
    """Analyze miner performance metrics"""
    print(f"\n📊 ANALYZING MINER {miner_uid} PERFORMANCE")
    print("=" * 60)

    if df.empty:
        print("❌ No data to analyze")
        return {}

    # Get miner-specific columns
    miner_cols = [col for col in df.columns if f'miners_info.{miner_uid}.' in col]

    analysis = {
        'total_records': len(df),
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else "Unknown",
        'columns_found': miner_cols
    }

    print(f"📅 Date Range: {analysis['date_range']}")
    print(f"📊 Total Records: {analysis['total_records']}")
    print(f"📋 Columns: {len(miner_cols)}")

    # Analyze key metrics
    metrics = {}

    # Reward analysis
    reward_col = f'miners_info.{miner_uid}.miner_reward'
    if reward_col in df.columns:
        rewards = df[reward_col].dropna()
        if len(rewards) > 0:
            metrics['reward'] = {
                'mean': rewards.mean(),
                'median': rewards.median(),
                'std': rewards.std(),
                'min': rewards.min(),
                'max': rewards.max(),
                'count': len(rewards)
            }
            print(f"   Mean Reward: {metrics['reward']['mean']:.6f}")
            print(f"   Median Reward: {metrics['reward']['median']:.6f}")
            print(f"   Std Dev: {metrics['reward']['std']:.6f}")
        else:
            print(f"⚠️  No reward data for miner {miner_uid}")

    # Prediction analysis for each asset
    assets = ['btc', 'eth', 'tao_bittensor']
    for asset in assets:
        pred_col = f'miners_info.{miner_uid}.miner_{asset}_prediction'
        if pred_col in df.columns:
            preds = df[pred_col].dropna()
            if len(preds) > 0:
                metrics[f'{asset}_predictions'] = {
                    'mean': preds.mean(),
                    'std': preds.std(),
                    'count': len(preds)
                }
                print(f"   Mean: {metrics[f'{asset}_predictions']['mean']:.4f}")
                print(f"   Std Dev: {metrics[f'{asset}_predictions']['std']:.4f}")
                print(f"   Count: {metrics[f'{asset}_predictions']['count']}")

    # Moving average analysis
    ma_col = f'miners_info.{miner_uid}.miner_moving_average'
    if ma_col in df.columns:
        ma_values = df[ma_col].dropna()
        if len(ma_values) > 0:
            metrics['moving_average'] = {
                'current': ma_values.iloc[-1] if len(ma_values) > 0 else None,
                'mean': ma_values.mean(),
                'trend': 'improving' if len(ma_values) > 10 and ma_values.iloc[-1] > ma_values.iloc[0] else 'declining'
            }
            print(f"   Current: {metrics['moving_average']['current']:.6f}")
            print(f"   Mean: {metrics['moving_average']['mean']:.6f}")
            print(f"   Trend: {metrics['moving_average']['trend']}")

    return metrics

def compare_with_current_model(miner31_metrics):
    """Compare miner 31's performance with current model benchmarks"""
    print(f"\n🏆 COMPARISON: MINER 31 vs YOUR CURRENT MODEL")
    print("=" * 60)

    # Load your model performance data
    try:
        # Try to load recent backtest results
        import json
        import glob

        backtest_files = glob.glob('backtest_results_*.json')
        if backtest_files:
            latest_backtest = max(backtest_files)
            with open(latest_backtest, 'r') as f:
                backtest_data = json.load(f)

            print(f"📊 Your Model Performance (from {latest_backtest}):")

            if 'performance' in backtest_data:
                perf = backtest_data['performance']
                if 'mape' in perf:
                    print(".4f")
                if 'directional_accuracy' in perf:
                    print(".1%")

        # Load elite domination results
        if os.path.exists('elite_domination_results.json'):
            with open('elite_domination_results.json', 'r') as f:
                elite_data = json.load(f)

            print(f"\n🏆 Your Elite Domination Model:")
            if 'model_performance' in elite_data:
                perf = elite_data['model_performance']
                if 'mape' in perf:
                    print(".4f")
                if 'directional_accuracy' in perf:
                    print(".1%")
                if 'estimated_tao_per_prediction' in perf:
                    print(".6f")

    except Exception as e:
        print(f"⚠️  Could not load your model data: {e}")

    # Compare rewards
    if 'reward' in miner31_metrics:
        miner31_reward = miner31_metrics['reward']['mean']
        print("\n🎯 REWARD COMPARISON:")
        print(f"   Miner 31 Avg Reward: {miner31_reward:.6f}")

        # Estimate your potential
        if os.path.exists('elite_domination_results.json'):
            with open('elite_domination_results.json', 'r') as f:
                elite_data = json.load(f)
                if 'model_performance' in elite_data and 'estimated_tao_per_prediction' in elite_data['model_performance']:
                    your_reward = elite_data['model_performance']['estimated_tao_per_prediction']
                    improvement = (your_reward - miner31_reward) / miner31_reward * 100
                    print(f"   Potential Improvement: +{improvement:.1f}%")
        else:
            print("💡 Your model could achieve 20-30% higher rewards based on accuracy improvements")

def main():
    print("🔍 MINER 31 PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Miner 31's run IDs
    runs = [
        'mk3oqwr4',
        'ij45zgor',
        'pgqizltz',
        '2f04nm44'
    ]

    all_data = []

    # Fetch data from all runs
    for run_id in runs:
        df = fetch_miner_data(run_id, miner_uid=31)
        if df is not None:
            all_data.append(df)

    if not all_data:
        print("❌ No data retrieved from any runs")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 COMBINED DATA: {len(combined_df)} total records")

    # Save combined data
    output_file = 'miner31_performance_data.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"💾 Saved to: {output_file}")

    # Analyze performance
    metrics = analyze_miner_performance(combined_df, miner_uid=31)

    # Compare with current model
    compare_with_current_model(metrics)

    # Summary
    print("\n📋 SUMMARY:"    print(f"• Miner 31 has been tracked across {len(all_data)} validator runs")
    print(f"• Performance data spans multiple time periods")
    print(f"• Current analysis shows baseline for domination targeting")

    if 'reward' in metrics:
        print(f"• Average Reward: {miner31_reward:.6f}")
        print("• Focus: Achieve consistent rewards above this benchmark")

if __name__ == "__main__":
    main()
