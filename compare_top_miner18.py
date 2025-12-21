#!/usr/bin/env python3
"""
Compare Your Model Performance with Top Miner #18

This script fetches performance data from WandB for the top miner #18
and compares it with your current robust model performance.
"""

import os
import json
import csv
from datetime import datetime, timezone
import sys

try:
    import wandb
    from wandb.apis.public import Api
except ImportError:
    print("❌ wandb package not found. Install with: pip install wandb")
    sys.exit(1)


def fetch_wandb_run_data(run_path, api_key=None):
    """Fetch run data from WandB"""
    try:
        # Set API key if provided
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key

        # Initialize API
        api = Api()

        print(f"📡 Fetching data for run: {run_path}")

        # Get the run
        run = api.run(run_path)

        print(f"✅ Connected to run: {run.name}")
        print(f"📊 Run state: {run.state}")
        print(f"🏃‍♂️ User: {run.user.username}")

        # Get history data
        print("📈 Fetching history data...")
        history = run.history()

        if history.empty:
            print("⚠️ No history data found")
            return None

        print(f"✅ Retrieved {len(history)} data points")

        # Get summary metrics
        summary = run.summary
        config = run.config

        return {
            'run_path': run_path,
            'run_name': run.name,
            'user': run.user.username,
            'state': run.state,
            'history': history,
            'summary': dict(summary),
            'config': dict(config)
        }

    except Exception as e:
        print(f"❌ Error fetching WandB data: {e}")
        return None


def analyze_wandb_data(wandb_data):
    """Analyze WandB performance data"""
    if not wandb_data:
        return None

    # Convert history to list of dicts if it's a pandas DataFrame
    history = wandb_data['history']
    if hasattr(history, 'to_dict'):  # pandas DataFrame
        history_list = history.to_dict('records')
    else:
        history_list = history

    if not history_list:
        return None

    analysis = {
        'total_predictions': len(history_list),
        'run_name': wandb_data['run_name'],
        'user': wandb_data['user']
    }

    # Get all available keys from first record
    if history_list:
        all_keys = set()
        for record in history_list:
            all_keys.update(record.keys())

        # Analyze key metrics
        metric_keys = [key for key in all_keys if any(keyword in key.lower()
                         for keyword in ['loss', 'accuracy', 'reward', 'score', 'mape', 'mae', 'rmse', 'tao', 'earning', 'profit'])]

        if metric_keys:
            analysis['available_metrics'] = list(metric_keys)

            for metric in metric_keys:
                values = []
                for record in history_list:
                    val = record.get(metric)
                    if val is not None and not (isinstance(val, float) and str(val).lower() in ['nan', 'inf', '-inf']):
                        try:
                            values.append(float(val))
                        except (ValueError, TypeError):
                            continue

                if values:
                    analysis[f'{metric}_mean'] = sum(values) / len(values)
                    analysis[f'{metric}_count'] = len(values)
                    analysis[f'{metric}_min'] = min(values)
                    analysis[f'{metric}_max'] = max(values)
                    analysis[f'{metric}_latest'] = values[-1] if values else None

                    # Calculate daily earnings for reward metrics
                    if any(keyword in metric.lower() for keyword in ['reward', 'tao', 'earning', 'profit']):
                        predictions_per_day = 6 * 24  # Conservative estimate
                        analysis[f'{metric}_daily_estimate'] = (sum(values) / len(values)) * predictions_per_day

    # Summary metrics
    summary = wandb_data.get('summary', {})
    if summary:
        analysis['summary_metrics'] = summary

    return analysis


def compare_with_your_model(wandb_analysis, your_model_data):
    """Compare WandB performance with your model"""

    if not wandb_analysis:
        return None

    comparison = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'comparison_type': 'your_model_vs_top_miner_18',
        'your_model': {
            'directional_accuracy': your_model_data.get('performance', {}).get('directional_accuracy', 0),
            'mape': your_model_data.get('performance', {}).get('mape', 0),
            'estimated_tao_per_prediction': your_model_data.get('competitiveness', {}).get('estimated_tao_per_prediction', 0),
            'daily_tao_estimate': your_model_data.get('competitiveness', {}).get('daily_tao_estimate', 0),
            'competitiveness_vs_miner221': your_model_data.get('competitiveness', {}).get('competitiveness_vs_miner221', 0)
        },
        'top_miner_18': wandb_analysis
    }

    # Calculate competitiveness ratios
    your_daily = comparison['your_model']['daily_tao_estimate']
    miner18_daily = comparison['top_miner_18'].get('reward_daily_estimate', 0)

    if miner18_daily > 0:
        comparison['competitiveness_analysis'] = {
            'your_vs_miner18_ratio': your_daily / miner18_daily if miner18_daily > 0 else 0,
            'gap_to_close': miner18_daily - your_daily,
            'improvement_needed_percent': ((miner18_daily - your_daily) / your_daily * 100) if your_daily > 0 else 0
        }

    return comparison


def main():
    """Main comparison function"""
    print("🏆 COMPARING YOUR MODEL WITH TOP MINER #18")
    print("=" * 50)

    # Load your model data
    your_model_file = 'robust_model_validation.json'
    if not os.path.exists(your_model_file):
        print(f"❌ Your model data file not found: {your_model_file}")
        return 1

    try:
        with open(your_model_file, 'r') as f:
            your_model_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading your model data: {e}")
        return 1

    print("✅ Loaded your model performance data")
    print(".1f")
    print(".6f")
    print(".1f")
    # WandB run paths
    wandb_runs = [
        "/yumaai/sn55-validators/runs/11t39nhr",
        "/yumaai/sn55-validators/runs/bemrn00a"
    ]

    # Check for WandB API key
    api_key = os.environ.get('WANDB_API_KEY')
    if not api_key:
        print("\\n⚠️ WandB API key not found in environment variables")
        print("💡 Set your API key with: export WANDB_API_KEY='your_key_here'")
        print("💡 Or enter it below:")

        api_key = input("Enter your WandB API key (or press Enter to skip): ").strip()
        if not api_key:
            print("❌ No API key provided. Cannot fetch WandB data.")
            return 1

    all_comparisons = []

    for run_path in wandb_runs:
        print(f"\\n🔍 ANALYZING RUN: {run_path}")
        print("-" * 40)

        # Fetch WandB data
        wandb_data = fetch_wandb_run_data(run_path, api_key)

        if not wandb_data:
            print(f"❌ Failed to fetch data for {run_path}")
            continue

        # Analyze the data
        analysis = analyze_wandb_data(wandb_data)

        if not analysis:
            print(f"❌ Failed to analyze data for {run_path}")
            continue

        # Print analysis results
        print("\\n📊 TOP MINER #18 PERFORMANCE ANALYSIS:")
        print(f"Run: {analysis['run_name']}")
        print(f"User: {analysis['user']}")
        print(f"Total predictions: {analysis['total_predictions']}")

        if 'available_metrics' in analysis:
            print(f"\\nAvailable metrics: {', '.join(analysis['available_metrics'])}")

        # Show key metrics
        for key, value in analysis.items():
            if key.endswith('_mean') or key.endswith('_latest') or key.endswith('_daily_estimate'):
                if isinstance(value, (int, float)):
                    print(".6f")
        # Compare with your model
        comparison = compare_with_your_model(analysis, your_model_data)

        if comparison:
            all_comparisons.append(comparison)

            comp_analysis = comparison.get('competitiveness_analysis', {})

            if comp_analysis:
                print("\\n🏁 COMPETITIVENESS ANALYSIS:")
                print(".3f")
                print(".1f")
                print(".1f")
    # Save comprehensive comparison
    if all_comparisons:
        comparison_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'your_model_file': your_model_file,
            'comparisons': all_comparisons,
            'summary': {
                'runs_analyzed': len(all_comparisons),
                'your_current_performance': {
                    'directional_accuracy': your_model_data.get('performance', {}).get('directional_accuracy', 0),
                    'daily_tao_estimate': your_model_data.get('competitiveness', {}).get('daily_tao_estimate', 0)
                }
            }
        }

        output_file = 'top_miner18_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)

        print(f"\\n✅ Comparison results saved to: {output_file}")

        # Print final assessment
        print("\\n🎯 FINAL ASSESSMENT:")

        total_gap = 0
        valid_comparisons = 0

        for comp in all_comparisons:
            comp_analysis = comp.get('competitiveness_analysis', {})
            if comp_analysis and 'gap_to_close' in comp_analysis:
                total_gap += comp_analysis['gap_to_close']
                valid_comparisons += 1

        if valid_comparisons > 0:
            avg_gap = total_gap / valid_comparisons
            print(".1f")
            if avg_gap > 0:
                print("💪 Keep improving - you're on the right track!")
            else:
                print("🎉 You're already competitive with the top miner!")

    else:
        print("\\n❌ No successful comparisons completed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
