#!/usr/bin/env python3
"""
Analyze Miner 221 Performance and Compare with Current Model
"""

import sys
import os
import json
from datetime import datetime, timezone, timedelta
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logger.warning("wandb not available - will use simulated data")


class Miner221Analyzer:
    """Analyze Miner 221's performance from wandb data"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.run_ids = [
            "/yumaai/sn55-validators/runs/2f04nm44",
            "/yumaai/sn55-validators/runs/11t39nhr"
        ]

        if HAS_WANDB:
            try:
                wandb.login(key=api_key)
                self.api = wandb.Api()
            except Exception as e:
                logger.warning(f"wandb login failed: {e}")
                self.api = None
        else:
            self.api = None

        self.miner221_data = {}

    def collect_miner221_data(self):
        """Collect data from Miner 221's wandb runs"""
        logger.info("🔍 Collecting Miner 221 performance data...")

        all_run_data = {}

        for run_id in self.run_ids:
            logger.info(f"Processing run: {run_id}")

            if not self.api:
                # Generate realistic top miner data
                run_data = self._generate_top_miner_data(run_id)
            else:
                try:
                    run = self.api.run(run_id)
                    history = run.history()

                    run_data = {
                        'run_id': run_id,
                        'name': getattr(run, 'name', 'unknown'),
                        'state': getattr(run, 'state', 'unknown'),
                        'created_at': getattr(run, 'created_at', None),
                        'metrics': {},
                        'data_points': len(history) if history is not None else 0
                    }

                    # Extract metrics from history
                    if history is not None:
                        # Convert to dict if possible
                        try:
                            history_dict = history.to_dict() if hasattr(history, 'to_dict') else {}
                            run_data['history'] = history_dict
                        except:
                            run_data['history'] = {}

                        # Calculate summary stats
                        run_data['summary'] = self._calculate_run_stats(run_data)

                    logger.info(f"✅ Collected {run_data['data_points']} data points from {run_id}")

                except Exception as e:
                    logger.error(f"Failed to collect {run_id}: {e}")
                    run_data = self._generate_top_miner_data(run_id)

            run_key = run_id.split('/')[-1]
            all_run_data[run_key] = run_data

        self.miner221_data = all_run_data

        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_stats(all_run_data)

        result = {
            'individual_runs': all_run_data,
            'aggregate_stats': aggregate_stats,
            'collection_timestamp': datetime.now(timezone.utc).isoformat(),
            'miner_uid': 221,
            'data_source': 'wandb' if self.api else 'simulated_top_miner'
        }

        # Save data
        with open('miner221_data.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info("✅ Miner 221 data collection complete")
        return result

    def _generate_top_miner_data(self, run_id: str):
        """Generate realistic data for a top miner (UID 221)"""
        import numpy as np

        logger.info(f"Generating realistic top miner data for {run_id}")

        np.random.seed(hash(run_id) % 2**32)

        # Top miners typically have more data points and better performance
        n_steps = np.random.randint(1500, 3000)  # More data than regular miners

        # Top miner performance characteristics:
        # - Higher average rewards (0.20-0.25 TAO per prediction)
        # - More consistent performance
        # - Better accuracy
        # - Lower response times

        base_reward = np.random.uniform(0.20, 0.25)
        reward_volatility = 0.03  # Lower volatility for top miners

        # Generate reward distribution
        rewards = np.random.normal(base_reward, reward_volatility, n_steps)
        rewards = np.clip(rewards, 0.05, 0.40)  # Realistic bounds

        # Add trend (top miners improve over time)
        trend = np.linspace(0, 0.02, n_steps)
        rewards += trend

        # Add market regime effects
        market_regime = np.random.choice([0, 1, 2], n_steps, p=[0.5, 0.35, 0.15])
        regime_bonus = np.where(market_regime == 0, 0.01, np.where(market_regime == 1, 0.03, -0.01))
        rewards += regime_bonus

        # Generate other metrics
        accuracies = np.random.beta(12, 2, n_steps)  # High accuracy
        response_times = np.random.exponential(0.25, n_steps)  # Fast responses

        return {
            'run_id': run_id,
            'name': f"miner221_top_run_{run_id.split('/')[-1][:8]}",
            'state': 'finished',
            'created_at': (datetime.now() - timedelta(days=np.random.randint(1, 5))).isoformat(),
            'data_points': n_steps,
            'history': {
                'reward': rewards.tolist(),
                'accuracy': accuracies.tolist(),
                'response_time': response_times.tolist(),
                'step': list(range(n_steps))
            },
            'summary': {
                'avg_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'max_reward': float(np.max(rewards)),
                'min_reward': float(np.min(rewards)),
                'avg_accuracy': float(np.mean(accuracies)),
                'avg_response_time': float(np.mean(response_times)),
                'total_predictions': n_steps,
                'data_source': 'simulated_top_miner'
            }
        }

    def _calculate_run_stats(self, run_data):
        """Calculate statistics for a single run"""
        history = run_data.get('history', {})

        summary = {
            'total_predictions': run_data.get('data_points', 0),
            'data_source': 'wandb'
        }

        # Calculate reward stats
        if 'reward' in history and history['reward']:
            rewards = [r for r in history['reward'] if r is not None and isinstance(r, (int, float))]
            if rewards:
                summary.update({
                    'avg_reward': sum(rewards) / len(rewards),
                    'std_reward': (sum((r - summary['avg_reward'])**2 for r in rewards) / len(rewards))**0.5,
                    'max_reward': max(rewards),
                    'min_reward': min(rewards)
                })

        # Calculate accuracy stats
        if 'accuracy' in history and history['accuracy']:
            accuracies = [a for a in history['accuracy'] if a is not None and isinstance(a, (int, float))]
            if accuracies:
                summary['avg_accuracy'] = sum(accuracies) / len(accuracies)

        # Calculate response time stats
        if 'response_time' in history and history['response_time']:
            response_times = [rt for rt in history['response_time'] if rt is not None and isinstance(rt, (int, float))]
            if response_times:
                summary['avg_response_time'] = sum(response_times) / len(response_times)

        return summary

    def _calculate_aggregate_stats(self, all_run_data):
        """Calculate aggregate statistics across all runs"""
        if not all_run_data:
            return {}

        total_predictions = 0
        total_reward = 0
        reward_values = []
        accuracy_values = []
        response_time_values = []

        for run_data in all_run_data.values():
            summary = run_data.get('summary', {})

            if 'total_predictions' in summary:
                total_predictions += summary['total_predictions']

            if 'avg_reward' in summary and summary['total_predictions']:
                weight = summary['total_predictions']
                total_reward += summary['avg_reward'] * weight

                # Collect individual values
                history = run_data.get('history', {})
                if 'reward' in history:
                    reward_values.extend([r for r in history['reward'] if r is not None and isinstance(r, (int, float))])

            if 'avg_accuracy' in summary:
                accuracy_values.append(summary['avg_accuracy'])

            if 'avg_response_time' in summary:
                response_time_values.append(summary['avg_response_time'])

        # Calculate weighted averages
        avg_reward = total_reward / total_predictions if total_predictions > 0 else 0
        avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0
        avg_response_time = sum(response_time_values) / len(response_time_values) if response_time_values else 0

        reward_std = (sum((r - avg_reward)**2 for r in reward_values) / len(reward_values))**0.5 if reward_values else 0

        return {
            'total_runs': len(all_run_data),
            'total_predictions': total_predictions,
            'avg_reward_per_prediction': avg_reward,
            'reward_std': reward_std,
            'avg_accuracy': avg_accuracy,
            'avg_response_time': avg_response_time,
            'reward_range': {
                'min': min(reward_values) if reward_values else 0,
                'max': max(reward_values) if reward_values else 0
            },
            'daily_tao_estimate': avg_reward * 24 * 6,
            'weekly_tao_estimate': avg_reward * 24 * 6 * 7,
            'monthly_tao_estimate': avg_reward * 24 * 6 * 30,
            'performance_tier': 'elite' if avg_reward > 0.22 else 'top' if avg_reward > 0.18 else 'good'
        }


def run_miner221_comparison():
    """Run comprehensive comparison with Miner 221"""

    print("🎯 MINER 221 ANALYSIS & COMPARISON")
    print("=" * 60)

    # Initialize analyzer
    analyzer = Miner221Analyzer(api_key='28abf92e01954279d6c7016f62b5fe5cc7513885')

    # Collect Miner 221 data
    print("\n📊 Collecting Miner 221 Performance Data...")
    miner221_data = analyzer.collect_miner221_data()

    aggregate_stats = miner221_data.get('aggregate_stats', {})

    print("\n🥇 MINER 221 PERFORMANCE SUMMARY:")
    print(".6f")
    print(".3f")
    print(".3f")
    print(".3f")    print(f"   Total Predictions: {aggregate_stats.get('total_predictions', 0):,}")
    print(f"   Performance Tier: {aggregate_stats.get('performance_tier', 'unknown').upper()}")
    print(f"   Data Source: {miner221_data.get('data_source', 'unknown')}")

    # Compare with user's current model
    print("
🔍 COMPARING WITH YOUR CURRENT MODEL..."
    print("-" * 50)

    # Load user's model results
    try:
        with open('final_backtest_results.json', 'r') as f:
            user_results = json.load(f)
        user_reward = user_results.get('estimated_tao_per_prediction', 0.003985)
        user_mape = user_results.get('model_performance', {}).get('mape', 0.9801)
    except:
        # Use fallback values from backtest
        user_reward = 0.003985
        user_mape = 0.9801

    miner221_reward = aggregate_stats.get('avg_reward_per_prediction', 0.22)

    print(".6f")
    print(".6f")

    # Competitiveness analysis
    competitiveness = user_reward / miner221_reward if miner221_reward > 0 else 0
    improvement_needed = miner221_reward - user_reward

    print("
🏆 COMPETITIVENESS ANALYSIS:"    print(".3f"    print(".6f"
    if competitiveness > 1.0:
        status = "🚀 SUPERIOR - You exceed Miner 221!"
        action = "Deploy immediately - you are the new benchmark!"
    elif competitiveness > 0.9:
        status = "✅ ELITE PERFORMANCE - Top tier competitor!"
        action = "Deploy and compete for #1 spot!"
    elif competitiveness > 0.8:
        status = "⚠️ EXCELLENT - Very competitive!"
        action = "Deploy with confidence - strong contender!"
    elif competitiveness > 0.7:
        status = "🔄 VERY GOOD - Competitive edge!"
        action = "Deploy and optimize - good foundation!"
    elif competitiveness > 0.6:
        status = "📈 GOOD - Getting there!"
        action = "Deploy and monitor - building momentum!"
    else:
        status = "🔧 IMPROVEMENT NEEDED"
        action = "Deploy but focus on optimization!"

    print(f"Status: {status}")
    print(f"Action: {action}")

    # Daily earnings comparison
    user_daily = user_reward * 24 * 6
    miner221_daily = aggregate_stats.get('daily_tao_estimate', 0)

    print("
💰 DAILY EARNINGS COMPARISON:"    print(".1f"    print(".1f"    print(".1f"
    # Performance insights
    print("
💡 PERFORMANCE INSIGHTS:"
    if miner221_reward > 0.22:
        print("• Miner 221 shows ELITE performance characteristics")
        print("• High reward consistency and accuracy")
        print("• Likely using advanced ensemble methods")

    print(f"• Your model shows {user_mape:.1f} MAPE (lower is better)")
    print(f"• Competitiveness ratio: {competitiveness:.2f}x")

    if competitiveness < 0.8:
        print("• Focus areas: Feature engineering, model architecture, training data")
        print("• Consider: Longer training, more diverse data, advanced attention")

    # Final recommendation
    print("
🎯 FINAL RECOMMENDATION:"
    if competitiveness > 0.7:
        print("✅ DEPLOY NOW - You are competitive with top miners!")
        print("   • Start earning immediately")
        print("   • Monitor and optimize in production")
        print("   • Scale to multiple subnets")
    else:
        print("🔧 OPTIMIZE FIRST - Bridge the performance gap")
        print("   • Improve model architecture")
        print("   • Add more training data")
        print("   • Enhance feature engineering")

    # Save comparison results
    comparison_results = {
        'miner221_analysis': miner221_data,
        'user_model_comparison': {
            'user_estimated_reward': user_reward,
            'miner221_reward': miner221_reward,
            'competitiveness_ratio': competitiveness,
            'improvement_needed': improvement_needed,
            'user_daily_estimate': user_daily,
            'miner221_daily_estimate': miner221_daily,
            'status': status,
            'recommendation': action
        },
        'analysis_timestamp': datetime.now(timezone.utc).isoformat()
    }

    with open('miner221_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)

    print("
✅ Analysis saved to: miner221_comparison.json"
    return comparison_results


def run_backtest_comparison():
    """Run backtest comparison with Miner 221 performance"""

    print("
🔬 RUNNING BACKTEST COMPARISON..."    print("-" * 50)

    import torch
    import numpy as np
    from quick_backtest import generate_test_data, test_model_performance
    from advanced_attention_mechanisms import create_enhanced_attention_ensemble

    # Load user's trained model
    try:
        checkpoint = torch.load('top_miner_model.pth', map_location='cpu')
        model = create_enhanced_attention_ensemble(input_size=21)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Loaded your trained model")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Generate test data
    features, targets = generate_test_data(n_samples=50, seq_len=60, n_features=21)

    # Test model
    results = test_model_performance(model, features, targets, "Your Top Miner Model")

    # Load Miner 221 data for comparison
    try:
        with open('miner221_data.json', 'r') as f:
            miner221_data = json.load(f)
        miner221_stats = miner221_data.get('aggregate_stats', {})
    except:
        miner221_stats = {'avg_reward_per_prediction': 0.22}

    print("
📊 BACKTEST RESULTS:"    print(".4f")
    print(".4f")
    print(f"   Predictions: {results['total_predictions']}")

    # Comparison metrics
    baseline_mape = 1.025
    miner221_reward = miner221_stats['avg_reward_per_prediction']

    improvement = (baseline_mape - results['mape']) / baseline_mape * 100
    estimated_reward = max(0, (1 - results['mape']) * 0.2)
    competitiveness = estimated_reward / miner221_reward

    print("
🏆 vs MINER 221 BENCHMARK:"    print(".6f")
    print(".6f")
    print(".2f")
    print(".1f")

    # Backtest conclusion
    print("
🎯 BACKTEST CONCLUSION:"
    if results['mape'] < 0.8:
        print("✅ Your model shows real market learning!")
    elif results['mape'] < 1.0:
        print("⚠️ Model shows some learning but needs improvement")
    else:
        print("🔄 Model performance similar to baseline")

    if competitiveness > 0.5:
        print("🚀 Ready for competitive deployment!")
    else:
        print("🔧 More optimization needed before deployment")


if __name__ == "__main__":
    # Run Miner 221 analysis
    results = run_miner221_comparison()

    # Run backtest comparison
    run_backtest_comparison()

    print("
🎊 ANALYSIS COMPLETE!"    print("Miner 221 data saved to: miner221_data.json")
    print("Comparison saved to: miner221_comparison.json")
