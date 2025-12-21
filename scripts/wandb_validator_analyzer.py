"""
Wandb Validator Data Collector and Deep Performance Analyzer
Collects live data from all 31 Precog subnet 55 validators and analyzes performance
"""

import wandb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import asyncio
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WandbValidatorAnalyzer:
    """Comprehensive analyzer for Precog subnet 55 validator performance"""

    def __init__(self, api_key: str = "28abf92e01954279d6c7016f62b5fe5cc7513885"):
        self.api_key = api_key
        self.project_path = "yumaai/sn55-validators"
        self.validators_data = {}
        self.live_metrics = {}

        # Known validator run IDs from user examples
        self.known_runs = [
            "11t39nhr", "bemrn00a", "srg3633h", "2f04nm44", "t86xd09z"
        ]

        # Authenticate with wandb
        wandb.login(key=self.api_key)
        self.api = wandb.Api()

    def discover_all_validators(self) -> List[str]:
        """Discover all validator runs in the project"""
        logger.info("🔍 Discovering all validator runs...")

        try:
            # Get all runs from the project
            runs = self.api.runs(f"{self.project_path}")

            validator_runs = []
            for run in runs:
                # Filter for validator runs (exclude miners)
                if "validator" in run.name.lower() or run.config.get("role") == "validator":
                    validator_runs.append(run.id)

            logger.info(f"✅ Found {len(validator_runs)} validator runs")
            return validator_runs

        except Exception as e:
            logger.error(f"❌ Error discovering validators: {e}")
            # Fallback to known runs
            return self.known_runs

    def collect_validator_data(self, run_ids: List[str], max_runs: int = 31) -> Dict[str, pd.DataFrame]:
        """Collect performance data from validator runs"""
        logger.info(f"📊 Collecting data from {min(len(run_ids), max_runs)} validators...")

        validator_data = {}

        for i, run_id in enumerate(run_ids[:max_runs]):
            try:
                logger.info(f"  Collecting data from validator {i+1}/{min(len(run_ids), max_runs)} (ID: {run_id})")

                run_path = f"{self.project_path}/runs/{run_id}"
                run = self.api.run(run_path)

                # Get history data (last 1000 points to avoid rate limits)
                history = run.history(samples=1000)

                if not history.empty:
                    # Add metadata
                    history['validator_id'] = run_id
                    history['run_name'] = run.name
                    history['timestamp'] = pd.to_datetime(history.get('_timestamp', history.index), unit='s')

                    validator_data[run_id] = history
                    logger.info(f"    ✅ Collected {len(history)} data points from {run_id}")

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"    ❌ Failed to collect data from {run_id}: {e}")
                continue

        logger.info(f"✅ Successfully collected data from {len(validator_data)} validators")
        return validator_data

    def extract_live_metrics(self, validator_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Extract current/live performance metrics from each validator"""
        logger.info("🔄 Extracting live performance metrics...")

        live_metrics = {}

        for validator_id, data in validator_data.items():
            try:
                # Get most recent data point
                latest_data = data.iloc[-1] if not data.empty else None

                if latest_data is not None:
                    # Extract key metrics
                    metrics = {
                        'validator_id': validator_id,
                        'timestamp': latest_data.get('timestamp'),
                        'run_name': latest_data.get('run_name'),

                        # Performance metrics
                        'total_predictions': latest_data.get('total_predictions', 0),
                        'correct_predictions': latest_data.get('correct_predictions', 0),
                        'accuracy': latest_data.get('accuracy', 0),

                        # Reward metrics
                        'total_rewards': latest_data.get('total_rewards', 0),
                        'avg_reward_per_prediction': latest_data.get('avg_reward_per_prediction', 0),

                        # Timing metrics
                        'avg_response_time': latest_data.get('avg_response_time', 0),
                        'uptime_percentage': latest_data.get('uptime_percentage', 0),

                        # Network metrics
                        'active_miners': latest_data.get('active_miners', 0),
                        'total_stake': latest_data.get('total_stake', 0),

                        # Scoring metrics
                        'validator_score': latest_data.get('validator_score', 0),
                        'miner_scores_mean': latest_data.get('miner_scores_mean', 0),
                        'miner_scores_std': latest_data.get('miner_scores_std', 0),

                        # Competition metrics
                        'rank': latest_data.get('rank', 0),
                        'percentile': latest_data.get('percentile', 0),
                    }

                    live_metrics[validator_id] = metrics

            except Exception as e:
                logger.warning(f"❌ Error extracting metrics from {validator_id}: {e}")
                continue

        logger.info(f"✅ Extracted live metrics from {len(live_metrics)} validators")
        return live_metrics

    def analyze_validator_performance(self, live_metrics: Dict[str, Dict]) -> Dict:
        """Deep analysis of validator performance patterns"""
        logger.info("🔬 Performing deep performance analysis...")

        if not live_metrics:
            return {}

        # Convert to DataFrame for analysis
        df = pd.DataFrame.from_dict(live_metrics, orient='index')

        analysis = {
            'overview': {
                'total_validators': len(df),
                'active_validators': len(df[df['total_predictions'] > 0]),
                'avg_predictions_per_validator': df['total_predictions'].mean(),
                'total_predictions_network': df['total_predictions'].sum()
            },

            'performance_distribution': {
                'accuracy': {
                    'mean': df['accuracy'].mean(),
                    'std': df['accuracy'].std(),
                    'min': df['accuracy'].min(),
                    'max': df['accuracy'].max(),
                    'top_10_percent': df['accuracy'].quantile(0.9),
                    'bottom_10_percent': df['accuracy'].quantile(0.1)
                },
                'avg_reward': {
                    'mean': df['avg_reward_per_prediction'].mean(),
                    'std': df['avg_reward_per_prediction'].std(),
                    'min': df['avg_reward_per_prediction'].min(),
                    'max': df['avg_reward_per_prediction'].max(),
                    'top_10_percent': df['avg_reward_per_prediction'].quantile(0.9),
                    'bottom_10_percent': df['avg_reward_per_prediction'].quantile(0.1)
                }
            },

            'response_time_analysis': {
                'mean_response_time': df['avg_response_time'].mean(),
                'fastest_validator': df.loc[df['avg_response_time'].idxmin()]['validator_id'] if df['avg_response_time'].min() > 0 else None,
                'slowest_validator': df.loc[df['avg_response_time'].idxmax()]['validator_id'] if df['avg_response_time'].max() > 0 else None,
                'sub_1_second_count': len(df[df['avg_response_time'] < 1.0])
            },

            'uptime_analysis': {
                'mean_uptime': df['uptime_percentage'].mean(),
                'high_uptime_count': len(df[df['uptime_percentage'] > 95]),
                'low_uptime_count': len(df[df['uptime_percentage'] < 90])
            },

            'top_performers': {
                'by_accuracy': df.nlargest(5, 'accuracy')[['validator_id', 'accuracy', 'avg_reward_per_prediction']].to_dict('records'),
                'by_reward': df.nlargest(5, 'avg_reward_per_prediction')[['validator_id', 'accuracy', 'avg_reward_per_prediction']].to_dict('records'),
                'by_speed': df.nsmallest(5, 'avg_response_time')[['validator_id', 'avg_response_time', 'accuracy']].to_dict('records'),
                'by_uptime': df.nlargest(5, 'uptime_percentage')[['validator_id', 'uptime_percentage', 'accuracy']].to_dict('records')
            },

            'correlations': self.calculate_correlations(df),

            'market_insights': self.extract_market_insights(df)
        }

        return analysis

    def calculate_correlations(self, df: pd.DataFrame) -> Dict:
        """Calculate correlations between different performance metrics"""
        try:
            numeric_cols = ['accuracy', 'avg_reward_per_prediction', 'avg_response_time',
                          'uptime_percentage', 'total_predictions', 'validator_score']

            corr_matrix = df[numeric_cols].corr()

            correlations = {
                'accuracy_vs_reward': corr_matrix.loc['accuracy', 'avg_reward_per_prediction'],
                'speed_vs_reward': corr_matrix.loc['avg_response_time', 'avg_reward_per_prediction'],
                'uptime_vs_reward': corr_matrix.loc['uptime_percentage', 'avg_reward_per_prediction'],
                'accuracy_vs_speed': corr_matrix.loc['accuracy', 'avg_response_time'],
                'volume_vs_accuracy': corr_matrix.loc['total_predictions', 'accuracy']
            }

            return correlations

        except Exception as e:
            logger.warning(f"❌ Error calculating correlations: {e}")
            return {}

    def extract_market_insights(self, df: pd.DataFrame) -> Dict:
        """Extract market insights from validator performance"""
        insights = {}

        try:
            # Performance thresholds
            high_accuracy_threshold = df['accuracy'].quantile(0.8)
            high_reward_threshold = df['avg_reward_per_prediction'].quantile(0.8)
            fast_response_threshold = df['avg_response_time'].quantile(0.2)  # Top 20% fastest

            # Market segments
            insights['performance_segments'] = {
                'elite': len(df[(df['accuracy'] >= high_accuracy_threshold) &
                               (df['avg_reward_per_prediction'] >= high_reward_threshold)]),
                'fast_and_accurate': len(df[(df['accuracy'] >= high_accuracy_threshold) &
                                          (df['avg_response_time'] <= fast_response_threshold)]),
                'high_volume': len(df[df['total_predictions'] > df['total_predictions'].quantile(0.8)])
            }

            # Success patterns
            insights['success_patterns'] = {
                'accuracy_importance': 'High' if df['accuracy'].corr(df['avg_reward_per_prediction']) > 0.5 else 'Medium',
                'speed_importance': 'High' if abs(df['avg_response_time'].corr(df['avg_reward_per_prediction'])) > 0.3 else 'Low',
                'volume_importance': 'High' if df['total_predictions'].corr(df['avg_reward_per_prediction']) > 0.4 else 'Medium'
            }

            # Competitive landscape
            insights['competitive_landscape'] = {
                'total_market_size': df['total_predictions'].sum(),
                'avg_market_accuracy': df['accuracy'].mean(),
                'avg_market_reward': df['avg_reward_per_prediction'].mean(),
                'performance_dispersion': df['accuracy'].std() / df['accuracy'].mean()  # Coefficient of variation
            }

        except Exception as e:
            logger.warning(f"❌ Error extracting market insights: {e}")

        return insights

    def compare_with_user_model(self, user_metrics: Dict, live_metrics: Dict[str, Dict]) -> Dict:
        """Compare user's model performance with validator network"""
        logger.info("🔍 Comparing your model with validator network...")

        if not live_metrics:
            return {}

        # Convert validator metrics to DataFrame
        df = pd.DataFrame.from_dict(live_metrics, orient='index')

        comparison = {
            'user_vs_network': {},
            'percentile_rankings': {},
            'competitive_position': {},
            'improvement_opportunities': []
        }

        # Compare key metrics
        metrics_to_compare = {
            'accuracy': 'accuracy',
            'avg_reward': 'avg_reward_per_prediction',
            'response_time': 'avg_response_time',
            'uptime': 'uptime_percentage'
        }

        for user_metric, network_metric in metrics_to_compare.items():
            if user_metric in user_metrics and network_metric in df.columns:
                user_value = user_metrics[user_metric]
                network_values = df[network_metric].dropna()

                if len(network_values) > 0:
                    percentile = (network_values < user_value).mean() * 100

                    comparison['user_vs_network'][user_metric] = {
                        'user_value': user_value,
                        'network_mean': network_values.mean(),
                        'network_median': network_values.median(),
                        'network_std': network_values.std(),
                        'percentile_rank': percentile,
                        'vs_top_10': percentile >= 90,
                        'vs_bottom_10': percentile <= 10
                    }

                    comparison['percentile_rankings'][user_metric] = percentile

        # Determine competitive position
        avg_percentile = np.mean(list(comparison['percentile_rankings'].values()))

        if avg_percentile >= 90:
            position = "🏆 TOP 10% - Elite Performer"
        elif avg_percentile >= 75:
            position = "✅ TOP 25% - Strong Competitor"
        elif avg_percentile >= 50:
            position = "📊 TOP 50% - Above Average"
        elif avg_percentile >= 25:
            position = "⚠️ BOTTOM 50% - Needs Improvement"
        else:
            position = "❌ BOTTOM 25% - Significant Improvement Needed"

        comparison['competitive_position'] = {
            'overall_percentile': avg_percentile,
            'position_category': position,
            'metrics_above_median': sum(1 for p in comparison['percentile_rankings'].values() if p >= 50),
            'total_metrics_compared': len(comparison['percentile_rankings'])
        }

        # Generate improvement opportunities
        comparison['improvement_opportunities'] = self.generate_improvement_suggestions(
            comparison, user_metrics, df
        )

        return comparison

    def generate_improvement_suggestions(self, comparison: Dict, user_metrics: Dict, df: pd.DataFrame) -> List[str]:
        """Generate specific improvement suggestions based on comparison"""
        suggestions = []

        percentile_rankings = comparison.get('percentile_rankings', {})

        # Accuracy suggestions
        if 'accuracy' in percentile_rankings:
            acc_percentile = percentile_rankings['accuracy']
            if acc_percentile < 50:
                suggestions.append(f"🎯 Improve accuracy (currently {user_metrics.get('accuracy', 0):.1%} - "
                                f"ranked {acc_percentile:.1f}th percentile). Target: {df['accuracy'].quantile(0.75):.1%}")
            elif acc_percentile < 80:
                suggestions.append(f"📈 Further improve accuracy to reach top quartile performance")

        # Reward suggestions
        if 'avg_reward' in percentile_rankings:
            reward_percentile = percentile_rankings['avg_reward']
            if reward_percentile < 50:
                suggestions.append(f"💰 Increase average reward (currently {user_metrics.get('avg_reward', 0):.6f} TAO - "
                                f"ranked {reward_percentile:.1f}th percentile). Target: {df['avg_reward_per_prediction'].quantile(0.75):.6f} TAO")

        # Speed suggestions
        if 'response_time' in percentile_rankings:
            speed_percentile = percentile_rankings['response_time']
            if speed_percentile > 50:  # Higher percentile means slower
                suggestions.append(f"⚡ Improve response time (currently {user_metrics.get('response_time', 0):.3f}s - "
                                f"ranked {speed_percentile:.1f}th percentile for speed). Target: <{df['avg_response_time'].quantile(0.25):.3f}s")

        # Uptime suggestions
        if 'uptime' in percentile_rankings:
            uptime_percentile = percentile_rankings['uptime']
            if uptime_percentile < 50:
                suggestions.append(f"🔋 Improve uptime (currently {user_metrics.get('uptime', 0):.1%} - "
                                f"ranked {uptime_percentile:.1f}th percentile). Target: >{df['uptime_percentage'].quantile(0.75):.1%}")

        # General suggestions
        if len(suggestions) == 0:
            suggestions.append("🎉 Your model is performing well! Focus on maintaining current performance levels")
        else:
            suggestions.insert(0, "🚀 Priority improvement opportunities:")

        return suggestions

    def save_analysis_results(self, analysis: Dict, comparison: Dict, filepath: str = "validator_analysis_results.json"):
        """Save comprehensive analysis results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'comparison': comparison,
            'summary': {
                'validators_analyzed': len(self.live_metrics),
                'competitive_position': comparison.get('competitive_position', {}).get('position_category', 'Unknown'),
                'key_insights': self.generate_key_insights(analysis, comparison)
            }
        }

        Path(filepath).parent.mkdir(exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Analysis results saved to {filepath}")

    def generate_key_insights(self, analysis: Dict, comparison: Dict) -> List[str]:
        """Generate key insights from the analysis"""
        insights = []

        if 'performance_distribution' in analysis:
            perf_dist = analysis['performance_distribution']

            # Accuracy insights
            acc_mean = perf_dist['accuracy']['mean']
            insights.append(f"📊 Network accuracy ranges from {perf_dist['accuracy']['min']:.1%} to {perf_dist['accuracy']['max']:.1%} (mean: {acc_mean:.1%})")

            # Reward insights
            reward_mean = perf_dist['avg_reward']['mean']
            insights.append(f"💰 Average validator reward: {reward_mean:.6f} TAO per prediction")

            # Top performer insights
            if 'top_performers' in analysis:
                top_acc = analysis['top_performers']['by_accuracy'][0]['accuracy'] if analysis['top_performers']['by_accuracy'] else 0
                top_reward = analysis['top_performers']['by_reward'][0]['avg_reward_per_prediction'] if analysis['top_performers']['by_reward'] else 0
                insights.append(f"🏆 Top performers achieve {top_acc:.1%} accuracy and {top_reward:.6f} TAO per prediction")

        # User comparison insights
        if comparison and 'competitive_position' in comparison:
            position = comparison['competitive_position']['position_category']
            percentile = comparison['competitive_position']['overall_percentile']
            insights.append(f"🎯 Your competitive position: {position} (overall percentile: {percentile:.1f}th)")

        return insights

    async def run_complete_analysis(self, user_metrics: Optional[Dict] = None) -> Dict:
        """Run complete analysis pipeline"""
        logger.info("🚀 Starting complete validator analysis...")

        # Discover all validators
        validator_runs = self.discover_all_validators()

        # Collect data from all validators
        validator_data = self.collect_validator_data(validator_runs)

        # Extract live metrics
        self.live_metrics = self.extract_live_metrics(validator_data)

        # Perform deep analysis
        analysis = self.analyze_validator_performance(self.live_metrics)

        # Compare with user's model if provided
        comparison = {}
        if user_metrics:
            comparison = self.compare_with_user_model(user_metrics, self.live_metrics)

        # Save results
        self.save_analysis_results(analysis, comparison)

        # Generate summary report
        summary = self.generate_analysis_summary(analysis, comparison)

        logger.info("✅ Complete analysis finished!")
        return {
            'analysis': analysis,
            'comparison': comparison,
            'summary': summary,
            'raw_data': {
                'validator_count': len(self.live_metrics),
                'total_runs_found': len(validator_runs)
            }
        }

    def generate_analysis_summary(self, analysis: Dict, comparison: Dict) -> str:
        """Generate human-readable analysis summary"""
        summary = []
        summary.append("🎯 PRECOG SUBNET 55 VALIDATOR ANALYSIS SUMMARY")
        summary.append("=" * 60)

        # Overview
        if 'overview' in analysis:
            ov = analysis['overview']
            summary.append(f"📊 Network Overview:")
            summary.append(f"   • Total Validators: {ov['total_validators']}")
            summary.append(f"   • Active Validators: {ov['active_validators']}")
            summary.append(f"   • Total Predictions: {ov['total_predictions_network']:,}")
            summary.append(f"   • Avg Predictions/Validator: {ov['avg_predictions_per_validator']:.0f}")
            summary.append("")

        # Performance distribution
        if 'performance_distribution' in analysis:
            pd = analysis['performance_distribution']
            summary.append(f"📈 Performance Distribution:")
            summary.append(f"   • Accuracy: {pd['accuracy']['mean']:.1%} ± {pd['accuracy']['std']:.1%}")
            summary.append(f"   • Rewards: {pd['avg_reward']['mean']:.6f} ± {pd['avg_reward']['std']:.6f} TAO")
            summary.append(f"   • Top 10% Accuracy: {pd['accuracy']['top_10_percent']:.1%}")
            summary.append(f"   • Top 10% Rewards: {pd['avg_reward']['top_10_percent']:.6f} TAO")
            summary.append("")

        # Top performers
        if 'top_performers' in analysis:
            tp = analysis['top_performers']
            summary.append(f"🏆 Top Performers:")
            if tp['by_accuracy']:
                top_acc = tp['by_accuracy'][0]
                summary.append(f"   • Highest Accuracy: {top_acc['accuracy']:.1%} (Validator: {top_acc['validator_id']})")
            if tp['by_reward']:
                top_rew = tp['by_reward'][0]
                summary.append(f"   • Highest Rewards: {top_rew['avg_reward_per_prediction']:.6f} TAO (Validator: {top_rew['validator_id']})")
            summary.append("")

        # User comparison
        if comparison and 'competitive_position' in comparison:
            cp = comparison['competitive_position']
            summary.append(f"🎯 Your Competitive Position:")
            summary.append(f"   • Overall Ranking: {cp['position_category']}")
            summary.append(f"   • Average Percentile: {cp['overall_percentile']:.1f}th")
            summary.append(f"   • Metrics Above Median: {cp['metrics_above_median']}/{cp['total_metrics_compared']}")
            summary.append("")

            if 'improvement_opportunities' in comparison:
                summary.append(f"🚀 Improvement Opportunities:")
                for opp in comparison['improvement_opportunities'][:3]:  # Top 3
                    summary.append(f"   • {opp}")
                summary.append("")

        return "\n".join(summary)

# Example usage and testing
async def main():
    """Main analysis function"""
    analyzer = WandbValidatorAnalyzer()

    # Example user metrics (replace with your actual metrics)
    user_metrics = {
        'accuracy': 0.85,  # 85% accuracy
        'avg_reward': 0.15,  # 0.15 TAO per prediction
        'response_time': 0.18,  # 0.18 seconds
        'uptime': 98.5  # 98.5% uptime
    }

    # Run complete analysis
    results = await analyzer.run_complete_analysis(user_metrics)

    # Print summary
    print(results['summary'])

    # Save detailed results
    analyzer.save_analysis_results(results['analysis'], results['comparison'])

    return results

if __name__ == "__main__":
    # Run analysis
    results = asyncio.run(main())

    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📊 Analyzed {results['raw_data']['validator_count']} validators")
    print(f"🔍 Found {results['raw_data']['total_runs_found']} total validator runs")
    print("💾 Results saved to 'validator_analysis_results.json'")
