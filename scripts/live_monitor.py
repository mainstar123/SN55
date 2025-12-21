"""
Live Performance Monitor for Precog Subnet 55 Miner
Tracks rewards, accuracy, and provides real-time adaptation
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import pandas as pd
import numpy as np
from pathlib import Path

from ensemble_trainer import EnsembleMetaLearner, OnlineLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LivePerformanceMonitor:
    """Monitors live miner performance and provides adaptation recommendations"""

    def __init__(self, miner_uid: int = 55, update_interval: int = 300):  # 5 minutes
        self.miner_uid = miner_uid
        self.update_interval = update_interval
        self.performance_data = []
        self.reward_history = []
        self.prediction_history = []
        self.competitor_analysis = {}

        # Performance thresholds based on current subnet data
        self.reward_thresholds = {
            'excellent': 0.15,    # Top 5% (0.166-0.200)
            'good': 0.05,         # Top 25%
            'average': 0.022,     # Current average
            'poor': 0.001         # Bottom 25%
        }

        self.performance_file = Path("performance/live_metrics.json")
        self.performance_file.parent.mkdir(exist_ok=True)

    async def monitor_performance(self):
        """Main monitoring loop"""
        logger.info("🚀 Starting live performance monitoring...")

        while True:
            try:
                await self.collect_performance_data()
                await self.analyze_performance()
                await self.generate_adaptation_recommendations()
                await self.save_performance_data()

                logger.info(f"📊 Performance update completed. Next update in {self.update_interval}s")
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"❌ Performance monitoring error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute

    async def collect_performance_data(self):
        """Collect current performance metrics"""
        # This would integrate with Bittensor CLI or API
        # For now, simulate data collection
        current_time = datetime.now()

        # Simulate getting latest prediction and reward data
        # In real implementation, this would query the subnet
        latest_data = {
            'timestamp': current_time.isoformat(),
            'prediction': np.random.normal(228.55, 2.0),  # Based on current average
            'actual_price': np.random.normal(88000, 500),  # BTC price simulation
            'reward': np.random.exponential(0.022),  # Based on current average
            'rank': np.random.randint(1, 256),  # Random rank in subnet
            'response_time': np.random.normal(0.18, 0.02),  # Your model's speed
            'uptime': 0.98 + np.random.random() * 0.04  # 98-100% uptime
        }

        self.performance_data.append(latest_data)
        self.reward_history.append(latest_data['reward'])

        # Keep only last 1000 data points
        if len(self.performance_data) > 1000:
            self.performance_data = self.performance_data[-1000:]
            self.reward_history = self.reward_history[-1000:]

    async def analyze_performance(self):
        """Analyze current performance trends"""
        if len(self.performance_data) < 10:
            return

        recent_data = self.performance_data[-100:]  # Last 100 predictions
        recent_rewards = [d['reward'] for d in recent_data]

        analysis = {
            'timestamp': datetime.now().isoformat(),
            'avg_reward_1h': np.mean(recent_rewards),
            'avg_reward_24h': np.mean(self.reward_history[-288:]) if len(self.reward_history) >= 288 else np.mean(self.reward_history),
            'reward_volatility': np.std(recent_rewards),
            'best_reward': max(recent_rewards),
            'worst_reward': min(recent_rewards),
            'reward_trend': self.calculate_trend(recent_rewards),
            'performance_category': self.categorize_performance(np.mean(recent_rewards)),
            'rank_percentile': np.percentile([d['rank'] for d in recent_data], 50),
            'response_time_avg': np.mean([d['response_time'] for d in recent_data]),
            'uptime_avg': np.mean([d['uptime'] for d in recent_data])
        }

        logger.info(f"📈 Performance Analysis: {analysis['performance_category']} "
                   f"(Avg: {analysis['avg_reward_1h']:.6f} TAO)")

        return analysis

    def calculate_trend(self, rewards: List[float]) -> str:
        """Calculate reward trend"""
        if len(rewards) < 20:
            return "insufficient_data"

        # Simple linear trend
        x = np.arange(len(rewards))
        slope = np.polyfit(x, rewards, 1)[0]

        if slope > 0.001:
            return "improving"
        elif slope < -0.001:
            return "declining"
        else:
            return "stable"

    def categorize_performance(self, avg_reward: float) -> str:
        """Categorize performance based on rewards"""
        if avg_reward >= self.reward_thresholds['excellent']:
            return "🏆 EXCELLENT (Top 5%)"
        elif avg_reward >= self.reward_thresholds['good']:
            return "✅ GOOD (Top 25%)"
        elif avg_reward >= self.reward_thresholds['average']:
            return "📊 AVERAGE"
        else:
            return "⚠️ NEEDS_IMPROVEMENT"

    async def generate_adaptation_recommendations(self):
        """Generate recommendations for model adaptation"""
        if len(self.performance_data) < 50:
            return

        recent_performance = self.performance_data[-50:]
        avg_reward = np.mean([d['reward'] for d in recent_performance])
        trend = self.calculate_trend([d['reward'] for d in recent_performance])

        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'immediate_actions': [],
            'model_adaptations': [],
            'hyperparameter_tuning': [],
            'market_focus': []
        }

        # Immediate actions based on performance
        if avg_reward < self.reward_thresholds['average']:
            recommendations['immediate_actions'].extend([
                "🔧 Check model prediction accuracy",
                "⚡ Verify response time < 0.35s",
                "🔍 Analyze prediction vs actual price correlation"
            ])

        if trend == "declining":
            recommendations['immediate_actions'].append(
                "📉 Performance declining - consider model retraining"
            )

        # Model adaptations
        if avg_reward < self.reward_thresholds['good']:
            recommendations['model_adaptations'].extend([
                "🎯 Implement ensemble methods with regime detection",
                "🧠 Add meta-learning layer for prediction refinement",
                "📊 Increase feature engineering complexity",
                "🔄 Enable online learning adaptation"
            ])

        # Hyperparameter suggestions
        recommendations['hyperparameter_tuning'].extend([
            "🎛️ Learning rate: Try 0.0005-0.001 for stability",
            "🏗️ Hidden size: Test 256-512 for more capacity",
            "🎯 Attention heads: Experiment with 12-16 heads",
            "📈 Dropout: Reduce to 0.05-0.1 for better learning"
        ])

        # Market focus based on performance patterns
        recommendations['market_focus'].extend([
            "🌙 Focus on BTC volatility patterns during Asian sessions",
            "💰 Weight TAO predictions higher during high volume periods",
            "📈 Monitor ETH correlation for better TAO predictions",
            "🎪 Adapt to market regime changes (bull/bear/sideways)"
        ])

        logger.info("💡 Adaptation Recommendations Generated")
        for action in recommendations['immediate_actions'][:2]:  # Show top 2
            logger.info(f"  {action}")

        return recommendations

    async def save_performance_data(self):
        """Save performance data for analysis"""
        data = {
            'performance_history': self.performance_data[-500:],  # Last 500 entries
            'reward_history': self.reward_history[-500:],
            'last_updated': datetime.now().isoformat(),
            'summary_stats': {
                'total_predictions': len(self.performance_data),
                'avg_reward_all_time': np.mean(self.reward_history) if self.reward_history else 0,
                'best_reward': max(self.reward_history) if self.reward_history else 0,
                'current_trend': self.calculate_trend(self.reward_history[-100:]) if len(self.reward_history) >= 100 else "unknown"
            }
        }

        with open(self.performance_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

class RewardOptimizer:
    """Optimizes miner behavior for maximum TAO rewards"""

    def __init__(self, monitor: LivePerformanceMonitor):
        self.monitor = monitor
        self.optimization_strategies = {
            'timing_optimization': self.optimize_prediction_timing,
            'accuracy_focus': self.focus_on_accuracy,
            'speed_optimization': self.optimize_speed,
            'regime_adaptation': self.adapt_to_market_regime
        }

    async def optimize_prediction_timing(self) -> Dict[str, str]:
        """Optimize when to make predictions for higher rewards"""
        # Analyze historical data for best timing
        return {
            'best_hours': '9-11 UTC, 13-15 UTC',  # High volume periods
            'avoid_hours': '2-6 UTC',  # Low volume periods
            'frequency': 'Every 30-60 minutes during optimal hours'
        }

    async def focus_on_accuracy(self) -> Dict[str, str]:
        """Strategies to improve prediction accuracy"""
        return {
            'feature_engineering': 'Add more technical indicators',
            'ensemble_methods': 'Combine multiple model predictions',
            'online_learning': 'Continuously adapt to new data',
            'regime_detection': 'Different models for different market conditions'
        }

    async def optimize_speed(self) -> Dict[str, str]:
        """Optimize inference speed"""
        return {
            'model_pruning': 'Remove unnecessary parameters',
            'quantization': 'Use 8-bit precision where possible',
            'caching': 'Cache frequent computations',
            'gpu_optimization': 'Maximize GPU utilization'
        }

    async def adapt_to_market_regime(self) -> Dict[str, str]:
        """Adapt to current market conditions"""
        return {
            'bull_market': 'Focus on momentum indicators',
            'bear_market': 'Use mean-reversion signals',
            'volatile_market': 'Increase prediction frequency',
            'ranging_market': 'Use support/resistance levels'
        }

    async def get_optimal_strategy(self) -> Dict[str, str]:
        """Get the best optimization strategy based on current performance"""
        current_performance = await self.monitor.analyze_performance()

        if current_performance['avg_reward_1h'] < self.monitor.reward_thresholds['average']:
            return await self.focus_on_accuracy()
        elif current_performance['response_time_avg'] > 0.25:
            return await self.optimize_speed()
        else:
            return await self.adapt_to_market_regime()

async def main():
    """Main monitoring and optimization loop"""
    monitor = LivePerformanceMonitor()
    optimizer = RewardOptimizer(monitor)

    logger.info("🎯 Starting Precog Subnet 55 Performance Monitor")
    logger.info("=" * 60)

    # Start monitoring tasks
    monitor_task = asyncio.create_task(monitor.monitor_performance())

    # Periodic optimization checks
    while True:
        try:
            strategy = await optimizer.get_optimal_strategy()
            logger.info("🎯 Current Optimal Strategy:")
            for key, value in strategy.items():
                logger.info(f"  {key}: {value}")

            await asyncio.sleep(1800)  # Check every 30 minutes

        except Exception as e:
            logger.error(f"❌ Optimization error: {e}")
            await asyncio.sleep(300)  # Retry after 5 minutes

if __name__ == "__main__":
    asyncio.run(main())
