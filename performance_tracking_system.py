"""
Real-Time Performance Tracking System for Precog #1 Miner
Monitors predictions, rewards, and automatically optimizes strategy
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from datetime import datetime, timezone, timedelta
import logging
import json
import os
from collections import deque
import threading
import time
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Data structure for prediction tracking"""
    timestamp: datetime
    prediction: float
    actual: Optional[float] = None
    reward: float = 0.0
    confidence: float = 0.0
    market_regime: str = 'unknown'
    is_peak_hour: bool = False
    model_version: str = 'v1.0'
    prediction_time_ms: float = 0.0
    batch_size: int = 1


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    total_predictions: int = 0
    successful_predictions: int = 0
    total_reward: float = 0.0
    avg_mape: float = 0.0
    avg_reward_per_prediction: float = 0.0
    peak_hour_accuracy: float = 0.0
    regime_performance: Dict[str, float] = None
    daily_rewards: List[float] = None
    uptime_percentage: float = 100.0

    def __post_init__(self):
        if self.regime_performance is None:
            self.regime_performance = {}
        if self.daily_rewards is None:
            self.daily_rewards = []


class RealTimePerformanceTracker:
    """
    Real-time performance tracking with automatic optimization
    Tracks predictions, rewards, and triggers model updates
    """

    def __init__(self, model, window_size: int = 1000, auto_optimize: bool = True,
                 optimization_interval: int = 100):
        self.model = model
        self.window_size = window_size
        self.auto_optimize = auto_optimize
        self.optimization_interval = optimization_interval

        # Data storage
        self.prediction_history = deque(maxlen=window_size)
        self.daily_stats = {}
        self.regime_stats = {}

        # Performance tracking
        self.current_metrics = PerformanceMetrics()
        self.performance_history = []

        # Optimization
        self.last_optimization = datetime.now(timezone.utc)
        self.optimization_thread = None
        self.optimization_lock = threading.Lock()

        # Alert thresholds
        self.alert_thresholds = {
            'min_accuracy': 0.85,  # Minimum MAPE threshold
            'min_daily_reward': 0.01,  # Minimum daily TAO
            'max_prediction_time_ms': 100,  # Maximum prediction time
            'min_uptime': 95.0  # Minimum uptime percentage
        }

        # Start background monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self.monitor_thread.start()

    def record_prediction(self, prediction_record: PredictionRecord):
        """Record a prediction for performance tracking"""
        self.prediction_history.append(prediction_record)

        # Update daily stats
        date_key = prediction_record.timestamp.date()
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                'predictions': 0,
                'rewards': 0.0,
                'mape_sum': 0.0,
                'peak_predictions': 0,
                'regime_counts': {}
            }

        daily = self.daily_stats[date_key]
        daily['predictions'] += 1
        daily['rewards'] += prediction_record.reward

        if prediction_record.actual is not None:
            mape = abs(prediction_record.prediction - prediction_record.actual) / max(abs(prediction_record.actual), 1e-6)
            daily['mape_sum'] += mape

        if prediction_record.is_peak_hour:
            daily['peak_predictions'] += 1

        # Update regime stats
        regime = prediction_record.market_regime
        if regime not in daily['regime_counts']:
            daily['regime_counts'][regime] = 0
        daily['regime_counts'][regime] += 1

        # Update current metrics
        self._update_current_metrics()

        # Check for optimization trigger
        if (self.auto_optimize and
            len(self.prediction_history) % self.optimization_interval == 0):
            self._trigger_optimization()

    def _update_current_metrics(self):
        """Update current performance metrics"""
        if not self.prediction_history:
            return

        recent_predictions = list(self.prediction_history)[-100:]  # Last 100 predictions

        # Basic metrics
        self.current_metrics.total_predictions = len(self.prediction_history)
        self.current_metrics.successful_predictions = sum(1 for p in recent_predictions
                                                        if p.actual is not None and
                                                        abs(p.prediction - p.actual) / max(abs(p.actual), 1e-6) < 0.1)

        self.current_metrics.total_reward = sum(p.reward for p in self.prediction_history)
        self.current_metrics.avg_reward_per_prediction = (self.current_metrics.total_reward /
                                                        max(1, self.current_metrics.total_predictions))

        # Accuracy metrics
        valid_predictions = [p for p in recent_predictions if p.actual is not None]
        if valid_predictions:
            mapes = [abs(p.prediction - p.actual) / max(abs(p.actual), 1e-6) for p in valid_predictions]
            self.current_metrics.avg_mape = np.mean(mapes)

        # Peak hour accuracy
        peak_predictions = [p for p in recent_predictions if p.is_peak_hour and p.actual is not None]
        if peak_predictions:
            peak_mapes = [abs(p.prediction - p.actual) / max(abs(p.actual), 1e-6) for p in peak_predictions]
            self.current_metrics.peak_hour_accuracy = 1 - np.mean(peak_mapes)

        # Regime performance
        regime_performance = {}
        for regime in ['bull', 'bear', 'volatile', 'ranging', 'unknown']:
            regime_preds = [p for p in recent_predictions if p.market_regime == regime and p.actual is not None]
            if regime_preds:
                regime_mapes = [abs(p.prediction - p.actual) / max(abs(p.actual), 1e-6) for p in regime_preds]
                regime_performance[regime] = 1 - np.mean(regime_mapes)

        self.current_metrics.regime_performance = regime_performance

        # Daily rewards (last 7 days)
        last_week = datetime.now(timezone.utc).date() - timedelta(days=7)
        recent_daily = [date for date in self.daily_stats.keys() if date >= last_week]
        self.current_metrics.daily_rewards = [self.daily_stats[date]['rewards'] for date in sorted(recent_daily)]

    def get_performance_report(self) -> Dict[str, Union[float, int, Dict, List]]:
        """Generate comprehensive performance report"""
        report = {
            'current_metrics': asdict(self.current_metrics),
            'alerts': self._check_alerts(),
            'trends': self._calculate_trends(),
            'recommendations': self._generate_recommendations(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        return report

    def _check_alerts(self) -> List[Dict[str, Union[str, float]]]:
        """Check for performance alerts"""
        alerts = []

        # Accuracy alert
        if self.current_metrics.avg_mape > (1 - self.alert_thresholds['min_accuracy']):
            alerts.append({
                'type': 'accuracy',
                'severity': 'high',
                'message': '.4f'                'current_value': self.current_metrics.avg_mape,
                'threshold': 1 - self.alert_thresholds['min_accuracy']
            })

        # Reward alert
        recent_daily_avg = np.mean(self.current_metrics.daily_rewards[-3:]) if self.current_metrics.daily_rewards else 0
        if recent_daily_avg < self.alert_thresholds['min_daily_reward']:
            alerts.append({
                'type': 'reward',
                'severity': 'high',
                'message': '.4f'                'current_value': recent_daily_avg,
                'threshold': self.alert_thresholds['min_daily_reward']
            })

        # Prediction time alert
        recent_times = [p.prediction_time_ms for p in list(self.prediction_history)[-50:]]
        if recent_times and np.mean(recent_times) > self.alert_thresholds['max_prediction_time_ms']:
            alerts.append({
                'type': 'latency',
                'severity': 'medium',
                'message': '.1f'                'current_value': np.mean(recent_times),
                'threshold': self.alert_thresholds['max_prediction_time_ms']
            })

        return alerts

    def _calculate_trends(self) -> Dict[str, Union[float, str]]:
        """Calculate performance trends"""
        trends = {}

        if len(self.performance_history) < 2:
            return {'trend_available': False}

        # Reward trend (last 7 days vs previous 7 days)
        recent_rewards = self.current_metrics.daily_rewards[-7:] if len(self.current_metrics.daily_rewards) >= 7 else []
        older_rewards = self.current_metrics.daily_rewards[-14:-7] if len(self.current_metrics.daily_rewards) >= 14 else []

        if recent_rewards and older_rewards:
            recent_avg = np.mean(recent_rewards)
            older_avg = np.mean(older_rewards)
            reward_trend = ((recent_avg - older_avg) / max(older_avg, 1e-6)) * 100

            trends['reward_trend_7d'] = reward_trend
            trends['reward_trend_direction'] = 'improving' if reward_trend > 5 else 'declining' if reward_trend < -5 else 'stable'

        # Accuracy trend
        recent_accuracy = 1 - self.current_metrics.avg_mape
        if self.performance_history:
            older_accuracy = np.mean([1 - h['avg_mape'] for h in self.performance_history[-7:] if 'avg_mape' in h])
            if older_accuracy > 0:
                accuracy_trend = ((recent_accuracy - older_accuracy) / older_accuracy) * 100
                trends['accuracy_trend'] = accuracy_trend

        return trends

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Based on current performance
        if self.current_metrics.avg_mape > 0.15:  # Poor accuracy
            recommendations.append("Retraining recommended: Current MAPE too high")

        if self.current_metrics.avg_reward_per_prediction < 0.0005:  # Low reward
            recommendations.append("Timing optimization needed: Low reward per prediction")

        # Based on regime performance
        regime_perf = self.current_metrics.regime_performance
        if regime_perf:
            worst_regime = min(regime_perf.items(), key=lambda x: x[1])
            if worst_regime[1] < 0.8:
                recommendations.append(f"Regime optimization needed: Poor performance in {worst_regime[0]} markets")

        # Based on trends
        trends = self._calculate_trends()
        if 'reward_trend_7d' in trends and trends['reward_trend_7d'] < -10:
            recommendations.append("Immediate attention needed: Reward declining significantly")

        if not recommendations:
            recommendations.append("Performance stable: Continue current strategy")

        return recommendations

    def _trigger_optimization(self):
        """Trigger background optimization"""
        if self.optimization_thread and self.optimization_thread.is_alive():
            return  # Already running

        self.optimization_thread = threading.Thread(target=self._run_optimization, daemon=True)
        self.optimization_thread.start()

    def _run_optimization(self):
        """Run background optimization"""
        with self.optimization_lock:
            logger.info("Starting background performance optimization...")

            try:
                # Analyze recent performance
                analysis = self._analyze_recent_performance()

                # Apply optimizations
                if analysis['needs_model_update']:
                    self._optimize_model_weights()

                if analysis['needs_timing_adjustment']:
                    self._optimize_prediction_timing()

                self.last_optimization = datetime.now(timezone.utc)
                logger.info("Background optimization completed")

            except Exception as e:
                logger.error(f"Optimization failed: {e}")

    def _analyze_recent_performance(self) -> Dict[str, Union[bool, float]]:
        """Analyze recent performance for optimization opportunities"""
        recent = list(self.prediction_history)[-self.optimization_interval:]

        analysis = {
            'needs_model_update': False,
            'needs_timing_adjustment': False,
            'accuracy_drop': 0.0,
            'reward_drop': 0.0
        }

        if len(recent) < 50:
            return analysis

        # Check accuracy trend
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]

        first_accuracy = np.mean([abs(p.prediction - (p.actual or p.prediction)) /
                                max(abs(p.actual or p.prediction), 1e-6) for p in first_half])
        second_accuracy = np.mean([abs(p.prediction - (p.actual or p.prediction)) /
                                 max(abs(p.actual or p.prediction), 1e-6) for p in second_half])

        accuracy_drop = second_accuracy - first_accuracy
        if accuracy_drop > 0.05:  # Accuracy worsened
            analysis['needs_model_update'] = True
            analysis['accuracy_drop'] = accuracy_drop

        # Check reward trend
        first_rewards = [p.reward for p in first_half]
        second_rewards = [p.reward for p in second_half]

        if first_rewards and second_rewards:
            first_avg_reward = np.mean(first_rewards)
            second_avg_reward = np.mean(second_rewards)
            reward_drop = (first_avg_reward - second_avg_reward) / max(first_avg_reward, 1e-6)

            if reward_drop > 0.2:  # 20% reward drop
                analysis['needs_timing_adjustment'] = True
                analysis['reward_drop'] = reward_drop

        return analysis

    def _optimize_model_weights(self):
        """Optimize model ensemble weights based on performance"""
        logger.info("Optimizing ensemble weights...")

        # This would implement dynamic weight adjustment
        # For now, just log the optimization trigger
        pass

    def _optimize_prediction_timing(self):
        """Optimize prediction timing based on reward patterns"""
        logger.info("Optimizing prediction timing...")

        # This would adjust peak hour detection
        # For now, just log the optimization trigger
        pass

    def _background_monitor(self):
        """Background monitoring thread"""
        while self.monitoring_active:
            try:
                # Periodic performance snapshot
                if len(self.prediction_history) > 0 and len(self.prediction_history) % 100 == 0:
                    snapshot = self.get_performance_report()
                    self.performance_history.append(snapshot)

                    # Keep only recent history
                    if len(self.performance_history) > 10:
                        self.performance_history = self.performance_history[-10:]

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                time.sleep(60)

    def save_performance_data(self, filepath: str):
        """Save performance data to file"""
        data = {
            'prediction_history': [asdict(p) for p in self.prediction_history],
            'daily_stats': {str(k): v for k, v in self.daily_stats.items()},
            'current_metrics': asdict(self.current_metrics),
            'performance_history': self.performance_history,
            'last_optimization': self.last_optimization.isoformat(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Performance data saved to {filepath}")

    def load_performance_data(self, filepath: str):
        """Load performance data from file"""
        if not os.path.exists(filepath):
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore prediction history
        self.prediction_history.clear()
        for record in data.get('prediction_history', []):
            record['timestamp'] = datetime.fromisoformat(record['timestamp'])
            self.prediction_history.append(PredictionRecord(**record))

        # Restore other data
        self.daily_stats = {k: v for k, v in data.get('daily_stats', {}).items()}
        self.performance_history = data.get('performance_history', [])

        logger.info(f"Performance data loaded from {filepath}")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)


class PerformanceDashboard:
    """
    Real-time performance dashboard for monitoring miner status
    """

    def __init__(self, tracker: RealTimePerformanceTracker):
        self.tracker = tracker

    def get_dashboard_data(self) -> Dict[str, Union[float, int, str, List, Dict]]:
        """Get formatted dashboard data"""
        report = self.tracker.get_performance_report()

        dashboard = {
            'summary': {
                'total_predictions': report['current_metrics']['total_predictions'],
                'total_reward': ".6f",
                'avg_daily_reward': ".4f",
                'current_accuracy': ".1%",
                'peak_hour_accuracy': ".1%",
                'uptime': ".1f"            },
            'charts': {
                'daily_rewards': report['current_metrics']['daily_rewards'],
                'regime_performance': report['current_metrics']['regime_performance']
            },
            'alerts': report['alerts'],
            'recommendations': report['recommendations'],
            'last_update': datetime.now(timezone.utc).isoformat()
        }

        return dashboard

    def print_dashboard(self):
        """Print formatted dashboard to console"""
        data = self.get_dashboard_data()

        print("\n" + "="*60)
        print("🎯 PRECOG MINER PERFORMANCE DASHBOARD")
        print("="*60)

        summary = data['summary']
        print("\n📊 SUMMARY:")
        print(f"  Total Predictions: {summary['total_predictions']}")
        print(f"  Total TAO Earned: {summary['total_reward']}")
        print(f"  Avg Daily TAO: {summary['avg_daily_reward']}")
        print(f"  Current Accuracy: {summary['current_accuracy']}")
        print(f"  Peak Hour Accuracy: {summary['peak_hour_accuracy']}")
        print(f"  System Uptime: {summary['uptime']}")

        if data['alerts']:
            print("
🚨 ALERTS:"            for alert in data['alerts']:
                severity_icon = "🔴" if alert['severity'] == 'high' else "🟡"
                print(f"  {severity_icon} {alert['message']}")

        if data['recommendations']:
            print("
💡 RECOMMENDATIONS:"            for rec in data['recommendations']:
                print(f"  • {rec}")

        print(f"\n⏰ Last Update: {data['last_update']}")
        print("="*60)


def create_performance_tracking_system(model) -> Tuple[RealTimePerformanceTracker, PerformanceDashboard]:
    """Create complete performance tracking system"""
    tracker = RealTimePerformanceTracker(model)
    dashboard = PerformanceDashboard(tracker)

    return tracker, dashboard


if __name__ == "__main__":
    # Test performance tracking system
    print("📊 Testing Real-Time Performance Tracking System")
    print("=" * 55)

    # Create mock model
    model = nn.Linear(10, 1)

    # Create tracking system
    tracker, dashboard = create_performance_tracking_system(model)

    # Simulate predictions
    print("\n🎯 Simulating predictions and tracking performance...")

    base_time = datetime.now(timezone.utc)
    for i in range(200):
        # Generate mock prediction
        actual = np.random.randn() * 0.01 + 0.001
        prediction = actual + np.random.randn() * 0.005  # Add some error
        reward = np.random.random() * 0.001  # Random reward
        confidence = np.random.random()

        # Create prediction record
        record = PredictionRecord(
            timestamp=base_time + timedelta(minutes=i*15),  # Every 15 minutes
            prediction=float(prediction),
            actual=float(actual),
            reward=float(reward),
            confidence=float(confidence),
            market_regime=np.random.choice(['bull', 'bear', 'volatile', 'ranging']),
            is_peak_hour=(i % 4 == 0),  # Every 4th prediction is peak hour
            prediction_time_ms=np.random.uniform(10, 50)
        )

        tracker.record_prediction(record)

    # Display dashboard
    dashboard.print_dashboard()

    # Save performance data
    tracker.save_performance_data('performance_test_data.json')

    print("\n💾 Performance data saved to performance_test_data.json")
    print("✅ Real-Time Performance Tracking System Ready!")

    # Stop monitoring
    tracker.stop_monitoring()
