#!/usr/bin/env python3
"""
Real-time performance validation for Precog miner

Continuously validates model predictions against actual CoinMetrics prices:
- Calculates MAPE, RMSE for point forecasts
- Measures interval coverage rates
- Monitors response times
- Sends alerts on performance degradation
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precog.utils.cm_data import CMData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceValidator:
    """Validate model performance in real-time"""

    def __init__(self):
        self.predictions_file = 'logs/predictions.log'
        self.performance_file = 'logs/performance_metrics.json'
        self.alerts_file = 'logs/performance_alerts.log'

        # Create logs directory
        os.makedirs('logs', exist_ok=True)

        # Performance thresholds
        self.thresholds = {
            'mape': 0.0012,  # 0.12%
            'coverage': 0.85,  # 85%
            'response_time': 5.0,  # 5 seconds
            'max_mape': 0.0015,  # 0.15% (alert threshold)
        }

        # Historical performance tracking
        self.performance_history = self.load_performance_history()

    def load_performance_history(self):
        """Load historical performance metrics"""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance history: {e}")

        return {
            'last_update': None,
            'hourly_metrics': [],
            'daily_summary': []
        }

    def save_performance_history(self):
        """Save performance metrics history"""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")

    def load_recent_predictions(self, hours=24):
        """Load recent predictions from log file"""
        if not os.path.exists(self.predictions_file):
            return []

        predictions = []
        cutoff_time = datetime.now() - timedelta(hours=hours)

        try:
            with open(self.predictions_file, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split(',')
                        if len(parts) >= 6:
                            timestamp = datetime.fromisoformat(parts[0])
                            if timestamp >= cutoff_time:
                                prediction = {
                                    'timestamp': timestamp,
                                    'asset': parts[1],
                                    'point': float(parts[2]),
                                    'lower': float(parts[3]),
                                    'upper': float(parts[4]),
                                    'regime': parts[5]
                                }
                                predictions.append(prediction)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing prediction line: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error loading predictions: {e}")

        return predictions

    def get_actual_prices(self, predictions):
        """Fetch actual prices for prediction timestamps + 1 hour"""
        if not predictions:
            return {}

        try:
            cm = CMData()

            # Group predictions by timestamp for batch fetching
            timestamps = {}
            for pred in predictions:
                actual_time = pred['timestamp'] + timedelta(hours=1)
                time_key = actual_time.isoformat()

                if time_key not in timestamps:
                    timestamps[time_key] = []
                timestamps[time_key].append(pred)

            actual_prices = {}

            # Fetch actual prices
            for time_str, preds in timestamps.items():
                try:
                    actual_time = datetime.fromisoformat(time_str)
                    start_time = actual_time - timedelta(minutes=5)  # 5-min window
                    end_time = actual_time + timedelta(minutes=5)

                    data = cm.get_CM_ReferenceRate(
                        assets=['btc'],
                        start=start_time.isoformat(),
                        end=end_time.isoformat(),
                        frequency="1m"
                    )

                    if not data.empty and 'ReferenceRateUSD' in data.columns:
                        # Take the price closest to the target time
                        data['time'] = pd.to_datetime(data['time'])
                        data['time_diff'] = abs(data['time'] - actual_time)
                        closest_price = data.loc[data['time_diff'].idxmin(), 'ReferenceRateUSD']

                        for pred in preds:
                            actual_prices[pred['timestamp'].isoformat()] = float(closest_price)

                except Exception as e:
                    logger.error(f"Error fetching price for {time_str}: {e}")
                    continue

            return actual_prices

        except Exception as e:
            logger.error(f"Error in get_actual_prices: {e}")
            return {}

    def calculate_metrics(self, predictions, actual_prices):
        """Calculate performance metrics"""
        if not predictions:
            return None

        point_errors = []
        coverages = []
        interval_widths = []

        for pred in predictions:
            timestamp_key = pred['timestamp'].isoformat()
            if timestamp_key in actual_prices:
                actual = actual_prices[timestamp_key]

                # Point forecast error
                point_error = abs(pred['point'] - actual) / actual
                point_errors.append(point_error)

                # Interval coverage
                coverage = 1 if pred['lower'] <= actual <= pred['upper'] else 0
                coverages.append(coverage)

                # Interval width
                width = (pred['upper'] - pred['lower']) / actual
                interval_widths.append(width)

        if not point_errors:
            return None

        # Calculate metrics
        mape = np.mean(point_errors)
        rmse = np.sqrt(np.mean(np.array(point_errors) ** 2))
        coverage_rate = np.mean(coverages)
        avg_width = np.mean(interval_widths)

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(point_errors),
            'mape': mape,
            'mape_percent': mape * 100,
            'rmse': rmse,
            'coverage_rate': coverage_rate,
            'coverage_percent': coverage_rate * 100,
            'avg_interval_width_percent': avg_width * 100,
            'point_errors': point_errors,
            'coverages': coverages
        }

        return metrics

    def check_alerts(self, metrics):
        """Check for performance alerts"""
        alerts = []

        if metrics['mape'] > self.thresholds['max_mape']:
            alerts.append({
                'type': 'high_mape',
                'severity': 'critical',
                'message': f"MAPE too high: {metrics['mape_percent']:.4f}% (threshold: {self.thresholds['max_mape']*100:.4f}%)",
                'metric': 'mape',
                'value': metrics['mape'],
                'threshold': self.thresholds['max_mape']
            })

        if metrics['coverage_rate'] < self.thresholds['coverage']:
            alerts.append({
                'type': 'low_coverage',
                'severity': 'warning',
                'message': f"Coverage too low: {metrics['coverage_percent']:.1f}% (threshold: {self.thresholds['coverage']*100:.1f}%)",
                'metric': 'coverage',
                'value': metrics['coverage_rate'],
                'threshold': self.thresholds['coverage']
            })

        return alerts

    def log_alerts(self, alerts):
        """Log performance alerts"""
        if not alerts:
            return

        try:
            with open(self.alerts_file, 'a') as f:
                for alert in alerts:
                    alert_line = f"{datetime.now().isoformat()}|{alert['severity']}|{alert['type']}|{alert['message']}\n"
                    f.write(alert_line)

            # Also log to console
            for alert in alerts:
                if alert['severity'] == 'critical':
                    logger.error(f"🚨 CRITICAL: {alert['message']}")
                else:
                    logger.warning(f"⚠️  WARNING: {alert['message']}")

        except Exception as e:
            logger.error(f"Error logging alerts: {e}")

    def validate_once(self, hours_lookback=24):
        """Run a single validation cycle"""
        logger.info(f"Validating performance over last {hours_lookback} hours...")

        # Load recent predictions
        predictions = self.load_recent_predictions(hours=hours_lookback)
        logger.info(f"Loaded {len(predictions)} predictions")

        if len(predictions) < 5:  # Need minimum sample size
            logger.warning("Insufficient predictions for validation")
            return None

        # Get actual prices
        actual_prices = self.get_actual_prices(predictions)
        logger.info(f"Fetched {len(actual_prices)} actual prices")

        if len(actual_prices) < 5:
            logger.warning("Insufficient actual prices for validation")
            return None

        # Calculate metrics
        metrics = self.calculate_metrics(predictions, actual_prices)

        if not metrics:
            logger.error("Failed to calculate metrics")
            return None

        # Check for alerts
        alerts = self.check_alerts(metrics)

        # Update performance history
        self.performance_history['hourly_metrics'].append(metrics)
        self.performance_history['last_update'] = datetime.now().isoformat()

        # Keep only last 7 days of hourly metrics
        cutoff = datetime.now() - timedelta(days=7)
        self.performance_history['hourly_metrics'] = [
            m for m in self.performance_history['hourly_metrics']
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]

        # Save history
        self.save_performance_history()

        # Log alerts
        self.log_alerts(alerts)

        # Log results
        logger.info("Performance Metrics:")
        logger.info(f"  Sample Size: {metrics['sample_size']}")
        logger.info(f"  MAPE: {metrics['mape_percent']:.4f}% (Target: <{self.thresholds['mape']*100:.4f}%)")
        logger.info(f"  RMSE: ${metrics['rmse']:.2f}")
        logger.info(f"  Coverage: {metrics['coverage_percent']:.1f}% (Target: >{self.thresholds['coverage']*100:.1f}%)")
        logger.info(f"  Avg Interval Width: {metrics['avg_interval_width_percent']:.1f}%")

        if alerts:
            logger.warning(f"Generated {len(alerts)} alerts")
        else:
            logger.info("✅ All metrics within acceptable ranges")

        return metrics

    def continuous_validation(self, interval_minutes=60):
        """Run continuous performance validation"""
        logger.info(f"Starting continuous validation every {interval_minutes} minutes...")

        while True:
            try:
                self.validate_once()
                logger.info(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("Validation stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in validation loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying


def main():
    """Main validation function"""
    import argparse

    parser = argparse.ArgumentParser(description='Validate Precog miner performance')
    parser.add_argument('--continuous', action='store_true', help='Run continuous validation')
    parser.add_argument('--interval', type=int, default=60, help='Validation interval in minutes')
    parser.add_argument('--hours', type=int, default=24, help='Hours of predictions to validate')

    args = parser.parse_args()

    validator = PerformanceValidator()

    if args.continuous:
        validator.continuous_validation(interval_minutes=args.interval)
    else:
        metrics = validator.validate_once(hours_lookback=args.hours)

        if metrics:
            print("\n" + "="*50)
            print("PERFORMANCE VALIDATION RESULTS")
            print("="*50)
            print(f"Sample Size: {metrics['sample_size']}")
            print(f"MAPE: {metrics['mape_percent']:.4f}%")
            print(f"RMSE: ${metrics['rmse']:.2f}")
            print(f"Coverage: {metrics['coverage_percent']:.1f}%")
            print(f"Avg Interval Width: {metrics['avg_interval_width_percent']:.1f}%")
            print("="*50)


if __name__ == "__main__":
    main()
