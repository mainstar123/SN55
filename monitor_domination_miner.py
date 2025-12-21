#!/usr/bin/env python3
"""
Monitoring Script for Precog #1 Miner Domination System
Real-time dashboard and performance tracking
"""

import sys
import os
import time
import logging
from datetime import datetime, timezone, timedelta
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import domination system components
from performance_tracking_system import PerformanceDashboard
from start_domination_miner import DominationMiner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DominationMonitor:
    """
    Real-time monitoring system for the domination miner
    """

    def __init__(self, config: dict):
        self.config = config
        self.miner = None
        self.dashboard = None
        self.is_monitoring = False

        # Monitoring settings
        self.update_interval = config.get('update_interval', 300)  # 5 minutes
        self.alert_check_interval = config.get('alert_check_interval', 60)  # 1 minute
        self.save_interval = config.get('save_interval', 3600)  # 1 hour

        # Alert thresholds
        self.alert_thresholds = {
            'min_daily_tao': config.get('min_daily_tao', 0.05),
            'max_mape': config.get('max_mape', 0.20),
            'min_uptime': config.get('min_uptime', 95.0)
        }

        logger.info("📊 Domination Monitor initialized")

    def connect_to_miner(self):
        """Connect to running domination miner"""
        try:
            self.miner = DominationMiner(self.config)
            self.miner.load_domination_system()
            logger.info("✅ Connected to Domination Miner")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to miner: {e}")
            return False

    def start_monitoring(self):
        """Start real-time monitoring"""
        logger.info("🚀 Starting Domination Miner Monitoring")
        self.is_monitoring = True

        if not self.connect_to_miner():
            logger.error("Cannot start monitoring without miner connection")
            return False

        print("\n" + "="*70)
        print("📊 PRECOG DOMINATION MINER MONITOR")
        print("="*70)
        print("💡 Press Ctrl+C to stop monitoring")
        print("="*70)

        last_update = datetime.now(timezone.utc)
        last_alert_check = datetime.now(timezone.utc)
        last_save = datetime.now(timezone.utc)

        try:
            while self.is_monitoring:
                current_time = datetime.now(timezone.utc)

                # Periodic full dashboard update
                if (current_time - last_update).total_seconds() >= self.update_interval:
                    self._display_full_dashboard()
                    last_update = current_time

                # Frequent alert checks
                if (current_time - last_alert_check).total_seconds() >= self.alert_check_interval:
                    self._check_alerts()
                    last_alert_check = current_time

                # Periodic data saves
                if (current_time - last_save).total_seconds() >= self.save_interval:
                    self._save_monitoring_data()
                    last_save = current_time

                time.sleep(10)  # Check every 10 seconds

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.stop_monitoring()

    def _display_full_dashboard(self):
        """Display comprehensive monitoring dashboard"""
        try:
            status = self.miner.get_status()
            performance = status['current_performance']

            # Clear screen for clean display
            os.system('clear' if os.name == 'posix' else 'cls')

            print("\n" + "="*70)
            print("🎯 PRECOG DOMINATION MINER - LIVE MONITOR")
            print("="*70)
            print(f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

            # Performance Summary
            print("
📊 PERFORMANCE SUMMARY:"            print(f"  Status: {'🟢 RUNNING' if status['is_running'] else '🔴 STOPPED'}")
            print(f"  Device: {status['device']}")
            print(f"  Model: {'✅ LOADED' if status['model_loaded'] else '❌ MISSING'}")
            print(f"  Total Predictions: {status['total_predictions']}")
            print(".6f"            print(".4f"
            # Current Metrics
            print("
🎯 CURRENT METRICS:"            print(".1%"            print(".1%"            if 'peak_hour_accuracy' in performance and performance['peak_hour_accuracy'] > 0:
                print(".1%"            print(f"  Current Regime: {status.get('current_regime', 'unknown')}")

            # Peak Hour Information
            next_peak = status.get('next_peak_window', {})
            if next_peak:
                print("
⏰ PEAK HOUR STATUS:"                peak_time = next_peak.get('next_peak_start')
                if peak_time:
                    print(f"  Next Peak: {peak_time.strftime('%m-%d %H:%M UTC')}")
                    print(f"  Wait Time: {next_peak.get('wait_minutes', 0)} minutes")
                    print(f"  Currently Peak: {'✅ YES' if next_peak.get('is_currently_peak') else '❌ NO'}")

            # Alerts
            alerts = status.get('alerts', [])
            if alerts:
                print("
🚨 ACTIVE ALERTS:"                for alert in alerts:
                    severity_icon = "🔴" if alert['severity'] == 'high' else "🟡"
                    print(f"  {severity_icon} {alert['message']}")
            else:
                print("
✅ NO ACTIVE ALERTS"
            # Recommendations
            recommendations = status.get('recommendations', [])
            if recommendations:
                print("
💡 RECOMMENDATIONS:"                for rec in recommendations:
                    print(f"  • {rec}")

            print("="*70)

        except Exception as e:
            logger.error(f"Dashboard display error: {e}")
            print(f"❌ Dashboard error: {e}")

    def _check_alerts(self):
        """Check for critical alerts and notifications"""
        try:
            status = self.miner.get_status()
            performance = status['current_performance']

            alerts_triggered = []

            # Daily TAO alert
            recent_daily = performance.get('daily_rewards', [])
            if recent_daily:
                avg_daily = sum(recent_daily[-3:]) / max(1, len(recent_daily[-3:]))
                if avg_daily < self.alert_thresholds['min_daily_tao']:
                    alerts_triggered.append({
                        'type': 'low_earnings',
                        'message': '.4f'                        'severity': 'high'
                    })

            # Accuracy alert
            current_mape = performance.get('avg_mape', 0)
            if current_mape > self.alert_thresholds['max_mape']:
                alerts_triggered.append({
                    'type': 'accuracy',
                    'message': '.1%"                    'severity': 'high'
                })

            # Display alerts
            for alert in alerts_triggered:
                severity_icon = "🔴" if alert['severity'] == 'high' else "🟡"
                print(f"\n{severity_icon} ALERT: {alert['message']}")

        except Exception as e:
            logger.error(f"Alert check error: {e}")

    def _save_monitoring_data(self):
        """Save monitoring data periodically"""
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"domination_monitor_{timestamp}.json"

            status = self.miner.get_status()
            data = {
                'timestamp': timestamp,
                'status': status,
                'alert_thresholds': self.alert_thresholds
            }

            with open(filename, 'w') as f:
                import json
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Monitoring data saved to {filename}")

        except Exception as e:
            logger.error(f"Data save error: {e}")

    def get_performance_report(self) -> dict:
        """Generate detailed performance report"""
        try:
            status = self.miner.get_status()

            report = {
                'summary': {
                    'total_predictions': status['total_predictions'],
                    'total_earnings': status['total_earnings'],
                    'avg_daily_earnings': status['total_earnings'] / max(1, (datetime.now(timezone.utc).date() - datetime(2024, 1, 1).date()).days),
                    'current_accuracy': 1 - status['current_performance'].get('avg_mape', 0),
                    'uptime_percentage': status.get('uptime_percentage', 100.0)
                },
                'alerts': status.get('alerts', []),
                'recommendations': status.get('recommendations', []),
                'peak_performance': status.get('next_peak_window', {}),
                'regime_performance': status['current_performance'].get('regime_performance', {}),
                'generated_at': datetime.now(timezone.utc).isoformat()
            }

            return report

        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {'error': str(e)}

    def export_report(self, filename: str = None):
        """Export detailed performance report"""
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"domination_performance_report_{timestamp}.json"

        report = self.get_performance_report()

        try:
            with open(filename, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Performance report exported to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Export error: {e}")
            return None

    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        logger.info("🛑 Stopping Domination Monitor")
        self.is_monitoring = False

        if self.miner:
            self.miner.stop_mining()

        print("\n📊 Final monitoring report exported")


def main():
    """Main monitoring function"""
    parser = argparse.ArgumentParser(description='Monitor Precog #1 Miner Domination System')
    parser.add_argument('--update_interval', type=int, default=300, help='Dashboard update interval (seconds)')
    parser.add_argument('--alert_interval', type=int, default=60, help='Alert check interval (seconds)')
    parser.add_argument('--save_interval', type=int, default=3600, help='Data save interval (seconds)')
    parser.add_argument('--min_daily_tao', type=float, default=0.05, help='Minimum daily TAO alert threshold')
    parser.add_argument('--max_mape', type=float, default=0.20, help='Maximum MAPE alert threshold')
    parser.add_argument('--export_report', action='store_true', help='Export performance report on exit')

    args = parser.parse_args()

    # Configuration
    config = {
        'update_interval': args.update_interval,
        'alert_check_interval': args.alert_interval,
        'save_interval': args.save_interval,
        'min_daily_tao': args.min_daily_tao,
        'max_mape': args.max_mape,
        'min_uptime': 95.0
    }

    print("📊 Starting Precog Domination Miner Monitor")
    print("=" * 50)

    try:
        # Create monitor
        monitor = DominationMonitor(config)

        # Start monitoring
        monitor.start_monitoring()

        # Export final report if requested
        if args.export_report:
            report_file = monitor.export_report()
            if report_file:
                print(f"📄 Final report saved: {report_file}")

    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise


if __name__ == "__main__":
    main()
