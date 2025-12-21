#!/usr/bin/env python3
"""
Monitor Mainnet Deployment Performance
Real-time tracking vs miner 31 and performance metrics
"""

import time
import json
import os
import subprocess
from datetime import datetime, timedelta

class DeploymentMonitor:
    """Monitor deployment performance and compare with miner 31"""

    def __init__(self):
        self.start_time = time.time()
        self.miner31_baseline = {
            'avg_reward': 0.10,  # TAO per prediction
            'position': 'Top 10-15',
            'consistency': 'Moderate'
        }
        self.performance_history = []

    def get_current_performance(self):
        """Get current miner performance from logs"""
        perf = {
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': (time.time() - self.start_time) / 3600
        }

        # Find latest log file
        log_files = [f for f in os.listdir('.') if f.startswith('miner_') and f.endswith('.log')]
        if not log_files:
            return perf

        latest_log = max(log_files, key=os.path.getctime)

        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()

            # Count predictions
            predictions = len([l for l in lines if 'Prediction made' in l])
            perf['total_predictions'] = predictions

            # Count rewards
            rewards = len([l for l in lines if 'reward' in l.lower()])
            perf['total_rewards'] = rewards

            # Calculate rates
            uptime_hours = perf['uptime_hours']
            if uptime_hours > 0:
                perf['predictions_per_hour'] = predictions / uptime_hours
                perf['rewards_per_hour'] = rewards / uptime_hours

            # Extract reward values (if available)
            reward_values = []
            for line in lines:
                if 'reward' in line.lower():
                    # Try to extract numeric values
                    import re
                    numbers = re.findall(r'[\d.]+', line)
                    if numbers:
                        try:
                            reward_values.append(float(numbers[0]))
                        except:
                            pass

            if reward_values:
                perf['avg_reward_per_prediction'] = sum(reward_values) / len(reward_values)
                perf['max_reward'] = max(reward_values)
                perf['min_reward'] = min(reward_values)

            # Check for domination features
            perf['peak_hour_active'] = any('Peak Hour' in l for l in lines[-50:])
            perf['market_regime_detected'] = any('Market Regime' in l for l in lines[-50:])

        except Exception as e:
            perf['error'] = str(e)

        return perf

    def compare_with_miner31(self, current_perf):
        """Compare current performance with miner 31"""
        comparison = {
            'timestamp': current_perf['timestamp'],
            'uptime_hours': current_perf['uptime_hours']
        }

        # Reward comparison
        current_avg_reward = current_perf.get('avg_reward_per_prediction', 0)
        miner31_avg = self.miner31_baseline['avg_reward']

        comparison['current_avg_reward'] = current_avg_reward
        comparison['miner31_avg_reward'] = miner31_avg

        if current_avg_reward > 0:
            if current_avg_reward > miner31_avg:
                improvement = (current_avg_reward - miner31_avg) / miner31_avg * 100
                comparison['status'] = 'ahead'
                comparison['improvement_pct'] = improvement
                comparison['message'] = f"AHEAD by +{improvement:.1f}%"
            else:
                deficit = (miner31_avg - current_avg_reward) / miner31_avg * 100
                comparison['status'] = 'behind'
                comparison['deficit_pct'] = deficit
                comparison['message'] = f"BEHIND by -{deficit:.1f}%"
        else:
            comparison['status'] = 'no_data'
            comparison['message'] = "Collecting data..."

        # Performance targets based on uptime
        uptime = current_perf['uptime_hours']
        predictions = current_perf.get('total_predictions', 0)

        if uptime >= 48:
            target_predictions = 200  # 48 hours at ~4 predictions/hour
            target_rewards = 150      # Conservative estimate
        elif uptime >= 24:
            target_predictions = 100
            target_rewards = 75
        elif uptime >= 12:
            target_predictions = 50
            target_rewards = 35
        else:
            target_predictions = 25
            target_rewards = 15

        comparison['targets'] = {
            'predictions_needed': max(0, target_predictions - predictions),
            'rewards_needed': max(0, target_rewards - current_perf.get('total_rewards', 0)),
            'predictions_actual': predictions,
            'rewards_actual': current_perf.get('total_rewards', 0)
        }

        return comparison

    def print_status_report(self, current_perf, comparison):
        """Print comprehensive status report"""
        print("\n" + "=" * 80)
        print("🎯 DEPLOYMENT PERFORMANCE REPORT")
        print("=" * 80)
        print(f"⏰ Uptime: {current_perf['uptime_hours']:.1f} hours")
        print(f"📅 Started: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n📊 CURRENT PERFORMANCE:")
        print(f"   Predictions made: {current_perf.get('total_predictions', 0)}")
        print(f"   Rewards received: {current_perf.get('total_rewards', 0)}")

        if 'predictions_per_hour' in current_perf:
            print(f"   Predictions/hour: {current_perf['predictions_per_hour']:.1f}")
        if 'rewards_per_hour' in current_perf:
            print(f"   Rewards/hour: {current_perf['rewards_per_hour']:.1f}")
        if 'avg_reward_per_prediction' in current_perf:
            print(f"   Avg reward/prediction: {current_perf['avg_reward_per_prediction']:.6f} TAO")
        print(f"   Peak hour optimization: {'✅ Active' if current_perf.get('peak_hour_active') else '⏸️  Inactive'}")
        print(f"   Market regime detection: {'✅ Active' if current_perf.get('market_regime_detected') else '⏸️  Inactive'}")

        print("\n🏆 COMPARISON WITH MINER 31:")
        print(f"   Your avg reward: {comparison['current_avg_reward']:.6f} TAO")
        print(f"   Miner 31 avg reward: {comparison['miner31_avg_reward']:.6f} TAO")
        print(f"   Status: {comparison['message']}")

        targets = comparison['targets']
        print("\n🎯 PROGRESS TOWARDS TARGETS:")
        if uptime < 12:
            print("   Phase: Early deployment (0-12h)")
        elif uptime < 24:
            print("   Phase: Establishment (12-24h)")
        elif uptime < 48:
            print("   Phase: Domination (24-48h)")
        else:
            print("   Phase: Sustained leadership (48h+)")

        print(f"   Predictions: {targets['predictions_actual']}/{targets['predictions_actual'] + targets['predictions_needed']}")
        print(f"   Rewards: {targets['rewards_actual']}/{targets['rewards_actual'] + targets['rewards_needed']}")

        remaining_predictions = targets['predictions_needed']
        remaining_rewards = targets['rewards_needed']

        if remaining_predictions <= 0 and remaining_rewards <= 0:
            print("   ✅ TARGETS ACHIEVED! You're dominating!")
        else:
            print(f"   📈 Need {remaining_predictions} more predictions, {remaining_rewards} more rewards")

        # Risk assessment
        print("\n🛡️ RISK ASSESSMENT:")
        if current_perf.get('total_predictions', 0) == 0:
            print("   ⚠️  No predictions made - check model loading")
        elif current_perf.get('total_rewards', 0) == 0:
            print("   ⚠️  No rewards received - check validator connections")
        elif comparison['status'] == 'behind':
            print("   ⚠️  Behind miner 31 - consider model improvements")
        else:
            print("   ✅ Performance looks good - continue monitoring")

    def run_continuous_monitoring(self):
        """Run continuous monitoring"""
        print("📊 STARTING CONTINUOUS DEPLOYMENT MONITORING")
        print("Press Ctrl+C to stop monitoring")
        print("Reports generated every 5 minutes")
        print()

        try:
            while True:
                current_perf = self.get_current_performance()
                comparison = self.compare_with_miner31(current_perf)

                self.print_status_report(current_perf, comparison)

                # Save to history
                self.performance_history.append({
                    'performance': current_perf,
                    'comparison': comparison
                })

                # Save snapshot every hour
                if len(self.performance_history) % 12 == 0:  # Every hour (12 * 5min)
                    self.save_snapshot()

                time.sleep(300)  # 5 minutes

        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped")
            self.save_final_report()

    def save_snapshot(self):
        """Save current performance snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deployment_snapshot_{timestamp}.json"

        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'performance': self.get_current_performance(),
            'comparison': self.compare_with_miner31(self.get_current_performance())
        }

        with open(filename, 'w') as f:
            json.dump(snapshot, f, indent=2)

        print(f"💾 Snapshot saved: {filename}")

    def save_final_report(self):
        """Save final comprehensive report"""
        final_report = {
            'deployment_summary': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_uptime_hours': (time.time() - self.start_time) / 3600,
                'snapshots_taken': len(self.performance_history)
            },
            'final_performance': self.get_current_performance(),
            'final_comparison': self.compare_with_miner31(self.get_current_performance()),
            'performance_history': self.performance_history
        }

        filename = f"deployment_final_report_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(final_report, f, indent=2)

        print(f"📋 Final report saved: {filename}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Mainnet Deployment Performance")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuous monitoring")
    parser.add_argument("--snapshot", action="store_true",
                       help="Save current performance snapshot")
    parser.add_argument("--report", action="store_true",
                       help="Generate final performance report")

    args = parser.parse_args()

    monitor = DeploymentMonitor()

    if args.snapshot:
        monitor.save_snapshot()
        print("✅ Performance snapshot saved")

    elif args.report:
        monitor.save_final_report()
        print("✅ Final performance report generated")

    elif args.continuous:
        monitor.run_continuous_monitoring()

    else:
        # Single status report
        current_perf = monitor.get_current_performance()
        comparison = monitor.compare_with_miner31(current_perf)
        monitor.print_status_report(current_perf, comparison)

if __name__ == "__main__":
    main()
