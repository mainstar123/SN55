#!/usr/bin/env python3
"""
Competitor monitoring and analysis for Precog Subnet 55

Features:
- Track top-20 miners' emissions and rankings
- Monitor model performance changes via Taostats API
- Alert on significant ranking/emission shifts
- Analyze competitor patterns and predict changes
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
import json

import requests
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompetitorMonitor:
    """Monitor competitor performance on Precog subnet"""

    def __init__(self, netuid=55, top_n=20):
        self.netuid = netuid
        self.top_n = top_n
        self.taostats_base = "https://api.taostats.io/api"
        self.history_file = "data/competitor_history.json"

        # Create data directory
        os.makedirs("data", exist_ok=True)

        # Load historical data
        self.history = self.load_history()

    def load_history(self):
        """Load competitor performance history"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading history: {e}")

        return {
            'last_update': None,
            'rankings': [],
            'alerts': []
        }

    def save_history(self):
        """Save competitor performance history"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving history: {e}")

    def fetch_subnet_data(self):
        """Fetch current subnet data from Taostats API"""
        try:
            url = f"{self.taostats_base}/subnet/{self.netuid}/metagraph"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching subnet data: {e}")
            return None

    def analyze_competitors(self, data):
        """Analyze competitor performance and detect changes"""
        if not data or 'neurons' not in data:
            logger.error("Invalid subnet data received")
            return None

        neurons = data['neurons']

        # Extract top miners
        df = pd.DataFrame(neurons)
        df['incentive'] = pd.to_numeric(df['incentive'], errors='coerce')
        df['emissions'] = pd.to_numeric(df['emissions'], errors='coerce')
        df['trust'] = pd.to_numeric(df['trust'], errors='coerce')
        df['uid'] = pd.to_numeric(df['uid'], errors='coerce')

        # Sort by incentive score (primary ranking metric)
        top_miners = df.nlargest(self.top_n, 'incentive').copy()

        # Calculate derived metrics
        top_miners['rank'] = range(1, len(top_miners) + 1)
        top_miners['daily_tao'] = top_miners['emissions'] * 9.72  # Current subnet emissions

        # Convert to dict for JSON serialization
        current_rankings = []
        for _, miner in top_miners.iterrows():
            miner_dict = {
                'uid': int(miner['uid']),
                'rank': int(miner['rank']),
                'incentive': float(miner['incentive']),
                'emissions': float(miner['emissions']),
                'trust': float(miner['trust']),
                'daily_tao': float(miner['daily_tao']),
                'timestamp': datetime.now().isoformat()
            }
            current_rankings.append(miner_dict)

        return current_rankings

    def detect_changes(self, current_rankings):
        """Detect significant changes from previous rankings"""
        if not self.history['rankings']:
            return []

        previous_rankings = self.history['rankings'][-1] if self.history['rankings'] else []
        if not previous_rankings:
            return []

        alerts = []
        current_uids = {r['uid']: r for r in current_rankings}
        previous_uids = {r['uid']: r for r in previous_rankings}

        # Check for new entrants to top-20
        new_entrants = set(current_uids.keys()) - set(previous_uids.keys())
        if new_entrants:
            for uid in new_entrants:
                rank = current_uids[uid]['rank']
                alerts.append({
                    'type': 'new_entrant',
                    'uid': uid,
                    'rank': rank,
                    'message': f"New miner UID {uid} entered top-{self.top_n} at rank {rank}"
                })

        # Check for ranking changes
        for uid, current in current_uids.items():
            if uid in previous_uids:
                previous = previous_uids[uid]
                rank_change = previous['rank'] - current['rank']
                incentive_change = current['incentive'] - previous['incentive']

                # Alert on significant rank improvements
                if rank_change >= 3:  # Improved by 3+ positions
                    alerts.append({
                        'type': 'rank_improvement',
                        'uid': uid,
                        'old_rank': previous['rank'],
                        'new_rank': current['rank'],
                        'change': rank_change,
                        'message': f"UID {uid} improved from rank {previous['rank']} to {current['rank']} (+{rank_change})"
                    })

                # Alert on incentive score jumps
                if incentive_change > 0.05:  # >5% improvement
                    alerts.append({
                        'type': 'incentive_jump',
                        'uid': uid,
                        'old_incentive': previous['incentive'],
                        'new_incentive': current['incentive'],
                        'change': incentive_change,
                        'message': f"UID {uid} incentive jumped by {incentive_change:.3f} to {current['incentive']:.3f}"
                    })

        # Check for emission drops (potential performance issues)
        for uid, previous in previous_uids.items():
            if uid in current_uids:
                current = current_uids[uid]
                emission_change = (current['emissions'] - previous['emissions']) / previous['emissions']

                if emission_change < -0.5:  # >50% drop
                    alerts.append({
                        'type': 'emission_drop',
                        'uid': uid,
                        'old_emissions': previous['emissions'],
                        'new_emissions': current['emissions'],
                        'change_percent': emission_change * 100,
                        'message': f"UID {uid} emissions dropped by {emission_change*100:.1f}% to {current['emissions']:.6f}"
                    })

        return alerts

    def get_performance_summary(self, current_rankings):
        """Generate performance summary for top miners"""
        if not current_rankings:
            return None

        df = pd.DataFrame(current_rankings)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'top_miner_count': len(current_rankings),
            'avg_incentive': df['incentive'].mean(),
            'avg_emissions': df['emissions'].mean(),
            'avg_daily_tao': df['daily_tao'].mean(),
            'incentive_std': df['incentive'].std(),
            'emissions_std': df['emissions'].std(),
            'top_performer': {
                'uid': int(df.loc[df['incentive'].idxmax(), 'uid']),
                'incentive': float(df['incentive'].max()),
                'rank': 1
            }
        }

        return summary

    def monitor_once(self):
        """Run a single monitoring cycle"""
        logger.info(f"Monitoring subnet {self.netuid} competitors...")

        # Fetch current data
        data = self.fetch_subnet_data()
        if not data:
            return None

        # Analyze competitors
        current_rankings = self.analyze_competitors(data)
        if not current_rankings:
            return None

        # Detect changes
        alerts = self.detect_changes(current_rankings)

        # Generate summary
        summary = self.get_performance_summary(current_rankings)

        # Update history
        self.history['rankings'].append(current_rankings)
        self.history['last_update'] = datetime.now().isoformat()

        # Keep only last 30 days of history
        cutoff = datetime.now() - timedelta(days=30)
        self.history['rankings'] = [
            r for r in self.history['rankings']
            if datetime.fromisoformat(r[0]['timestamp']) > cutoff
        ]

        # Record alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            self.history['alerts'].append(alert)

        # Save history
        self.save_history()

        # Log results
        logger.info(f"Found {len(current_rankings)} top miners")
        logger.info(f"Top performer: UID {summary['top_performer']['uid']} "
                   f"(incentive: {summary['top_performer']['incentive']:.4f})")

        if alerts:
            logger.warning(f"Detected {len(alerts)} significant changes:")
            for alert in alerts:
                logger.warning(f"  {alert['message']}")
        else:
            logger.info("No significant changes detected")

        return {
            'rankings': current_rankings,
            'alerts': alerts,
            'summary': summary
        }

    def continuous_monitoring(self, interval_minutes=60):
        """Run continuous monitoring"""
        logger.info(f"Starting continuous monitoring every {interval_minutes} minutes...")

        while True:
            try:
                self.monitor_once()
                logger.info(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying


def print_rankings_table(rankings):
    """Print formatted rankings table"""
    if not rankings:
        return

    print("\n" + "="*80)
    print("PRECROG SUBNET 55 - TOP MINERS RANKINGS")
    print("="*80)

    print(f"{'Rank':<5} {'UID':<6} {'Incentive':<10} {'Emissions':<10} {'Trust':<8} {'Daily TAO':<10}")
    print("-" * 80)

    for miner in rankings:
        print(f"{miner['rank']:<5} {miner['uid']:<6} "
              f"{miner['incentive']:<10.4f} {miner['emissions']:<10.6f} "
              f"{miner['trust']:<8.4f} {miner['daily_tao']:<10.4f}")

    print("="*80)


def main():
    """Main monitoring function"""
    import argparse

    parser = argparse.ArgumentParser(description='Monitor Precog subnet competitors')
    parser.add_argument('--netuid', type=int, default=55, help='Subnet UID to monitor')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top miners to track')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in minutes')

    args = parser.parse_args()

    monitor = CompetitorMonitor(netuid=args.netuid, top_n=args.top_n)

    if args.continuous:
        monitor.continuous_monitoring(interval_minutes=args.interval)
    else:
        results = monitor.monitor_once()

        if results:
            print_rankings_table(results['rankings'])

            if results['alerts']:
                print(f"\n🚨 ALERTS ({len(results['alerts'])}):")
                for alert in results['alerts']:
                    print(f"  • {alert['message']}")


if __name__ == "__main__":
    main()
