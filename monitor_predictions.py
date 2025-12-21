#!/usr/bin/env python3
"""
Prediction Request Monitor - Shows real-time prediction activity

This script monitors for prediction requests and displays the communication content
when your miner receives them on testnet.
"""

import os
import time
import subprocess
from datetime import datetime
import json

def monitor_logs():
    """Monitor log files for prediction activity"""
    print("🎯 PREDICTION REQUEST MONITOR")
    print("=" * 50)
    print("Monitoring for prediction requests...")
    print("Press Ctrl+C to stop\n")

    log_file = 'logs/predictions.log'
    last_size = 0

    while True:
        try:
            # Check if miner is running
            result = subprocess.run(['pgrep', '-f', 'miner.py'],
                                  capture_output=True, text=True)
            miner_running = result.returncode == 0

            status = "🟢 RUNNING" if miner_running else "🔴 STOPPED"
            print(f"\rMiner Status: {status} | Last checked: {datetime.now().strftime('%H:%M:%S')}", end="", flush=True)

            # Check prediction log
            if os.path.exists(log_file):
                current_size = os.path.getsize(log_file)

                if current_size > last_size:
                    print(f"\n\n🎯 PREDICTION REQUEST DETECTED! ({current_size - last_size} bytes added)")
                    print("-" * 50)

                    # Read new content
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        new_lines = lines[last_size // 100:]  # Rough estimate of new lines

                        for line in new_lines[-5:]:  # Show last 5 entries
                            if line.strip():
                                try:
                                    # Parse CSV format: timestamp,asset,point_pred,lower_bound,upper_bound,regime
                                    parts = line.strip().split(',')
                                    if len(parts) >= 6:
                                        timestamp = parts[0]
                                        asset = parts[1]
                                        point_pred = float(parts[2])
                                        lower_bound = float(parts[3])
                                        upper_bound = float(parts[4])
                                        regime = parts[5]

                                        print("📨 PREDICTION REQUEST PROCESSED:"                                        print(f"   🕒 Timestamp: {timestamp}")
                                        print(f"   💰 Asset: {asset.upper()}")
                                        print(f"   🎯 Point Prediction: ${point_pred:,.2f}")
                                        print(f"   📊 Confidence Interval: [${lower_bound:,.2f}, ${upper_bound:,.2f}]")
                                        print(f"   📈 Market Regime: {regime}")
                                        print(f"   🎪 Interval Width: ${(upper_bound - lower_bound):,.2f}")
                                        print()

                                except Exception as e:
                                    print(f"   Raw log entry: {line.strip()}")

                    last_size = current_size
                    print("Continuing to monitor...\n")

            time.sleep(2)  # Check every 2 seconds

        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(5)

def show_recent_predictions():
    """Show recent prediction activity from logs"""
    print("📚 RECENT PREDICTION HISTORY")
    print("=" * 50)

    log_file = 'logs/predictions.log'

    if not os.path.exists(log_file):
        print("❌ No prediction log found")
        print("💡 Start miner and wait for requests to create this log")
        return

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            print("📝 Prediction log is empty")
            return

        print(f"📄 Total predictions logged: {len(lines)}")
        print()

        # Show last 10 predictions
        recent_lines = lines[-10:]

        for i, line in enumerate(recent_lines):
            if line.strip():
                try:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        timestamp = parts[0]
                        asset = parts[1]
                        point_pred = float(parts[2])
                        lower_bound = float(parts[3])
                        upper_bound = float(parts[4])
                        regime = parts[5]

                        print(f"#{len(lines) - 10 + i + 1}: {asset.upper()} | ${point_pred:.2f} | [{lower_bound:.2f}, {upper_bound:.2f}] | {regime} | {timestamp}")

                except Exception as e:
                    print(f"   Error parsing: {line.strip()}")

    except Exception as e:
        print(f"❌ Error reading log: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--history":
        show_recent_predictions()
    else:
        print("💡 Usage:")
        print("   python3 monitor_predictions.py          # Live monitoring")
        print("   python3 monitor_predictions.py --history # Show recent history")
        print()
        monitor_logs()




