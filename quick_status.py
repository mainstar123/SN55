#!/usr/bin/env python3
import sys
sys.path.append('.')
from mining_dashboard import get_metagraph_data, get_wallet_balance
import subprocess
import os

def check_prediction_requests():
    """Check if miner has received prediction requests"""
    print("🔍 CHECKING PREDICTION REQUEST STATUS...")

    # Check if miner is running
    try:
        with open('miner.pid', 'r') as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # Check if process exists
        print("✅ Miner is running")
    except:
        print("❌ Miner not running")
        return

    # Get metagraph data
    df = get_metagraph_data()
    your_miner = df[df['uid'] == 35]  # Your UID

    if len(your_miner) > 0:
        row = your_miner.iloc[0]
        print("✅ YOU ARE IN METAGRAPH - REQUESTS RECEIVED!")
        print(f"🏆 Rank: {row['rank']} / {len(df)} miners")
        print(f"💰 Emissions: {row['emissions']} τ")
        print(f"📊 Incentive: {row['incentive']}")
        print(f"🤝 Trust: {row['trust']}")

        # Check for activity (emissions > 0 means requests processed)
        if row['emissions'] > 0:
            print("🎯 PREDICTION REQUESTS: CONFIRMED ACTIVE")
        else:
            print("⏳ PREDICTION REQUESTS: WAITING FOR FIRST ONES")
    else:
        print("❌ NOT IN METAGRAPH YET - NO REQUESTS PROCESSED")

    # Check wallet balance
    balance = get_wallet_balance()
    print(f"💰 Wallet Balance: {balance} τ")

if __name__ == "__main__":
    check_prediction_requests()