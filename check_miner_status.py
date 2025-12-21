#!/usr/bin/env python3
"""
Miner Status Checker - Comprehensive testnet monitoring tool

Checks:
- Process status
- Network connectivity
- Metagraph registration
- Prediction request activity
- Wallet balance and emissions
"""

import os
import sys
import time
import subprocess
import requests
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_process_status():
    """Check if miner process is running"""
    print("🔍 CHECKING MINER PROCESS...")
    print("-" * 40)

    # Check for miner.py processes
    result = subprocess.run(['pgrep', '-f', 'miner.py'],
                          capture_output=True, text=True)

    if result.returncode == 0:
        pids = result.stdout.strip().split('\n')
        print(f"✅ Miner running (PIDs: {', '.join(pids)})")

        # Get process details
        for pid in pids:
            try:
                cmd = f"ps -p {pid} -o pid,ppid,cmd,etime"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"   Process details: {result.stdout.strip()}")
            except:
                pass
        return True
    else:
        print("❌ No miner process running")
        return False

def check_network_connectivity():
    """Check if we can reach the testnet endpoint"""
    print("\n🌐 CHECKING NETWORK CONNECTIVITY...")
    print("-" * 40)

    # Test basic HTTP connectivity
    test_url = "https://archive.opentensor.ai:443"
    try:
        response = requests.get(test_url, timeout=10)
        print(f"✅ HTTP connectivity: {response.status_code}")
    except Exception as e:
        print(f"❌ HTTP connectivity failed: {e}")

    # Test WebSocket endpoint (simulated)
    ws_endpoint = "wss://test.finney.opentensor.ai:443"
    print(f"📡 WebSocket endpoint: {ws_endpoint}")
    print("   Note: Full WebSocket test requires running miner")

def check_wallet_registration():
    """Check wallet registration status"""
    print("\n👛 CHECKING WALLET REGISTRATION...")
    print("-" * 40)

    try:
        # Import bittensor with proper environment
        os.environ['HOME'] = '/home/ocean'
        import bittensor as bt

        # Load wallet
        wallet = bt.wallet(name='cold_draven', hotkey='default')
        hotkey_address = wallet.hotkey.ss58_address

        print(f"📧 Hotkey address: {hotkey_address[:20]}...")

        # Check registration on testnet
        subtensor = bt.subtensor(network='test')
        is_registered = subtensor.is_hotkey_registered(
            netuid=256,
            hotkey_ss58=hotkey_address
        )

        if is_registered:
            print("✅ Wallet registered on testnet subnet 256")

            # Get stake and balance
            stake = subtensor.get_total_stake_for_hotkey(hotkey_address)
            balance = subtensor.get_balance(wallet.coldkey.ss58_address)

            print(f"💰 Stake: {stake:.6f} τ")
            print(f"💵 Balance: {balance.tao:.6f} τ")

            return True
        else:
            print("❌ Wallet NOT registered on testnet subnet 256")
            print("💡 Register with: btcli subnet register --netuid 256 --wallet.name cold_draven")
            return False

    except Exception as e:
        print(f"❌ Error checking registration: {e}")
        return False

def check_prediction_activity():
    """Check for prediction request activity"""
    print("\n🎯 CHECKING PREDICTION ACTIVITY...")
    print("-" * 40)

    # Check prediction logs
    log_files = [
        'logs/predictions.log',
        'miner.log',
        'miner_domination.log'
    ]

    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            mtime = datetime.fromtimestamp(os.path.getmtime(log_file))

            print(f"📄 {log_file}:")
            print(f"   Size: {size} bytes")
            print(f"   Modified: {mtime}")

            if size > 0:
                # Check for recent prediction activity
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        lines = content.split('\n')

                        # Look for prediction keywords
                        pred_lines = [line for line in lines if any(
                            keyword in line.lower() for keyword in
                            ['prediction', 'forward', 'synapse', 'btc', 'price']
                        )]

                        if pred_lines:
                            print(f"   🎯 Found {len(pred_lines)} prediction-related entries")
                            # Show last few
                            for line in pred_lines[-3:]:
                                if line.strip():
                                    print(f"   └─ {line.strip()[:100]}...")
                        else:
                            print("   📝 No prediction activity found")

                except Exception as e:
                    print(f"   ❌ Error reading log: {e}")
            else:
                print("   📝 Empty log file")
        else:
            print(f"📄 {log_file}: NOT FOUND")

def get_metagraph_status():
    """Get metagraph status if possible"""
    print("\n📊 CHECKING METAGRAPH STATUS...")
    print("-" * 40)

    try:
        os.environ['HOME'] = '/home/ocean'
        import bittensor as bt

        # Try to load metagraph
        metagraph = bt.metagraph(netuid=256, network='test')
        metagraph.sync()

        print(f"✅ Metagraph loaded: {len(metagraph.uids)} miners")

        # Find our miner
        wallet = bt.wallet(name='cold_draven', hotkey='default')
        our_uid = None

        for uid in range(len(metagraph.hotkeys)):
            if metagraph.hotkeys[uid] == wallet.hotkey.ss58_address:
                our_uid = uid
                break

        if our_uid is not None:
            print(f"🎯 Your miner UID: {our_uid}")
            print(f"🏆 Rank: {our_uid + 1}/{len(metagraph.uids)}")
            print(f"💰 Emissions: {metagraph.emission[our_uid]:.8f} τ")
            print(f"📈 Incentive: {metagraph.incentive[our_uid]:.6f}")
            print(f"🤝 Trust: {metagraph.trust[our_uid]:.6f}")
            print(f"🔄 Dividends: {metagraph.dividends[our_uid]:.6f}")

            if metagraph.emission[our_uid] > 0:
                print("🎉 ACTIVE: Receiving emissions (processing requests!)")
            else:
                print("⏳ INACTIVE: No emissions yet")

            return our_uid
        else:
            print("❌ Your miner not found in metagraph")
            return None

    except Exception as e:
        print(f"❌ Cannot load metagraph: {e}")
        print("💡 This is expected if testnet is down")
        return None

def main():
    """Main status check"""
    print("🚀 PRECOG MINER STATUS CHECK")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    print()

    # Run all checks
    process_running = check_process_status()
    check_network_connectivity()
    wallet_registered = check_wallet_registration()
    check_prediction_activity()
    uid = get_metagraph_status()

    # Summary and recommendations
    print("\n📋 SUMMARY & RECOMMENDATIONS")
    print("=" * 50)

    issues = []
    recommendations = []

    if not process_running:
        issues.append("Miner process not running")
        recommendations.append("Start miner with: ./start_testnet_miner.sh")

    if not wallet_registered:
        issues.append("Wallet not registered")
        recommendations.append("Register wallet: btcli subnet register --netuid 256 --wallet.name cold_draven")

    if uid is None:
        issues.append("Miner not in metagraph")
        recommendations.append("Check registration and metagraph sync")

    if not os.path.exists('logs/predictions.log') or os.path.getsize('logs/predictions.log') == 0:
        issues.append("No prediction activity")
        recommendations.append("Wait for connection and requests, or check testnet status")

    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   • {issue}")
        print()
        print("🔧 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   • {rec}")
    else:
        print("✅ All systems operational!")
        print("🎯 Miner should be receiving prediction requests")

    print("\n💡 MONITORING COMMANDS:")
    print("   • Live logs: tail -f miner.log")
    print("   • Check periodically: python3 check_miner_status.py")
    print("   • Stop miner: pkill -f miner.py")

if __name__ == "__main__":
    main()
