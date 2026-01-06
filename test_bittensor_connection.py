#!/usr/bin/env python3
"""Test basic Bittensor connection for subnet 55"""

import bittensor as bt
import sys
import os

def test_connection():
    """Test basic Bittensor connection"""
    try:
        print("🔍 Testing Bittensor connection to subnet 55...")

        # Initialize subtensor
        subtensor = bt.subtensor(network='finney')

        # Test connection
        print("✅ Subtensor initialized")

        # Get metagraph
        metagraph = subtensor.metagraph(netuid=55)
        print(f"✅ Metagraph loaded: {len(metagraph.uids)} neurons")

        # Check our wallet
        wallet = bt.wallet(name='precog_coldkey', hotkey='miner_hotkey')
        print(f"✅ Wallet loaded: {wallet.hotkey.ss58_address}")

        # Check if we're registered
        is_registered = subtensor.is_hotkey_registered(
            hotkey_ss58=wallet.hotkey.ss58_address,
            netuid=55
        )
        print(f"✅ Registration status: {is_registered}")

        if is_registered:
            uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            print(f"🎯 Our UID: {uid}")
            print(f"🎯 Our stake: {metagraph.stake[uid]}")
            print(f"🎯 Our rank: {metagraph.ranks[uid]}")
        else:
            print("❌ Not registered on subnet 55")

        print("🎉 Bittensor connection test successful!")
        return True

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
