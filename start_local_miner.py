#!/usr/bin/env python3
"""
Local Miner Setup for Mock Testing
Starts a miner that can communicate with mock validator without blockchain
"""

import sys
import os
import argparse
import asyncio
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock bittensor components for local testing
class MockSubtensor:
    def __init__(self):
        self.network = "local"
        self.chain_endpoint = "mock://local"

    def get_current_block(self):
        return 1000

    def is_hotkey_registered(self, **kwargs):
        return True

    def get_total_stake_for_hotkey(self, hotkey):
        return 100.0

    def get_balance(self, address):
        return type('Balance', (), {'tao': 50.0})()

class MockMetagraph:
    def __init__(self):
        self.uids = [0]
        self.hotkeys = ["mock_hotkey"]
        self.axons = []
        self.S = [1.0]

    def sync(self, subtensor=None):
        pass

class MockWallet:
    def __init__(self):
        self.name = "mock_wallet"
        self.hotkey = type('Hotkey', (), {
            'ss58_address': 'mock_hotkey_address',
            'sign': lambda x: b'mock_signature'
        })()
        self.coldkey = type('Coldkey', (), {
            'ss58_address': 'mock_coldkey_address'
        })()

class MockAxon:
    def __init__(self, port=8092):
        self.port = port
        self.ip = "127.0.0.1"
        self.hotkey = "mock_hotkey"

    def serve(self, netuid=None, subtensor=None):
        print(f"🖧 Mock axon serving on port {self.port}")

    def start(self):
        print("🚀 Mock axon started")

    def stop(self):
        print("⏹️  Mock axon stopped")

def mock_bittensor_setup():
    """Apply mocks to bittensor components"""
    # Mock the main bittensor components
    with patch('bittensor.subtensor', return_value=MockSubtensor()), \
         patch('bittensor.metagraph', return_value=MockMetagraph()), \
         patch('bittensor.wallet', return_value=MockWallet()), \
         patch('bittensor.axon', return_value=MockAxon()), \
         patch('bittensor.dendrite') as mock_dendrite:

        # Import and run the miner
        from precog.miners.miner import Miner

        # Create mock config
        config = type('Config', (), {
            'neuron': type('Neuron', (), {
                'type': 'Miner',
                'name': 'local_miner'
            })(),
            'subtensor': type('Subtensor', (), {
                'chain_endpoint': 'mock://local',
                'network': 'local'
            })(),
            'netuid': 256,
            'axon': type('Axon', (), {
                'port': 8092
            })(),
            'wallet': type('Wallet', (), {
                'name': 'mock_wallet',
                'hotkey': 'default'
            })(),
            'forward_function': 'custom_model',
            'print_cadence': 30,
            'logging': type('Logging', (), {
                'level': 'info'
            })()
        })()

        print("🏭 Starting Local Mock Miner...")
        print("=" * 50)
        print(f"Port: {config.axon.port}")
        print(f"Forward function: {config.forward_function}")
        print()

        try:
            miner = Miner(config=config)
            print("✅ Local miner initialized successfully")
            print("🎯 Ready to receive prediction requests from mock validator")
            print()
            print("💡 To test: Run 'python3 mock_validator.py' in another terminal")

        except KeyboardInterrupt:
            print("\n⏹️  Local miner stopped by user")
        except Exception as e:
            print(f"❌ Error starting local miner: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Start Local Mock Miner")
    parser.add_argument("--port", type=int, default=8092,
                       help="Port for the miner to listen on (default: 8092)")
    parser.add_argument("--forward-function", type=str, default="custom_model",
                       help="Forward function to use (default: custom_model)")

    args = parser.parse_args()

    print("🚀 PRECOG LOCAL MOCK MINER")
    print("=" * 50)
    print(f"Port: {args.port}")
    print(f"Forward Function: {args.forward_function}")
    print()

    # Apply the mocks and start miner
    mock_bittensor_setup()

if __name__ == "__main__":
    main()


