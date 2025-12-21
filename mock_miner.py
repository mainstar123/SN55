#!/usr/bin/env python3
"""
Mock Miner for Local Testing
A miner that uses bittensor's axon but with mocked blockchain components
"""

import asyncio
import time
import torch
import numpy as np
from datetime import datetime
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bittensor as bt
from precog.protocol import Challenge
from precog import constants


class MockMiner:
    """Miner with mocked bittensor components for local testing"""

    def __init__(self, port: int = 8092, model_path: str = None):
        self.port = port
        self.model_path = model_path or 'elite_domination_model.pth'
        self.request_count = 0
        self.start_time = time.time()

        # Load model
        self.load_model()

        print(f"🏭 Mock Miner initialized on port {port}")

    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location='cpu')
                print(f"✅ Loaded model from {self.model_path}")

                # Create mock model for inference
                self.model = type('MockModel', (), {
                    'predict': self._mock_predict
                })()
            else:
                print(f"⚠️  Model file {self.model_path} not found, using random predictions")
                self.model = type('MockModel', (), {
                    'predict': self._mock_predict
                })()
        except Exception as e:
            print(f"⚠️  Error loading model: {e}, using random predictions")
            self.model = type('MockModel', (), {
                'predict': self._mock_predict
            })()

    def _mock_predict(self, timestamp: str, assets: list) -> tuple:
        """Generate mock predictions"""
        predictions = {}
        intervals = {}

        # Parse timestamp for reproducible predictions
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            seed = int(dt.timestamp()) % 1000000
            np.random.seed(seed)
        except:
            np.random.seed(42)

        for asset in assets:
            # Generate realistic predictions
            if asset == "btc":
                base_price = 50000 + np.random.normal(0, 2000)
            elif asset == "eth":
                base_price = 3000 + np.random.normal(0, 200)
            elif asset == "tao_bittensor":
                base_price = 200 + np.random.normal(0, 20)
            else:
                base_price = 100 + np.random.normal(0, 10)

            # Add trend and noise
            trend = np.sin(time.time() * 0.001) * 0.05
            noise = np.random.normal(0, base_price * 0.02)
            prediction = base_price * (1 + trend) + noise

            predictions[asset] = float(prediction)

            # Generate intervals
            uncertainty = abs(prediction) * 0.1
            intervals[asset] = [
                float(prediction - uncertainty),
                float(prediction + uncertainty)
            ]

        return predictions, intervals

    async def forward(self, synapse: Challenge) -> Challenge:
        """Handle prediction requests"""
        self.request_count += 1

        print(f"\n📥 Request #{self.request_count} received")
        print(f"   Timestamp: {synapse.timestamp}")
        print(f"   Assets: {synapse.assets}")

        start_time = time.time()

        try:
            # Generate predictions
            predictions, intervals = self.model.predict(synapse.timestamp, synapse.assets)

            # Fill response
            synapse.predictions = predictions
            synapse.intervals = intervals

            processing_time = time.time() - start_time

            print(f"📤 Response sent in {processing_time:.3f}s")
            print(f"   Predictions: {predictions}")
            print(f"   Intervals: {intervals}")

            return synapse

        except Exception as e:
            print(f"❌ Error processing request: {e}")
            synapse.predictions = None
            synapse.intervals = None
            return synapse

    async def run(self):
        """Run the miner with mocked components"""
        print("🚀 Starting Mock Miner...")
        print("=" * 50)

        # Mock bittensor components
        mock_subtensor = MagicMock()
        mock_subtensor.get_current_block.return_value = 1000
        mock_subtensor.network = "mock"
        mock_subtensor.chain_endpoint = "mock://local"

        mock_metagraph = MagicMock()
        mock_metagraph.uids = [0]
        mock_metagraph.hotkeys = ["mock_hotkey"]
        mock_metagraph.axons = []
        mock_metagraph.S = [1.0]
        mock_metagraph.sync.return_value = None

        mock_wallet = MagicMock()
        mock_wallet.name = "mock_wallet"
        mock_wallet.hotkey.ss58_address = "mock_hotkey_address"
        mock_wallet.coldkey.ss58_address = "mock_coldkey_address"

        # Create axon
        axon = bt.axon(wallet=mock_wallet)
        axon.attach(forward_fn=self.forward)

        print(f"🖧 Axon configured on port {self.port}")
        print("🎯 Ready for validator connections...")

        try:
            # Serve the axon
            axon.serve(netuid=256, subtensor=mock_subtensor)
            axon.start()

            print("✅ Mock miner is running!")
            print("💡 Run 'python3 mock_validator.py' in another terminal to test")
            print()

            # Keep running
            while True:
                await asyncio.sleep(5)

                # Print stats every 30 seconds
                if int(time.time() - self.start_time) % 30 == 0 and self.request_count > 0:
                    uptime = time.time() - self.start_time
                    print(f"📊 Stats: {self.request_count} requests processed, {uptime:.0f}s uptime")

        except KeyboardInterrupt:
            print("\n⏹️  Mock miner stopped")
            axon.stop()

            uptime = time.time() - self.start_time
            print(f"📊 Final stats: {self.request_count} requests processed, {uptime:.0f}s uptime")

        except Exception as e:
            print(f"❌ Error running miner: {e}")
            import traceback
            traceback.print_exc()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mock Miner for Local Testing")
    parser.add_argument("--port", type=int, default=8092,
                       help="Port to run the miner on (default: 8092)")
    parser.add_argument("--model-path", type=str, default="elite_domination_model.pth",
                       help="Path to trained model file")

    args = parser.parse_args()

    # Create and run miner
    miner = MockMiner(port=args.port, model_path=args.model_path)
    await miner.run()


if __name__ == "__main__":
    asyncio.run(main())


