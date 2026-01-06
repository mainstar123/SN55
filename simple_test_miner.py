#!/usr/bin/env python3
"""Simple test miner to verify Bittensor integration works"""

import bittensor as bt
import torch
import time
import sys
import os

class SimpleTestMiner(bt.MinerNeuron):
    """Simple test miner that returns basic predictions"""

    def __init__(self, config):
        super().__init__(config)
        self.step_count = 0

    def forward(self, synapse):
        """Simple forward pass that returns basic predictions"""
        try:
            self.step_count += 1
            bt.logging.info(f"🔄 Processing request #{self.step_count}")

            # Get assets to predict
            assets = synapse.assets if hasattr(synapse, "assets") and synapse.assets else ["btc"]

            predictions = {}
            intervals = {}

            for asset in assets:
                # Simple prediction: current price + small random change
                if asset.lower() == 'btc':
                    base_price = 95000  # Approximate BTC price
                elif asset.lower() == 'eth':
                    base_price = 3100   # Approximate ETH price
                elif asset.lower() == 'tao':
                    base_price = 290    # Approximate TAO price
                else:
                    base_price = 100

                # Add small random change (-1% to +1%)
                import random
                change_pct = (random.random() - 0.5) * 0.02  # -1% to +1%
                prediction = base_price * (1 + change_pct)

                # Convert to TAO units (rough approximation)
                tao_prediction = prediction * 1000 if asset.lower() != 'tao' else prediction

                predictions[asset] = tao_prediction

                # Simple interval (±2% of prediction)
                interval_width = abs(tao_prediction * 0.02)
                lower_bound = tao_prediction - interval_width
                upper_bound = tao_prediction + interval_width

                intervals[asset] = [lower_bound, upper_bound]

                bt.logging.info(f"🎯 {asset.upper()}: {tao_prediction:.2f} TAO")
            return predictions, intervals

        except Exception as e:
            bt.logging.error(f"❌ Forward pass error: {e}")
            # Return safe fallback predictions
            predictions = {asset: 100.0 for asset in (synapse.assets if hasattr(synapse, "assets") and synapse.assets else ["btc"])}
            intervals = {asset: [90.0, 110.0] for asset in predictions.keys()}
            return predictions, intervals

def main():
    """Main miner function"""
    bt.logging.info("🚀 Starting Simple Test Miner for Precog Subnet 55")

    # Parse command line arguments
    config = bt.MinerNeuron.config()
    config.netuid = 55
    config.subtensor.network = 'finney'
    config.subtensor.chain_endpoint = 'wss://entrypoint-finney.opentensor.ai:443'
    config.wallet.name = 'precog_coldkey'
    config.wallet.hotkey = 'miner_hotkey'
    config.axon.port = 8091
    config.logging.level = 'INFO'

    # Create and run miner
    miner = SimpleTestMiner(config)
    bt.logging.info("✅ Simple Test Miner initialized")

    try:
        miner.run()
    except KeyboardInterrupt:
        bt.logging.info("🛑 Miner stopped by user")
    except Exception as e:
        bt.logging.error(f"❌ Miner crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
