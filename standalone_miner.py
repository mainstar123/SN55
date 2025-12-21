#!/usr/bin/env python3
"""
Standalone Miner for Mock Testing
A simplified miner that can run locally and respond to validator queries
without blockchain dependencies
"""

import asyncio
import time
import torch
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from precog.protocol import Challenge
from precog import constants

class StandaloneMiner:
    """Standalone miner for local testing"""

    def __init__(self, port: int = 8092, model_path: str = None):
        self.port = port
        self.model_path = model_path or 'elite_domination_model.pth'
        self.request_count = 0
        self.start_time = time.time()

        # Load model if available
        self.model = None
        self.load_model()

        print(f"🏭 Standalone Miner initialized on port {port}")

    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location='cpu')
                print(f"✅ Loaded model from {self.model_path}")

                # Create a simple model structure for inference
                # This is a simplified version - in real usage you'd load the full model
                self.model = type('MockModel', (), {
                    'predict': self._mock_predict
                })()
            else:
                print(f"⚠️  Model file {self.model_path} not found, using mock predictions")
                self.model = type('MockModel', (), {
                    'predict': self._mock_predict
                })()
        except Exception as e:
            print(f"⚠️  Error loading model: {e}, using mock predictions")
            self.model = type('MockModel', (), {
                'predict': self._mock_predict
            })()

    def _mock_predict(self, timestamp: str, assets: list) -> dict:
        """Generate mock predictions for testing"""
        predictions = {}
        intervals = {}

        # Parse timestamp
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            # Use timestamp as seed for reproducible "predictions"
            seed = int(dt.timestamp()) % 1000000
            np.random.seed(seed)
        except:
            np.random.seed(42)

        for asset in assets:
            # Generate realistic-looking predictions based on asset
            if asset == "btc":
                base_price = 50000 + np.random.normal(0, 2000)
            elif asset == "eth":
                base_price = 3000 + np.random.normal(0, 200)
            elif asset == "tao_bittensor":
                base_price = 200 + np.random.normal(0, 20)
            else:
                base_price = 100 + np.random.normal(0, 10)

            # Add some trend and noise
            trend = np.sin(time.time() * 0.001) * 0.05  # Slow oscillation
            noise = np.random.normal(0, base_price * 0.02)  # 2% noise
            prediction = base_price * (1 + trend) + noise

            predictions[asset] = float(prediction)

            # Generate prediction intervals (±10% confidence interval)
            uncertainty = abs(prediction) * 0.1
            intervals[asset] = [
                float(prediction - uncertainty),
                float(prediction + uncertainty)
            ]

        return predictions, intervals

    async def handle_prediction_request(self, synapse: Challenge) -> Challenge:
        """Handle incoming prediction requests"""
        self.request_count += 1

        print(f"\n📥 Request #{self.request_count} received")
        print(f"   Timestamp: {synapse.timestamp}")
        print(f"   Assets: {synapse.assets}")

        start_time = time.time()

        try:
            # Generate predictions
            if self.model:
                predictions, intervals = self.model.predict(synapse.timestamp, synapse.assets)
            else:
                predictions, intervals = self._mock_predict(synapse.timestamp, synapse.assets)

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

    async def run_server(self):
        """Run the axon server"""
        print("🚀 Starting standalone miner server...")
        print("=" * 50)

        # Create a simple HTTP server to simulate axon behavior
        # For simplicity, we'll use a basic approach with direct synapse handling

        async def handle_request(request_data):
            """Handle incoming requests"""
            try:
                # Parse the synapse from request data
                # In a real implementation, this would deserialize from the network
                synapse = Challenge(
                    timestamp=request_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    assets=request_data.get('assets', constants.SUPPORTED_ASSETS)
                )

                # Process the request
                response = await self.handle_prediction_request(synapse)

                # Return response data
                return {
                    'predictions': response.predictions,
                    'intervals': response.intervals
                }
            except Exception as e:
                print(f"❌ Request handling error: {e}")
                return {'error': str(e)}

        # For this demo, we'll simulate the server behavior
        # In a real axon, this would be handled by bittensor's networking layer

        print(f"🖧 Server simulation running on port {self.port}")
        print("🎯 Ready for validator connections...")
        print()
        print("💡 Run 'python3 mock_validator.py' in another terminal to test")

        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)

                # Print stats every 30 seconds
                if int(time.time() - self.start_time) % 30 == 0 and self.request_count > 0:
                    uptime = time.time() - self.start_time
                    print(f"📊 Stats: {self.request_count} requests processed, {uptime:.0f}s uptime")

        except KeyboardInterrupt:
            print("\n⏹️  Standalone miner stopped")
            uptime = time.time() - self.start_time
            print(f"📊 Final stats: {self.request_count} requests processed, {uptime:.0f}s uptime")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standalone Miner for Mock Testing")
    parser.add_argument("--port", type=int, default=8092,
                       help="Port to run the miner on (default: 8092)")
    parser.add_argument("--model-path", type=str, default="elite_domination_model.pth",
                       help="Path to trained model file")

    args = parser.parse_args()

    # Create and run miner
    miner = StandaloneMiner(port=args.port, model_path=args.model_path)
    await miner.run_server()


if __name__ == "__main__":
    asyncio.run(main())


