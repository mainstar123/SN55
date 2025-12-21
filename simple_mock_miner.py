#!/usr/bin/env python3
"""
Simple Mock Miner - HTTP-based version
No bittensor dependencies required
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime
import sys
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from precog.protocol import Challenge
from precog import constants


class MinerRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for miner"""

    def __init__(self, miner_instance, *args, **kwargs):
        self.miner = miner_instance
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        # Suppress default HTTP server logs
        pass

    def do_POST(self):
        """Handle POST requests (prediction requests)"""
        if self.path == '/predict':
            self.handle_prediction_request()
        else:
            self.send_error(404, "Not found")

    def handle_prediction_request(self):
        """Handle prediction request"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))

            # Create challenge synapse
            synapse = Challenge(
                timestamp=request_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                assets=request_data.get('assets', constants.SUPPORTED_ASSETS)
            )

            # Process request
            response_synapse = asyncio.run(self.miner.forward(synapse))

            # Send response
            response_data = {
                'predictions': response_synapse.predictions,
                'intervals': response_synapse.intervals
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")

    def do_GET(self):
        """Handle GET requests (health check)"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            health_data = {
                'status': 'healthy',
                'requests_processed': self.miner.request_count,
                'uptime': time.time() - self.miner.start_time
            }
            self.wfile.write(json.dumps(health_data).encode('utf-8'))
        else:
            self.send_error(404, "Not found")


class SimpleMockMiner:
    """Simple HTTP-based mock miner"""

    def __init__(self, port: int = 8092, model_path: str = None):
        self.port = port
        self.model_path = model_path or 'elite_domination_model.pth'
        self.request_count = 0
        self.start_time = time.time()
        self.server = None
        self.server_thread = None

        # Load model
        self.load_model()

        print(f"🏭 Simple Mock Miner initialized on port {port}")

    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                print(f"✅ Model file {self.model_path} found")
                self.model = type('MockModel', (), {
                    'predict': self._mock_predict
                })()
            else:
                print(f"⚠️  Model file {self.model_path} not found, using random predictions")
                self.model = type('MockModel', (), {
                    'predict': self._mock_predict
                })()
        except Exception as e:
            print(f"⚠️  Error setting up model: {e}, using random predictions")
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

    def run_server(self):
        """Run the HTTP server"""
        def create_handler(*args, **kwargs):
            return MinerRequestHandler(self, *args, **kwargs)

        self.server = HTTPServer(('127.0.0.1', self.port), create_handler)
        print("🚀 Starting HTTP server...")
        print("=" * 50)
        print(f"🖧 Server listening on http://127.0.0.1:{self.port}")
        print("🎯 Ready for validator connections...")
        print("💡 Run 'python3 simple_mock_validator.py' in another terminal to test")
        print()

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\n⏹️  Server stopped")
            self.server.shutdown()

    def start_in_thread(self):
        """Start server in a separate thread"""
        self.server_thread = threading.Thread(target=self.run_server, daemon=True)
        self.server_thread.start()

        # Wait a bit for server to start
        time.sleep(1)

        if self.server_thread.is_alive():
            print("✅ Server started in background thread")
            return True
        else:
            print("❌ Server failed to start")
            return False

    def stop(self):
        """Stop the server"""
        if self.server:
            print("⏹️  Stopping server...")
            self.server.shutdown()
            self.server.server_close()

        if self.server_thread:
            self.server_thread.join(timeout=5)

        uptime = time.time() - self.start_time
        print(f"📊 Final stats: {self.request_count} requests processed, {uptime:.0f}s uptime")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple Mock Miner (HTTP-based)")
    parser.add_argument("--port", type=int, default=8092,
                       help="Port to run the miner on (default: 8092)")
    parser.add_argument("--model-path", type=str, default="elite_domination_model.pth",
                       help="Path to trained model file")

    args = parser.parse_args()

    # Create and run miner
    miner = SimpleMockMiner(port=args.port, model_path=args.model_path)

    try:
        miner.run_server()
    except KeyboardInterrupt:
        miner.stop()


if __name__ == "__main__":
    main()


