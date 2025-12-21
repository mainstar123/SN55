#!/usr/bin/env python3
"""
Mock Validator for Precog - Tests miner communication locally
Simulates validator behavior without blockchain dependencies
"""

import asyncio
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bittensor as bt
from precog.protocol import Challenge
from precog import constants


class MockValidator:
    """Mock validator that simulates querying miners locally"""

    def __init__(self, miner_host: str = "127.0.0.1", miner_port: int = 8092, assets: List[str] = None):
        self.miner_host = miner_host
        self.miner_port = miner_port
        self.assets = assets or constants.SUPPORTED_ASSETS
        self.query_count = 0
        self.responses_received = 0

        # Create a mock dendrite for local communication
        self.dendrite = bt.dendrite(wallet=None)  # No wallet needed for local testing

        print(f"🚀 Mock Validator initialized")
        print(f"   Target miner: {miner_host}:{miner_port}")
        print(f"   Assets to query: {self.assets}")

    async def create_mock_axon(self, host: str, port: int) -> bt.Axon:
        """Create a mock axon for the miner"""
        axon = bt.Axon(wallet=None)  # No wallet for local testing
        axon.ip = host
        axon.port = port
        axon.hotkey = "mock_miner_hotkey"
        return axon

    async def query_miner(self) -> Dict[str, Any]:
        """Send a prediction request to the miner"""
        self.query_count += 1

        # Create timestamp for current time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create challenge synapse
        synapse = Challenge(
            timestamp=timestamp,
            assets=self.assets
        )

        print(f"\n📤 Query #{self.query_count} - Requesting predictions for {self.assets}")
        print(f"   Timestamp: {timestamp}")

        try:
            # Create mock axon for the miner
            miner_axon = await self.create_mock_axon(self.miner_host, self.miner_port)

            # Send the query
            start_time = time.time()
            responses = await self.dendrite.forward(
                axons=[miner_axon],
                synapse=synapse,
                deserialize=False,
                timeout=30.0,  # 30 second timeout
            )
            end_time = time.time()

            response_time = end_time - start_time

            if responses and len(responses) > 0:
                response = responses[0]  # Get first (only) response
                self.responses_received += 1

                print(f"✅ Response received in {response_time:.2f}s")
                print(f"   Predictions: {response.predictions}")
                print(f"   Intervals: {response.intervals}")

                # Validate response
                self.validate_response(response, synapse)

                return {
                    'success': True,
                    'response': response,
                    'response_time': response_time,
                    'timestamp': timestamp
                }
            else:
                print("❌ No response received")
                return {
                    'success': False,
                    'error': 'No response',
                    'response_time': response_time,
                    'timestamp': timestamp
                }

        except Exception as e:
            print(f"❌ Query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': timestamp
            }

    def validate_response(self, response: Challenge, request: Challenge):
        """Validate that the response is properly formatted"""
        print("🔍 Validating response...")

        # Check if predictions exist
        if response.predictions is None:
            print("   ⚠️  No predictions in response")
            return False

        # Check if all requested assets have predictions
        missing_assets = []
        for asset in request.assets:
            if asset not in response.predictions:
                missing_assets.append(asset)

        if missing_assets:
            print(f"   ⚠️  Missing predictions for assets: {missing_assets}")
            return False

        # Check prediction values are reasonable (not None, not extreme)
        for asset, prediction in response.predictions.items():
            if prediction is None:
                print(f"   ⚠️  Prediction for {asset} is None")
                continue

            # Basic sanity check - predictions should be reasonable numbers
            if not isinstance(prediction, (int, float)):
                print(f"   ⚠️  Prediction for {asset} is not a number: {type(prediction)}")
            elif abs(prediction) > 1000000:  # Extremely large values
                print(f"   ⚠️  Prediction for {asset} seems extreme: {prediction}")

        # Check intervals if present
        if response.intervals:
            for asset, interval in response.intervals.items():
                if interval and len(interval) == 2:
                    min_val, max_val = interval
                    if min_val > max_val:
                        print(f"   ⚠️  Invalid interval for {asset}: min > max ({min_val} > {max_val})")

        print("   ✅ Response validation complete")
        return True

    async def run_continuous_queries(self, interval_seconds: int = 30, max_queries: int = None):
        """Run continuous queries to the miner"""
        print("
🔄 Starting continuous query mode"        print(f"   Query interval: {interval_seconds}s")
        if max_queries:
            print(f"   Max queries: {max_queries}")
        print("   Press Ctrl+C to stop\n")

        query_num = 0
        results = []

        try:
            while True:
                if max_queries and query_num >= max_queries:
                    break

                result = await self.query_miner()
                results.append(result)
                query_num += 1

                # Print stats every 10 queries
                if query_num % 10 == 0:
                    success_rate = sum(1 for r in results[-10:] if r['success']) / 10
                    avg_response_time = sum(r.get('response_time', 0) for r in results[-10:] if r.get('response_time')) / 10
                    print(f"\n📊 Last 10 queries: {success_rate:.1%} success rate, {avg_response_time:.2f}s avg response time")

                if query_num < max_queries:  # Don't sleep after last query
                    await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n⏹️  Continuous queries stopped by user")

        # Print final statistics
        total_queries = len(results)
        successful_queries = sum(1 for r in results if r['success'])
        success_rate = successful_queries / total_queries if total_queries > 0 else 0

        response_times = [r.get('response_time', 0) for r in results if r.get('response_time')]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        print("
📈 FINAL STATISTICS:"        print(f"   Total queries: {total_queries}")
        print(f"   Successful: {successful_queries}")
        print(".1%"        print(".2f"        print(".1f"
    async def run_single_query(self):
        """Run a single query and exit"""
        await self.query_miner()


async def main():
    parser = argparse.ArgumentParser(description="Mock Validator for Precog Testing")
    parser.add_argument("--miner-host", type=str, default="127.0.0.1",
                       help="Miner host (default: 127.0.0.1)")
    parser.add_argument("--miner-port", type=int, default=8092,
                       help="Miner port (default: 8092)")
    parser.add_argument("--assets", type=str, nargs="+",
                       default=constants.SUPPORTED_ASSETS,
                       help=f"Assets to query (default: {constants.SUPPORTED_ASSETS})")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuous queries instead of single query")
    parser.add_argument("--interval", type=int, default=30,
                       help="Query interval in seconds for continuous mode (default: 30)")
    parser.add_argument("--max-queries", type=int,
                       help="Maximum number of queries for continuous mode (default: unlimited)")

    args = parser.parse_args()

    # Create mock validator
    validator = MockValidator(
        miner_host=args.miner_host,
        miner_port=args.miner_port,
        assets=args.assets
    )

    # Run queries
    if args.continuous:
        await validator.run_continuous_queries(
            interval_seconds=args.interval,
            max_queries=args.max_queries
        )
    else:
        await validator.run_single_query()


if __name__ == "__main__":
    asyncio.run(main())


