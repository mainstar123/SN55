#!/usr/bin/env python3
"""
Standalone Mock Validator - Completely independent version
No external dependencies required
"""

import asyncio
import json
import time
import requests
from datetime import datetime
import sys
import os


class StandaloneMockValidator:
    """Standalone HTTP-based mock validator with no external dependencies"""

    def __init__(self, miner_host: str = "127.0.0.1", miner_port: int = 8092, assets: list = None):
        self.miner_host = miner_host
        self.miner_port = miner_port
        self.miner_url = f"http://{miner_host}:{miner_port}"
        self.assets = assets or ["btc", "eth", "tao_bittensor"]
        self.query_count = 0
        self.responses_received = 0

        print(f"🔍 Standalone Mock Validator initialized")
        print(f"   Target miner: {self.miner_url}")
        print(f"   Assets to query: {self.assets}")

    def query_miner(self) -> dict:
        """Send a prediction request to the miner"""
        self.query_count += 1

        # Create timestamp for current time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create request data
        request_data = {
            'timestamp': timestamp,
            'assets': self.assets
        }

        print(f"\n📤 Query #{self.query_count} - Requesting predictions for {self.assets}")
        print(f"   Timestamp: {timestamp}")

        try:
            # Send HTTP POST request
            start_time = time.time()
            response = requests.post(
                f"{self.miner_url}/predict",
                json=request_data,
                timeout=30.0
            )
            end_time = time.time()

            response_time = end_time - start_time

            if response.status_code == 200:
                response_data = response.json()
                self.responses_received += 1

                print(f"✅ Response received in {response_time:.2f}s")
                print(f"   Predictions: {response_data.get('predictions')}")
                print(f"   Intervals: {response_data.get('intervals')}")

                # Validate response
                validation_result = self.validate_response(response_data, request_data)

                return {
                    'success': True,
                    'response': response_data,
                    'response_time': response_time,
                    'timestamp': timestamp,
                    'validation': validation_result
                }
            else:
                print(f"❌ HTTP error: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}',
                    'response_time': response_time,
                    'timestamp': timestamp
                }

        except requests.exceptions.Timeout:
            print("❌ Request timeout")
            return {
                'success': False,
                'error': 'Timeout',
                'timestamp': timestamp
            }
        except requests.exceptions.ConnectionError:
            print("❌ Connection refused - is miner running?")
            return {
                'success': False,
                'error': 'Connection refused',
                'timestamp': timestamp
            }
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': timestamp
            }

    def validate_response(self, response_data: dict, request_data: dict) -> dict:
        """Validate that the response is properly formatted"""
        print("🔍 Validating response...")

        validation = {
            'valid': True,
            'issues': []
        }

        predictions = response_data.get('predictions')
        intervals = response_data.get('intervals')
        requested_assets = request_data.get('assets', [])

        # Check if predictions exist
        if predictions is None:
            validation['valid'] = False
            validation['issues'].append("No predictions in response")
            print("   ⚠️  No predictions in response")
            return validation

        # Check if all requested assets have predictions
        missing_assets = []
        for asset in requested_assets:
            if asset not in predictions:
                missing_assets.append(asset)

        if missing_assets:
            validation['valid'] = False
            validation['issues'].append(f"Missing predictions for assets: {missing_assets}")
            print(f"   ⚠️  Missing predictions for assets: {missing_assets}")
            return validation

        # Check prediction values are reasonable
        for asset, prediction in predictions.items():
            if prediction is None:
                validation['issues'].append(f"Prediction for {asset} is None")
                print(f"   ⚠️  Prediction for {asset} is None")
                continue

            if not isinstance(prediction, (int, float)):
                validation['valid'] = False
                validation['issues'].append(f"Prediction for {asset} is not a number: {type(prediction)}")
                print(f"   ⚠️  Prediction for {asset} is not a number: {type(prediction)}")
            elif abs(prediction) > 1000000:  # Extremely large values
                validation['issues'].append(f"Prediction for {asset} seems extreme: {prediction}")
                print(f"   ⚠️  Prediction for {asset} seems extreme: {prediction}")

        # Check intervals if present
        if intervals:
            for asset, interval in intervals.items():
                if interval and len(interval) == 2:
                    min_val, max_val = interval
                    if min_val > max_val:
                        validation['issues'].append(f"Invalid interval for {asset}: min > max ({min_val} > {max_val})")
                        print(f"   ⚠️  Invalid interval for {asset}: min > max ({min_val} > {max_val})")
                else:
                    validation['issues'].append(f"Invalid interval format for {asset}: {interval}")

        if validation['issues']:
            validation['valid'] = False

        if validation['valid']:
            print("   ✅ Response validation complete")
        else:
            print(f"   ❌ Response validation failed: {len(validation['issues'])} issues")

        return validation

    async def run_continuous_queries(self, interval_seconds: int = 30, max_queries: int = None):
        """Run continuous queries to the miner"""
        print("\n🔄 Starting continuous query mode")
        print(f"   Query interval: {interval_seconds}s")
        if max_queries:
            print(f"   Max queries: {max_queries}")
        print("   Press Ctrl+C to stop\n")

        query_num = 0
        results = []

        try:
            while True:
                if max_queries and query_num >= max_queries:
                    break

                result = self.query_miner()
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

        print("\n📈 FINAL STATISTICS:")
        print(f"   Total queries: {total_queries}")
        print(f"   Successful: {successful_queries}")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Average response time: {avg_response_time:.2f}s")
        print(f"   Total uptime: {time.time() - self.start_time:.1f}s")
    async def run_single_query(self):
        """Run a single query and exit"""
        self.query_miner()

    def check_miner_health(self):
        """Check if miner is healthy"""
        try:
            response = requests.get(f"{self.miner_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Miner health check passed")
                print(f"   Status: {health_data.get('status')}")
                print(f"   Requests processed: {health_data.get('requests_processed')}")
                print(f"   Uptime: {health_data.get('uptime', 0):.0f}s")
                return True
            else:
                print(f"❌ Miner health check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Miner health check failed: {e}")
            return False


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standalone Mock Validator (No dependencies)")
    parser.add_argument("--miner-host", type=str, default="127.0.0.1",
                       help="Miner host (default: 127.0.0.1)")
    parser.add_argument("--miner-port", type=int, default=8092,
                       help="Miner port (default: 8092)")
    parser.add_argument("--assets", type=str, nargs="+",
                       default=["btc", "eth", "tao_bittensor"],
                       help="Assets to query (default: btc eth tao_bittensor)")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuous queries instead of single query")
    parser.add_argument("--interval", type=int, default=30,
                       help="Query interval in seconds for continuous mode (default: 30)")
    parser.add_argument("--max-queries", type=int,
                       help="Maximum number of queries for continuous mode (default: unlimited)")
    parser.add_argument("--health-check", action="store_true",
                       help="Only perform health check and exit")

    args = parser.parse_args()

    # Create validator
    validator = StandaloneMockValidator(
        miner_host=args.miner_host,
        miner_port=args.miner_port,
        assets=args.assets
    )

    # Health check only
    if args.health_check:
        validator.check_miner_health()
        return

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
