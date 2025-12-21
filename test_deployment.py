#!/usr/bin/env python3
"""
Test Deployment Script
Tests the complete validator-miner communication flow
"""

import asyncio
import subprocess
import time
import signal
import sys
import os
from datetime import datetime


class DeploymentTester:
    """Test the validator-miner deployment"""

    def __init__(self):
        self.miner_process = None
        self.validator_process = None
        self.start_time = time.time()

    async def start_miner(self):
        """Start the mock miner"""
        print("🏭 Starting mock miner...")

        # Start miner in background
        self.miner_process = subprocess.Popen([
            sys.executable, "standalone_mock_miner.py",
            "--port", "8092"
        ], cwd=os.getcwd())

        # Wait for miner to start
        await asyncio.sleep(3)

        if self.miner_process.poll() is None:
            print("✅ Miner started successfully")
            return True
        else:
            print("❌ Miner failed to start")
            return False

    async def start_validator(self, continuous=False, max_queries=5):
        """Start the mock validator"""
        print("🔍 Starting mock validator...")

        cmd = [sys.executable, "standalone_mock_validator.py",
               "--miner-host", "127.0.0.1",
               "--miner-port", "8092"]

        if continuous:
            cmd.extend(["--continuous", "--interval", "5", "--max-queries", str(max_queries)])
        else:
            # Single query test
            pass

        # Start validator
        self.validator_process = subprocess.Popen(cmd, cwd=os.getcwd())

        # Wait a bit for validator to complete
        if continuous:
            await asyncio.sleep(max_queries * 6)  # Wait for queries + some buffer
        else:
            await asyncio.sleep(10)  # Wait for single query

        return self.validator_process.poll() == 0

    def stop_processes(self):
        """Stop all running processes"""
        print("\n⏹️  Stopping processes...")

        if self.miner_process and self.miner_process.poll() is None:
            self.miner_process.terminate()
            try:
                self.miner_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.miner_process.kill()

        if self.validator_process and self.validator_process.poll() is None:
            self.validator_process.terminate()
            try:
                self.validator_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.validator_process.kill()

    async def run_single_test(self):
        """Run a single validator-miner test"""
        print("🧪 RUNNING SINGLE TEST")
        print("=" * 50)

        # Start miner
        miner_started = await self.start_miner()
        if not miner_started:
            return False

        # Start validator for single query
        validator_success = await self.start_validator(continuous=False)

        # Check results
        success = validator_success and self.miner_process.poll() == 0

        if success:
            print("✅ Single test completed successfully")
        else:
            print("❌ Single test failed")

        return success

    async def run_continuous_test(self, num_queries=10):
        """Run a continuous validator-miner test"""
        print(f"🔄 RUNNING CONTINUOUS TEST ({num_queries} queries)")
        print("=" * 50)

        # Start miner
        miner_started = await self.start_miner()
        if not miner_started:
            return False

        # Start validator for continuous queries
        validator_success = await self.start_validator(continuous=True, max_queries=num_queries)

        # Check results
        success = validator_success and self.miner_process.poll() == 0

        if success:
            print(f"✅ Continuous test completed successfully ({num_queries} queries)")
        else:
            print(f"❌ Continuous test failed")

        return success

    async def run_full_test_suite(self):
        """Run the complete test suite"""
        print("🚀 PRECOG MOCK DEPLOYMENT TEST SUITE")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
        print()

        results = []

        try:
            # Test 1: Single query
            print("Test 1: Single Query Test")
            single_success = await self.run_single_test()
            results.append(("Single Query", single_success))

            # Wait a bit between tests
            await asyncio.sleep(2)

            # Test 2: Continuous queries
            print("\nTest 2: Continuous Query Test")
            continuous_success = await self.run_continuous_test(num_queries=5)
            results.append(("Continuous Queries", continuous_success))

        finally:
            # Cleanup
            self.stop_processes()

        # Print results
        print("\n📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        passed = 0
        total = len(results)

        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name}: {status}")
            if success:
                passed += 1

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed! Validator-miner communication is working.")
            return True
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
            return False


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Validator-Miner Deployment")
    parser.add_argument("--single", action="store_true",
                       help="Run only single query test")
    parser.add_argument("--continuous", action="store_true",
                       help="Run only continuous query test")
    parser.add_argument("--queries", type=int, default=5,
                       help="Number of queries for continuous test")

    args = parser.parse_args()

    tester = DeploymentTester()

    try:
        if args.single:
            success = await tester.run_single_test()
        elif args.continuous:
            success = await tester.run_continuous_test(args.queries)
        else:
            success = await tester.run_full_test_suite()

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        tester.stop_processes()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        tester.stop_processes()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
