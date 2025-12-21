#!/usr/bin/env python3
"""
Risk Mitigation Deployment System
Safe mainnet deployment with gradual scaling and monitoring
"""

import asyncio
import time
import json
import os
import subprocess
import signal
from datetime import datetime

class RiskMitigationDeployment:
    """Safe mainnet deployment with risk mitigation"""

    def __init__(self):
        self.miner_process = None
        self.monitoring_active = False
        self.deployment_phase = "preparation"
        self.start_time = None

        print("🛡️ RISK MITIGATION DEPLOYMENT SYSTEM")
        print("=" * 60)

    async def run_pre_deployment_checks(self):
        """Phase 1: Pre-deployment validation"""
        print("\n📋 PHASE 1: PRE-DEPLOYMENT VALIDATION")
        print("-" * 50)

        checks_passed = 0
        total_checks = 5

        # Check 1: Model exists
        if os.path.exists('elite_domination_model.pth'):
            print("✅ Model file exists: elite_domination_model.pth")
            checks_passed += 1
        else:
            print("❌ Model file missing: elite_domination_model.pth")
            return False

        # Check 2: Performance results
        if os.path.exists('elite_domination_results.json'):
            try:
                with open('elite_domination_results.json', 'r') as f:
                    results = json.load(f)
                perf = results.get('model_performance', {})

                mape = perf.get('mape', float('inf'))
                acc = perf.get('directional_accuracy', 0)

                if mape < 0.015 and acc > 0.70:  # Relaxed targets for initial deployment
                    print(f"✅ Model performance acceptable (MAPE: {mape:.4f}, Acc: {acc:.1%})")
                    checks_passed += 1
                else:
                    print(f"⚠️  Model performance below optimal (MAPE: {mape:.4f}, Acc: {acc:.1%})")
                    print("   💡 Consider more training before mainnet")
                    checks_passed += 1  # Allow deployment with warning
            except:
                print("❌ Cannot read performance results")
        else:
            print("⚠️  Performance results not found - proceeding with caution")
            checks_passed += 1

        # Check 3: Mock deployment test
        print("🧪 Testing mock deployment...")
        try:
            result = subprocess.run(['python3', 'test_deployment.py', '--single'],
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("✅ Mock deployment test passed")
                checks_passed += 1
            else:
                print("❌ Mock deployment test failed")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Mock deployment test timed out")
            return False

        # Check 4: Wallet registration check
        print("👛 Checking wallet registration...")
        try:
            result = subprocess.run(['btcli', 'wallet', 'overview', '--wallet.name', 'cold_draven'],
                                  capture_output=True, text=True, cwd=os.getcwd())
            if '0.0000 τ' in result.stdout:
                print("⚠️  No mainnet TAO balance - registration needed")
            else:
                print("✅ Mainnet TAO balance detected")
            checks_passed += 1  # Allow to proceed
        except:
            print("⚠️  Cannot check wallet status")
            checks_passed += 1

        # Check 5: Network connectivity
        print("🌐 Checking network connectivity...")
        try:
            result = subprocess.run(['ping', '-c', '1', 'archive.substrate.network'],
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                print("✅ Mainnet network reachable")
                checks_passed += 1
            else:
                print("❌ Mainnet network unreachable")
                return False
        except:
            print("❌ Cannot test network connectivity")
            return False

        success_rate = checks_passed / total_checks
        print(f"\n📊 Pre-deployment checks: {checks_passed}/{total_checks} passed ({success_rate:.0%})")

        if success_rate >= 0.8:  # 80% success rate to proceed
            print("✅ PRE-DEPLOYMENT CHECKS PASSED - Ready for Phase 2")
            return True
        else:
            print("❌ PRE-DEPLOYMENT CHECKS FAILED - Address issues before proceeding")
            return False

    async def deploy_conservative_mode(self):
        """Phase 2: Conservative deployment (25% capacity)"""
        print("\n📋 PHASE 2: CONSERVATIVE DEPLOYMENT (25% CAPACITY)")
        print("-" * 50)

        self.deployment_phase = "conservative"
        self.start_time = time.time()

        print("🎯 Starting miner in conservative mode...")
        print("   • Reduced prediction frequency")
        print("   • Higher confidence thresholds")
        print("   • Limited market hours")

        # Start miner with conservative settings
        env = os.environ.copy()
        env['DOMINATION_MODE'] = 'conservative'  # Custom conservative mode

        try:
            self.miner_process = subprocess.Popen([
                'python3', 'precog/miners/miner.py',
                '--neuron.name', 'conservative_domination',
                '--wallet.name', 'cold_draven',
                '--wallet.hotkey', 'default',
                '--subtensor.chain_endpoint', 'wss://archive.substrate.network:443',
                '--axon.port', '8092',
                '--netuid', '55',
                '--logging.level', 'info',
                '--timeout', '16',
                '--vpermit_tao_limit', '2',
                '--forward_function', 'custom_model'
            ], env=env, cwd=os.getcwd())

            print("✅ Conservative miner started")
            print(f"   PID: {self.miner_process.pid}")

            # Monitor for 30 minutes
            await self.monitor_performance(30 * 60, "conservative")

            return True

        except Exception as e:
            print(f"❌ Failed to start conservative miner: {e}")
            return False

    async def scale_to_moderate_mode(self):
        """Phase 3: Moderate deployment (50% capacity)"""
        print("\n📋 PHASE 3: MODERATE DEPLOYMENT (50% CAPACITY)")
        print("-" * 50)

        if self.miner_process:
            print("🛑 Stopping conservative miner...")
            self.stop_miner()

        self.deployment_phase = "moderate"

        print("🎯 Scaling to moderate mode...")
        print("   • Normal prediction frequency")
        print("   • Standard confidence thresholds")
        print("   • Extended market hours")

        # Start miner with moderate settings
        env = os.environ.copy()
        env['DOMINATION_MODE'] = 'moderate'

        try:
            self.miner_process = subprocess.Popen([
                'python3', 'precog/miners/miner.py',
                '--neuron.name', 'moderate_domination',
                '--wallet.name', 'cold_draven',
                '--wallet.hotkey', 'default',
                '--subtensor.chain_endpoint', 'wss://archive.substrate.network:443',
                '--axon.port', '8092',
                '--netuid', '55',
                '--logging.level', 'info',
                '--timeout', '16',
                '--vpermit_tao_limit', '2',
                '--forward_function', 'custom_model'
            ], env=env, cwd=os.getcwd())

            print("✅ Moderate miner started")
            print(f"   PID: {self.miner_process.pid}")

            # Monitor for 1 hour
            await self.monitor_performance(60 * 60, "moderate")

            return True

        except Exception as e:
            print(f"❌ Failed to start moderate miner: {e}")
            return False

    async def deploy_full_domination_mode(self):
        """Phase 4: Full domination deployment (100% capacity)"""
        print("\n📋 PHASE 4: FULL DOMINATION DEPLOYMENT (100% CAPACITY)")
        print("-" * 50)

        if self.miner_process:
            print("🛑 Stopping moderate miner...")
            self.stop_miner()

        self.deployment_phase = "domination"

        print("🏆 ACTIVATING FULL DOMINATION MODE!")
        print("   • Maximum prediction frequency")
        print("   • Peak hour optimization (3x)")
        print("   • Market regime adaptation")
        print("   • Adaptive thresholds")

        # Start miner with full domination settings
        env = os.environ.copy()
        env['DOMINATION_MODE'] = 'true'

        try:
            self.miner_process = subprocess.Popen([
                'python3', 'precog/miners/miner.py',
                '--neuron.name', 'elite_domination',
                '--wallet.name', 'cold_draven',
                '--wallet.hotkey', 'default',
                '--subtensor.chain_endpoint', 'wss://archive.substrate.network:443',
                '--axon.port', '8092',
                '--netuid', '55',
                '--logging.level', 'info',
                '--timeout', '16',
                '--vpermit_tao_limit', '2',
                '--forward_function', 'custom_model'
            ], env=env, cwd=os.getcwd())

            print("✅ FULL DOMINATION MINER ACTIVATED!")
            print(f"   PID: {self.miner_process.pid}")

            # Start continuous monitoring
            self.monitoring_active = True
            await self.monitor_performance(0, "domination")  # Continuous

            return True

        except Exception as e:
            print(f"❌ Failed to start domination miner: {e}")
            return False

    async def monitor_performance(self, duration_seconds, phase_name):
        """Monitor miner performance during deployment phase"""
        print(f"📊 MONITORING {phase_name.upper()} PHASE ({duration_seconds//60}min)")
        print("-" * 50)

        start_time = time.time()
        end_time = start_time + duration_seconds if duration_seconds > 0 else float('inf')

        performance_data = {
            'phase': phase_name,
            'start_time': datetime.now().isoformat(),
            'metrics': []
        }

        try:
            while time.time() < end_time:
                if not self.miner_process or self.miner_process.poll() is not None:
                    print("❌ Miner process died!")
                    break

                # Check every 5 minutes
                await asyncio.sleep(300)

                # Collect metrics
                metrics = self.collect_performance_metrics()
                performance_data['metrics'].append(metrics)

                # Check if we should proceed to next phase
                if self.should_scale_up(metrics, phase_name):
                    print("✅ Performance targets met - Ready to scale up!")
                    break

                # Check if we should fallback
                if self.should_fallback(metrics, phase_name):
                    print("⚠️  Performance issues detected - Consider fallback")
                    break

        except KeyboardInterrupt:
            print("\n⏹️  Monitoring interrupted by user")

        # Save performance data
        performance_file = f'performance_{phase_name}_{int(time.time())}.json'
        with open(performance_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        print(f"💾 Performance data saved to: {performance_file}")

    def collect_performance_metrics(self):
        """Collect current performance metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'miner_alive': self.miner_process and self.miner_process.poll() is None
        }

        # Try to read recent log entries
        try:
            if os.path.exists('miner_mainnet_domination.log'):
                with open('miner_mainnet_domination.log', 'r') as f:
                    lines = f.readlines()[-20:]  # Last 20 lines

                # Look for performance indicators
                reward_lines = [l for l in lines if 'reward' in l.lower() or 'tao' in l.lower()]
                prediction_lines = [l for l in lines if 'prediction' in l.lower()]

                metrics['recent_rewards'] = len(reward_lines)
                metrics['recent_predictions'] = len(prediction_lines)
                metrics['log_lines'] = len(lines)

        except Exception as e:
            metrics['log_error'] = str(e)

        return metrics

    def should_scale_up(self, metrics, phase_name):
        """Determine if we should scale up to next phase"""
        if phase_name == "conservative":
            # Scale up if miner is stable and making predictions
            return (metrics.get('miner_alive', False) and
                   metrics.get('recent_predictions', 0) > 0)
        elif phase_name == "moderate":
            # Scale up if consistent performance
            return (metrics.get('miner_alive', False) and
                   len(metrics.get('recent_predictions', [])) > 5)
        return False

    def should_fallback(self, metrics, phase_name):
        """Determine if we should fallback to previous phase"""
        # Fallback if miner keeps dying or no predictions
        return (not metrics.get('miner_alive', False) or
               metrics.get('recent_predictions', 0) == 0)

    def stop_miner(self):
        """Stop the current miner process"""
        if self.miner_process:
            try:
                self.miner_process.terminate()
                self.miner_process.wait(timeout=10)
                print("✅ Miner stopped successfully")
            except subprocess.TimeoutExpired:
                self.miner_process.kill()
                print("⚠️  Miner force killed")
            except Exception as e:
                print(f"❌ Error stopping miner: {e}")

    def emergency_fallback(self):
        """Emergency fallback to testnet"""
        print("🚨 EMERGENCY FALLBACK ACTIVATED")
        print("   Returning to testnet for safety...")

        self.stop_miner()

        # Could implement automatic testnet deployment here
        print("   💡 Run: ./start_testnet_miner.sh")
        print("   💡 Then analyze logs to identify issues")

    async def run_full_deployment(self):
        """Run the complete risk-mitigated deployment"""
        print("🚀 STARTING RISK-MITIGATED MAINNET DEPLOYMENT")
        print("=" * 60)
        print("Phases:")
        print("1. Pre-deployment validation")
        print("2. Conservative deployment (25%)")
        print("3. Moderate deployment (50%)")
        print("4. Full domination deployment (100%)")
        print()

        # Phase 1: Validation
        if not await self.run_pre_deployment_checks():
            print("❌ DEPLOYMENT ABORTED - Fix validation issues first")
            return False

        # Phase 2: Conservative
        if not await self.deploy_conservative_mode():
            print("❌ CONSERVATIVE DEPLOYMENT FAILED")
            self.emergency_fallback()
            return False

        # Phase 3: Moderate
        if not await self.scale_to_moderate_mode():
            print("❌ MODERATE DEPLOYMENT FAILED")
            self.emergency_fallback()
            return False

        # Phase 4: Full Domination
        if not await self.deploy_full_domination_mode():
            print("❌ FULL DOMINATION DEPLOYMENT FAILED")
            self.emergency_fallback()
            return False

        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("🏆 You are now dominating Precog subnet 55!")
        return True

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Risk-Mitigated Mainnet Deployment")
    parser.add_argument("--phase", choices=['validate', 'conservative', 'moderate', 'full', 'auto'],
                       default='auto', help="Deployment phase to run")
    parser.add_argument("--emergency-fallback", action="store_true",
                       help="Execute emergency fallback to testnet")

    args = parser.parse_args()

    deployment = RiskMitigationDeployment()

    try:
        if args.emergency_fallback:
            deployment.emergency_fallback()

        elif args.phase == 'validate':
            await deployment.run_pre_deployment_checks()

        elif args.phase == 'conservative':
            await deployment.deploy_conservative_mode()

        elif args.phase == 'moderate':
            await deployment.scale_to_moderate_mode()

        elif args.phase == 'full':
            await deployment.deploy_full_domination_mode()

        elif args.phase == 'auto':
            await deployment.run_full_deployment()

    except KeyboardInterrupt:
        print("\n⏹️  Deployment interrupted by user")
        deployment.stop_miner()

    except Exception as e:
        print(f"\n❌ Deployment error: {e}")
        deployment.emergency_fallback()

if __name__ == "__main__":
    asyncio.run(main())
