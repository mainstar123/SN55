#!/usr/bin/env python3
"""
Precog Subnet 55 - Complete Setup and Deployment Guide

This script guides you through the complete process of setting up and deploying
a competitive Precog miner to achieve top rankings.

PHASES:
1. Local Development & Testing
2. Testnet Validation
3. Mainnet Deployment

TARGET PERFORMANCE:
- MAPE: <0.08% (top-1 goal)
- Interval Coverage: >92%
- Response Time: <3s
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PrecogSetup:
    """Complete Precog setup and deployment guide"""

    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.venv_path = os.path.join(self.project_root, 'venv')

    def run_command(self, cmd, cwd=None, check=True):
        """Run shell command with logging"""
        try:
            logger.info(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, cwd=cwd or self.project_root,
                                  capture_output=True, text=True)
            if check and result.returncode != 0:
                logger.error(f"Command failed: {cmd}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)
            return result
        except Exception as e:
            logger.error(f"Error running command: {e}")
            raise

    def check_venv(self):
        """Check if virtual environment is set up"""
        if not os.path.exists(self.venv_path):
            logger.error("Virtual environment not found. Please run: python3 -m venv venv")
            return False

        # Check if we're in the venv
        python_path = sys.executable
        if 'venv' not in python_path:
            logger.warning("Not running in virtual environment. Activating...")
            self.run_command("source venv/bin/activate", check=False)

        return True

    def install_dependencies(self):
        """Install all required dependencies"""
        logger.info("Installing dependencies...")

        # Install PyTorch with CUDA support
        self.run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

        # Install ML libraries
        self.run_command("pip install scikit-learn==1.4.0 xgboost==2.0.3")

        # Install project dependencies (excluding torch versions that might conflict)
        deps = [
            "bittensor==9.9.0", "bittensor-cli==9.10.1", "numpy==1.26.4", "pydantic==2.3.0",
            "requests==2.32.3", "coinmetrics-api-client==2025.8.15.15", "pytz==2024.2",
            "pandas==2.2.3", "gitpython==3.1.43", "wandb==0.18.6", "black==25.1.0",
            "mypy==1.17.1", "flake8==7.3.0", "pytest==8.4.1", "pre-commit==4.3.0",
            "setuptools==70.0.0"
        ]

        for dep in deps:
            self.run_command(f"pip install {dep}")

        logger.info("Dependencies installed successfully")

    def setup_environment(self):
        """Setup environment files and project structure"""
        logger.info("Setting up environment...")

        # Create necessary directories
        dirs = ['models', 'scripts', 'data', 'logs']
        for d in dirs:
            os.makedirs(d, exist_ok=True)

        # Copy environment files
        if os.path.exists('.env.miner.example'):
            if not os.path.exists('.env.miner'):
                self.run_command("cp .env.miner.example .env.miner")
                logger.info("Created .env.miner - please edit with your wallet information")
            else:
                logger.info(".env.miner already exists")

        logger.info("Environment setup complete")

    def run_quality_checks(self):
        """Run code quality checks"""
        logger.info("Running quality checks...")

        try:
            # Check syntax
            self.run_command("python -m py_compile scripts/train_models.py")
            self.run_command("python -m py_compile scripts/backtest_local.py")
            self.run_command("python -m py_compile precog/miners/custom_model.py")

            # Run basic import test
            self.run_command("python -c 'import torch; import pandas as pd; import bittensor as bt; print(\"Imports successful\")'")

            logger.info("Quality checks passed")
            return True

        except Exception as e:
            logger.error(f"Quality checks failed: {e}")
            return False

    def phase_1_local_development(self):
        """Phase 1: Local Development & Testing"""
        print("\n" + "="*60)
        print("PHASE 1: LOCAL DEVELOPMENT & TESTING")
        print("="*60)

        print("This phase focuses on building and validating your model locally.")
        print("Goals:")
        print("- Implement custom GRU/LSTM ensemble model")
        print("- Train on historical data with walk-forward validation")
        print("- Achieve MAPE <0.15% and coverage >85% in backtesting")
        print()

        # Step 1: Environment check
        print("Step 1: Environment Setup")
        if not self.check_venv():
            print("❌ Virtual environment not set up")
            return False

        if not self.run_quality_checks():
            print("❌ Code quality checks failed")
            return False

        print("✅ Environment ready")
        print()

        # Step 2: Training data
        print("Step 2: Fetch Training Data")
        print("Run: python scripts/fetch_training_data.py --days 30")
        print("(This will download 30 days of BTC price data)")
        print()

        # Step 3: Model training
        print("Step 3: Train Models")
        print("Run: python scripts/train_models.py")
        print("Expected output:")
        print("- GRU point forecast model (target MAPE <0.09%)")
        print("- Quantile interval model (target coverage >85%)")
        print("- Feature scalers for real-time inference")
        print()

        # Step 4: Local backtesting
        print("Step 4: Local Backtesting")
        print("Run: python scripts/backtest_local.py")
        print("Quality Gates:")
        print("- ✅ MAPE <0.15%")
        print("- ✅ Coverage >85%")
        print("- ✅ Response time <16s")
        print()

        print("Commands to run:")
        print("1. python scripts/fetch_training_data.py --days 30")
        print("2. python scripts/train_models.py")
        print("3. python scripts/backtest_local.py")
        print()

        input("Press Enter when you've completed Phase 1...")
        return True

    def phase_2_testnet_validation(self):
        """Phase 2: Testnet Validation"""
        print("\n" + "="*60)
        print("PHASE 2: TESTNET VALIDATION")
        print("="*60)

        print("Deploy your model on Finney testnet (UID 256) and iterate until")
        print("you consistently outperform top-20 mainnet miners.")
        print()

        # Step 1: Wallet setup
        print("Step 1: Wallet Setup")
        print("Create testnet wallets:")
        print("btcli wallet new_coldkey --wallet.name precog_test")
        print("btcli wallet new_hotkey --wallet.name precog_test --wallet.hotkey test_miner")
        print()
        print("Fund your wallet with testnet TAO (faucet or transfer)")
        print("btcli wallet balance --wallet.name precog_test --subtensor.network finney")
        print()

        # Step 2: Registration
        print("Step 2: Register on Testnet")
        print("Register on subnet 256:")
        print("btcli subnet register --netuid 256 --wallet.name precog_test --wallet.hotkey test_miner --subtensor.network finney")
        print()
        print("Verify registration:")
        print("btcli wallet overview --netuid 256 --subtensor.network finney")
        print()

        # Step 3: Environment configuration
        print("Step 3: Configure Environment")
        print("Edit .env.miner with your testnet wallet information:")
        print("- NETWORK=testnet")
        print("- COLDKEY=precog_test")
        print("- MINER_HOTKEY=test_miner")
        print("- FORWARD_FUNCTION=custom_model")
        print()

        # Step 4: Deploy and monitor
        print("Step 4: Deploy and Monitor")
        print("Start miner:")
        print("make miner_custom ENV_FILE=.env.miner")
        print()
        print("Monitor performance:")
        print("python scripts/validate_performance.py --continuous --interval 60")
        print()
        print("Monitor competitors:")
        print("python scripts/monitor_competitors.py --continuous --interval 60")
        print()

        # Step 5: Iteration goals
        print("Step 5: Iteration Goals (Weeks 3-6)")
        print("Week 3: MAPE <0.12%, coverage >80%")
        print("Week 4: MAPE <0.10%, coverage >85% (add order flow features)")
        print("Week 5: Reduce RMSE by 10% (implement ensemble)")
        print("Week 6: MAPE <0.09%, coverage >87% (rolling retrain)")
        print()

        print("Quality Gate: Consistently outperform top-20 mainnet miners")
        print("Monitor: taostats.io/subnets/55 for comparative performance")
        print()

        input("Press Enter when ready for Phase 2 deployment...")
        return True

    def phase_3_mainnet_deployment(self):
        """Phase 3: Mainnet Deployment"""
        print("\n" + "="*60)
        print("PHASE 3: MAINNET DEPLOYMENT")
        print("="*60)

        print("Deploy to production (UID 55) with redundant infrastructure.")
        print("Target: Top-10 ranking with MAPE <0.08%")
        print()

        # Step 1: Production wallets
        print("Step 1: Production Wallets")
        print("Create production wallets:")
        print("btcli wallet new_coldkey --wallet.name precog_prod")
        print("btcli wallet new_hotkey --wallet.name precog_prod --wallet.hotkey prod_miner")
        print()
        print("Fund with mainnet TAO (minimum 0.1 TAO for registration)")
        print()

        # Step 2: Registration
        print("Step 2: Register on Mainnet")
        print("Register on subnet 55:")
        print("btcli subnet register --netuid 55 --wallet.name precog_prod --wallet.hotkey prod_miner --subtensor.network finney")
        print()
        print("Enable auto-staking:")
        print("btcli stake set-auto --wallet.name precog_prod --netuid 55")
        print()

        # Step 3: Production infrastructure
        print("Step 3: Production Infrastructure")
        print("Recommended setup:")
        print("- Primary: H200 GPU ($1500/month) - North Virginia")
        print("- Backup: RTX 4090 ($500/month) - Oregon")
        print("- RPC: OnFinality North Virginia ($100/month)")
        print("- Storage: 500GB NVMe SSD")
        print()

        # Step 4: PM2 configuration
        print("Step 4: PM2 Production Configuration")
        print("Edit .env.miner.prod with production settings")
        print("Start with production config:")
        print("make miner_custom ENV_FILE=.env.miner.prod")
        print()

        # Step 5: Continuous optimization
        print("Step 5: Continuous Optimization")
        print("Daily retraining:")
        print("crontab -e")
        print("Add: 0 2 * * * /path/to/precog/scripts/retrain_production.sh")
        print()
        print("Monitor performance:")
        print("python scripts/validate_performance.py --continuous")
        print("python scripts/monitor_competitors.py --continuous")
        print()

        # Success milestones
        print("Success Milestones:")
        print("Month 1: Survive immunity, emissions >0.038 TAO daily")
        print("Month 2: Top-20, MAPE <0.095%, coverage >87%")
        print("Month 3: Top-10, MAPE <0.085%, coverage >90%")
        print("Month 6: Top-5, MAPE <0.08%, coverage >92%")
        print("Month 12: #1 ranking, MAPE <0.075%, coverage >93%")
        print()

        print("Economic Model (Post-Halving):")
        print("Top-10: ~$6,000/month revenue")
        print("#1: ~$22,500/month revenue")
        print()

        return True

    def run_setup(self):
        """Run complete setup guide"""
        print("PRECROG SUBNET 55 - COMPLETE SETUP GUIDE")
        print("=========================================")
        print()
        print("This guide will walk you through achieving #1 ranking on Precog Subnet 55.")
        print("The process takes 3-6 months of disciplined execution.")
        print()
        print("TARGET PERFORMANCE:")
        print("- MAPE: <0.08% (top-1 goal)")
        print("- Interval Coverage: >92%")
        print("- Response Time: <3s")
        print()

        try:
            # Phase 1
            if not self.phase_1_local_development():
                return

            # Phase 2
            if not self.phase_2_testnet_validation():
                return

            # Phase 3
            if not self.phase_3_mainnet_deployment():
                return

            print("\n" + "="*60)
            print("🎉 SETUP COMPLETE!")
            print("="*60)
            print()
            print("You now have everything needed to achieve #1 ranking on Precog Subnet 55.")
            print("Remember the key success factors:")
            print()
            print("1. Model Innovation: Proprietary features beyond baselines")
            print("2. Infrastructure Reliability: Redundant systems, 99.9% uptime")
            print("3. Continuous Optimization: Daily retraining, regime adaptation")
            print("4. Competitive Intelligence: Monitor competitors, adapt quickly")
            print()
            print("Good luck! 🚀")
            print()

        except KeyboardInterrupt:
            print("\nSetup interrupted by user")
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            print(f"\nSetup failed: {e}")


def main():
    """Main setup function"""
    setup = PrecogSetup()
    setup.run_setup()


if __name__ == "__main__":
    main()
