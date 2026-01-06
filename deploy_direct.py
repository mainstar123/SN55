#!/usr/bin/env python3
"""
DIRECT DEPLOYMENT OF MULTI-ASSET DOMINATION MINER
Run without PM2 for testing
"""

import os
import sys
import time
import logging
from datetime import datetime

# Set training mode to prevent model loading conflicts
os.environ['TRAINING_MODE'] = 'true'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('direct_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.append('.')

def check_environment():
    """Check if all required files exist"""
    logger.info("🔍 Checking deployment environment...")

    required_files = [
        'models/multi_asset_domination_model.pth',
        'models/multi_asset_feature_scaler.pkl',
        '.env.miner'
    ]

    for file in required_files:
        if not os.path.exists(file):
            logger.error(f"❌ Missing required file: {file}")
            return False
        else:
            logger.info(f"✅ Found: {file}")

    logger.info("✅ Environment check passed")
    return True

def run_miner():
    """Run the miner directly using Bittensor framework"""
    logger.info("🚀 Starting multi-asset domination miner...")

    try:
        # Import Bittensor miner
        from precog.miners.miner import Miner
        import bittensor as bt

        # Load configuration
        config = bt.config()
        config.netuid = 55
        config.forward_function = 'enhanced_domination'  # Custom forward function

        # Create and run miner
        miner = Miner(config=config)
        miner.run()

    except KeyboardInterrupt:
        logger.info("🛑 Miner stopped by user")
    except Exception as e:
        logger.error(f"❌ Miner failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    logger.info("🎯 DIRECT MULTI-ASSET MINER DEPLOYMENT")
    logger.info("=" * 50)

    if not check_environment():
        logger.error("❌ Environment check failed. Cannot deploy.")
        return

    logger.info("🌟 Multi-asset domination miner ready for subnet 55!")
    logger.info("   • BTC predictions: ✅")
    logger.info("   • ETH predictions: ✅")
    logger.info("   • TAO predictions: ✅")
    logger.info("   • Competition intelligence: ✅")
    logger.info("   • Meta-learning optimization: ✅")

    # Run the miner
    run_miner()

if __name__ == "__main__":
    main()
