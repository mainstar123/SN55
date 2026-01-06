#!/usr/bin/env python3
"""
Test the updated deployment script with simple model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from start_domination_miner import DominationMiner
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_deployment():
    """Test the deployment with simple model"""
    logger.info("🚀 Testing deployment with simple model...")

    # Create config
    config = {
        'save_dir': '.',
        'timezone_offset': 0,
        'seq_len': 24
    }

    try:
        # Create miner instance
        miner = DominationMiner(config)
        logger.info("✅ Miner instance created")

        # Load domination system
        miner.load_domination_system()
        logger.info("✅ Domination system loaded")

        # Test with dummy market data
        dummy_market_data = [1.0] * 24  # 24 features

        # Test prediction
        prediction_result = miner.make_prediction(dummy_market_data)
        logger.info("✅ Prediction made successfully")
        logger.info(f"Prediction: {prediction_result['prediction']:.6f}")
        logger.info(f"Confidence: {prediction_result['confidence']:.6f}")
        logger.info(f"Should predict: {prediction_result['should_predict']}")

        return True

    except Exception as e:
        logger.error(f"❌ Deployment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_deployment()
    sys.exit(0 if success else 1)