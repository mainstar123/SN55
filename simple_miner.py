#!/usr/bin/env python3
"""
Simple working miner for immediate deployment
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime, timezone
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import simple model
from simple_working_model import load_simple_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_miner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SimpleMiner:
    """Simple miner for immediate deployment"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.prediction_count = 0
        self.total_earnings = 0.0

        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load the simple working model"""
        logger.info("Loading simple working model...")

        model_path = 'simple_working_model.pth'
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, creating one...")
            from simple_working_model import create_and_save_simple_model
            create_and_save_simple_model()

        self.model = load_simple_model(model_path, self.device.type)

        if hasattr(self.model, 'feature_scaler') and self.model.feature_scaler:
            self.scaler = self.model.feature_scaler
            logger.info("✅ Scaler loaded from model")
        else:
            logger.warning("⚠️  No scaler found, using raw features")
            self.scaler = None

        logger.info("✅ Simple model loaded successfully")

    def prepare_features(self, market_data: list) -> torch.Tensor:
        """Prepare market data for prediction"""
        # Convert to numpy array
        features = np.array(market_data).reshape(1, -1)

        # Scale features if scaler available
        if self.scaler and isinstance(self.scaler, dict):
            features = (features - self.scaler.get('mean', 0)) / (self.scaler.get('std', 1) + 1e-8)

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)

        return features_tensor

    def make_prediction(self, market_data: list) -> dict:
        """Make a prediction with the simple model"""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare features
        features_tensor = self.prepare_features(market_data)

        # Make prediction
        prediction_start = datetime.now(timezone.utc)
        self.model.eval()
        with torch.no_grad():
            point_pred, interval_pred = self.model(features_tensor)

        prediction_time = (datetime.now(timezone.utc) - prediction_start).total_seconds() * 1000

        prediction_value = float(point_pred.item())
        lower_bound = float(interval_pred[0, 0].item())
        upper_bound = float(interval_pred[0, 1].item())
        uncertainty = (upper_bound - lower_bound) / abs(prediction_value) if prediction_value != 0 else 0.1

        # Create prediction result
        prediction_result = {
            'prediction': prediction_value,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': uncertainty,
            'confidence': 1 - min(uncertainty, 1.0),  # Cap at 1.0
            'prediction_time_ms': prediction_time,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_version': 'simple_working_v1.0'
        }

        self.prediction_count += 1

        logger.info(f"Prediction: {prediction_value:.6f}")
        return prediction_result

    def get_status(self) -> dict:
        """Get miner status"""
        return {
            'total_predictions': self.prediction_count,
            'total_earnings': self.total_earnings,
            'model_loaded': self.model is not None,
            'device': str(self.device)
        }


def main():
    """Main function for testing"""
    logger.info("🚀 Starting Simple Miner Test")

    # Create config
    config = {}

    # Create miner
    miner = SimpleMiner(config)

    # Load model
    miner.load_model()

    # Test with dummy data
    dummy_data = [1.0] * 24  # 24 features
    prediction = miner.make_prediction(dummy_data)

    logger.info("✅ Test prediction completed")
    logger.info(f"Prediction: {prediction['prediction']:.6f}")
    logger.info(f"Lower bound: {prediction['lower_bound']:.6f}")
    logger.info(f"Upper bound: {prediction['upper_bound']:.6f}")
    # Show status
    status = miner.get_status()
    logger.info(f"Status: {status}")


if __name__ == "__main__":
    main()
