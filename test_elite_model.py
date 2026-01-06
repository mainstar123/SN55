#!/usr/bin/env python3
"""
Test script to validate elite domination model loading and inference
"""

import sys
import os
import torch
import numpy as np
import pickle
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model loading function
from advanced_ensemble_model import load_advanced_ensemble

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading the elite domination model"""
    logger.info("Testing elite domination model loading...")

    model_path = 'elite_domination_model.pth'
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # Load the model
        model = load_advanced_ensemble(model_path, device_type)
        logger.info("✅ Model loaded successfully")
        logger.info(f"Model device: {next(model.parameters()).device}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None

def test_scaler_loading():
    """Test loading the feature scaler"""
    logger.info("Testing feature scaler loading...")

    scaler_path = 'models/feature_scaler.pkl'

    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("✅ Scaler loaded successfully")
        logger.info(f"Scaler type: {type(scaler)}")
        return scaler
    except Exception as e:
        logger.error(f"❌ Failed to load scaler: {e}")
        return None

def test_inference(model, scaler):
    """Test model inference with sample data"""
    logger.info("Testing model inference...")

    try:
        # Create sample data (24 features as expected by the model)
        sample_features = np.random.randn(24).astype(np.float32)

        # Scale the features
        scaled_features = scaler.transform(sample_features.reshape(1, -1))

        # Convert to tensor
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

        # Move to same device as model
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Run inference
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        logger.info("✅ Inference successful")
        logger.info(f"Input shape: {input_tensor.shape}")
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Output values: {output.cpu().numpy().flatten()[:5]}")  # Show first 5 values

        return True

    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting elite domination model validation...")

    # Test model loading
    model = test_model_loading()
    if not model:
        logger.error("❌ Model loading test FAILED")
        return False

    # Test scaler loading
    scaler = test_scaler_loading()
    if not scaler:
        logger.error("❌ Scaler loading test FAILED")
        return False

    # Test inference
    inference_success = test_inference(model, scaler)
    if not inference_success:
        logger.error("❌ Inference test FAILED")
        return False

    logger.info("🎉 All tests PASSED! Elite domination model is ready for deployment.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
