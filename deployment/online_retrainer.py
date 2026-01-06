#!/usr/bin/env python3
"""
ONLINE MODEL RETRAINER
Continuously improves model performance during runtime
Usage: python3 deployment/online_retrainer.py
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import joblib
from collections import deque
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnlineRetrainer:
    def __init__(self, model_path, scaler_path, buffer_size=100, retrain_threshold=50):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.buffer_size = buffer_size
        self.retrain_threshold = retrain_threshold

        # Performance buffer
        self.performance_buffer = deque(maxlen=buffer_size)
        self.retrain_count = 0

        # Load initial model and scaler
        self.load_model_and_scaler()

        # Start background retraining thread
        self.retraining_thread = threading.Thread(target=self._background_retraining, daemon=True)
        self.retraining_thread.start()

        logger.info(f"Online retrainer initialized with buffer size {buffer_size}")

    def load_model_and_scaler(self):
        """Load model and scaler from disk"""
        try:
            self.model = torch.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.model.eval()
            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model/scaler: {e}")
            raise

    def update_performance(self, features, prediction, actual, timestamp=None):
        """Add new prediction to performance buffer"""
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate error
        error = abs(prediction - actual) / actual if actual != 0 else 0
        mape = error * 100

        # Store in buffer
        self.performance_buffer.append({
            'timestamp': timestamp,
            'features': np.array(features),
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'mape': mape
        })

        # Log performance
        if len(self.performance_buffer) % 10 == 0:
            recent_mape = np.mean([p['mape'] for p in list(self.performance_buffer)[-10:]])
            logger.info(".2f")

    def get_current_performance(self):
        """Get current average performance metrics"""
        if len(self.performance_buffer) < 10:
            return None

        recent = list(self.performance_buffer)[-50:]  # Last 50 predictions
        avg_mape = np.mean([p['mape'] for p in recent])
        avg_error = np.mean([p['error'] for p in recent])

        return {
            'avg_mape': avg_mape,
            'avg_error': avg_error,
            'sample_size': len(recent)
        }

    def should_retrain(self):
        """Check if model should be retrained"""
        if len(self.performance_buffer) < self.retrain_threshold:
            return False

        perf = self.get_current_performance()
        if perf is None:
            return False

        # Retrain if MAPE > 0.12% (conservative threshold)
        return perf['avg_mape'] > 0.12

    def _background_retraining(self):
        """Background thread for model retraining"""
        while True:
            try:
                if self.should_retrain():
                    logger.info("🚀 Starting online retraining...")
                    self._perform_retraining()
                    self.retrain_count += 1
                    logger.info(f"✅ Retraining {self.retrain_count} completed")

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Retraining error: {e}")
                time.sleep(60)  # Wait before retry

    def _perform_retraining(self):
        """Perform online retraining with recent data"""
        try:
            # Prepare training data from buffer
            recent_data = list(self.performance_buffer)[-self.retrain_threshold:]

            # Extract features and targets
            features = np.array([p['features'] for p in recent_data])
            actuals = np.array([p['actual'] for p in recent_data])

            # Scale features
            features_scaled = self.scaler.transform(features)

            logger.info(f"Retraining on {len(recent_data)} samples")

            # Fine-tune model
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00005)  # Very low LR
            criterion = torch.nn.MSELoss()

            # Multiple epochs on recent data
            for epoch in range(20):
                optimizer.zero_grad()
                outputs = self.model(torch.FloatTensor(features_scaled))
                loss = criterion(outputs.squeeze(), torch.FloatTensor(actuals))
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 5 == 0:
                    logger.debug(f"Epoch {epoch+1}/20 - Loss: {loss.item():.6f}")

            # Save updated model
            torch.save(self.model, self.model_path)
            self.model.eval()

            logger.info("Model updated and saved")

        except Exception as e:
            logger.error(f"Retraining failed: {e}")

    def predict(self, features):
        """Make prediction with current model"""
        try:
            features_scaled = self.scaler.transform([features])
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(torch.FloatTensor(features_scaled)).item()
            return prediction
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

# Global retrainer instance
retrainer = None

def get_retrainer():
    """Get or create retrainer instance"""
    global retrainer
    if retrainer is None:
        model_path = 'models/domination_model_trained.pth'
        scaler_path = 'models/feature_scaler.pkl'
        retrainer = OnlineRetrainer(model_path, scaler_path)
    return retrainer

if __name__ == "__main__":
    # Standalone testing
    logger.info("Starting online retrainer...")

    retrainer = get_retrainer()

    # Keep running
    try:
        while True:
            time.sleep(60)
            perf = retrainer.get_current_performance()
            if perf:
                logger.info(f"Current performance - MAPE: {perf['avg_mape']:.4f}%, Samples: {perf['sample_size']}")

    except KeyboardInterrupt:
        logger.info("Online retrainer stopped")
