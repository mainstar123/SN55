#!/usr/bin/env python3
"""
Mainnet Deployment Script for Precog #1 Miner Domination System
Activates all advanced features for maximum TAO earnings
"""

import sys
import os
import torch
import logging
from datetime import datetime, timezone
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import domination system components
from simple_working_model import load_simple_model
from market_regime_detector import create_adaptive_prediction_system
from peak_hour_optimizer import create_ultra_precise_prediction_system
from performance_tracking_system import create_performance_tracking_system
from gpu_accelerated_training import create_inference_optimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('domination_miner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DominationMiner:
    """
    Complete domination miner with all advanced features activated
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = self._setup_device()

        # Load domination model
        self.model = None
        self.scaler = None
        self.inference_optimizer = None

        # Adaptive systems
        self.adaptive_system = None
        self.peak_optimizer = None

        # Performance tracking
        self.performance_tracker = None
        self.dashboard = None

        # Miner state
        self.is_running = False
        self.prediction_count = 0
        self.total_earnings = 0.0

        logger.info("🎯 Domination Miner initialized")

    def _setup_device(self) -> torch.device:
        """Setup compute device for inference"""
        device = self.config.get('device', 'auto')
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        logger.info(f"Using device: {device}")
        return device

    def load_domination_system(self):
        """Load the trained domination model and systems"""
        logger.info("Loading domination system...")

        # Use simple working model for immediate deployment
        model_path = 'simple_working_model.pth'

        if not os.path.exists(model_path):
            logger.warning(f"Simple working model not found at {model_path}. Creating one...")
            from simple_working_model import create_and_save_simple_model
            create_and_save_simple_model()

        logger.info(f"Loading simple working model from: {model_path}")
        # Load model
        self.model = load_simple_model(model_path, self.device.type)

        # Create inference optimizer
        self.inference_optimizer = create_inference_optimizer(self.model, self.device.type)

        # Scaler is loaded with the model in simple_working_model.py
        # It's stored in model.feature_scaler
        if hasattr(self.model, 'feature_scaler') and self.model.feature_scaler:
            self.scaler = self.model.feature_scaler
            logger.info("✅ Scaler loaded from model")
        else:
            logger.warning("⚠️  No scaler found in model, predictions may be less accurate")
            self.scaler = None

        # Initialize adaptive systems
        self.adaptive_system = create_adaptive_prediction_system(
            self.model,
            timezone_offset=self.config.get('timezone_offset', 0)
        )

        self.peak_optimizer = create_ultra_precise_prediction_system(
            timezone_offset=self.config.get('timezone_offset', 0)
        )

        # Initialize performance tracking
        self.performance_tracker, self.dashboard = create_performance_tracking_system(self.model)

        logger.info("✅ Domination system loaded successfully")

    def prepare_market_data(self, raw_features: list) -> torch.Tensor:
        """Prepare market data for prediction"""
        # Convert to numpy array
        features = np.array(raw_features).reshape(1, -1)

        # Scale features
        if self.scaler and isinstance(self.scaler, dict):
            # Simple dict-based scaler
            features = (features - self.scaler.get('mean', 0)) / (self.scaler.get('std', 1) + 1e-8)
        elif self.scaler and hasattr(self.scaler, 'transform'):
            # sklearn-style scaler
            features = self.scaler.transform(features)

        # Reshape for model input (add sequence dimension)
        seq_len = self.config.get('seq_len', 60)
        if features.shape[1] < seq_len:
            # Pad with zeros if needed
            padding = np.zeros((1, seq_len - features.shape[1]))
            features = np.concatenate([padding, features], axis=1)
        elif features.shape[1] > seq_len:
            # Take last seq_len features
            features = features[:, -seq_len:]

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)

        return features_tensor

    def should_make_prediction(self, market_data: list, current_time: datetime = None) -> tuple:
        """
        Determine if we should make a prediction based on all adaptive systems
        Returns: (should_predict, confidence_info)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Prepare market data
        features_tensor = self.prepare_market_data(market_data)

        # Get regime detection
        regime, regime_confidence = self.adaptive_system['regime_detector'].detect_regime()

        # Get peak hour status
        is_peak, confidence_multiplier, interval_info = self.peak_optimizer.should_predict_now(current_time)

        # Combine decision factors
        base_confidence = confidence_multiplier

        # Adjust for market regime
        regime_multipliers = {
            'bull': 1.3,
            'bear': 0.7,
            'volatile': 0.6,
            'ranging': 1.0,
            'unknown': 0.8
        }
        regime_multiplier = regime_multipliers.get(regime, 0.8)
        final_confidence = base_confidence * regime_multiplier

        # Decision logic
        should_predict = False
        reason = "low_confidence"

        if is_peak and final_confidence > 0.8:
            should_predict = True
            reason = "peak_high_confidence"
        elif is_peak and final_confidence > 0.6:
            should_predict = True
            reason = "peak_medium_confidence"
        elif final_confidence > 1.2:
            should_predict = True
            reason = "off_peak_high_reward_potential"

        decision_info = {
            'should_predict': should_predict,
            'reason': reason,
            'is_peak_hour': is_peak,
            'market_regime': regime,
            'regime_confidence': regime_confidence,
            'confidence_multiplier': final_confidence,
            'interval_info': interval_info,
            'regime_multiplier': regime_multiplier
        }

        return should_predict, decision_info

    def make_prediction(self, market_data: list, current_time: datetime = None) -> dict:
        """
        Make a domination prediction with full feature set
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Prepare features
        features_tensor = self.prepare_market_data(market_data)

        # Make prediction with simple model
        prediction_start = datetime.now(timezone.utc)
        self.model.eval()
        with torch.no_grad():
            # Use features directly (already scaled)
            features_input = features_tensor.squeeze(0)  # Remove batch dimension for simple model
            point_pred, interval_pred = self.model(features_input.unsqueeze(0))  # Add batch dimension

        prediction_time = (datetime.now(timezone.utc) - prediction_start).total_seconds() * 1000

        prediction_value = point_pred.item()
        # Calculate uncertainty as interval width
        lower_bound = interval_pred[0].item()
        upper_bound = interval_pred[1].item()
        uncertainty = (upper_bound - lower_bound) / prediction_value if prediction_value != 0 else 0.1

        # Get decision info
        should_predict, decision_info = self.should_make_prediction(market_data, current_time)

        # Create prediction result
        prediction_result = {
            'prediction': prediction_value,
            'uncertainty': uncertainty,
            'confidence': 1 - uncertainty,
            'should_predict': should_predict,
            'decision_info': decision_info,
            'prediction_time_ms': prediction_time,
            'timestamp': current_time.isoformat(),
            'model_version': 'domination_ensemble_v1.0'
        }

        # Update prediction count
        self.prediction_count += 1

        logger.info(f"Prediction made: {prediction_value:.6f} (confidence: {1-uncertainty:.3f})")
        return prediction_result

    def record_prediction_result(self, prediction_result: dict, actual_value: float = None,
                               reward: float = 0.0):
        """Record prediction result for performance tracking"""
        from performance_tracking_system import PredictionRecord

        # Create prediction record
        record = PredictionRecord(
            timestamp=datetime.fromisoformat(prediction_result['timestamp']),
            prediction=prediction_result['prediction'],
            actual=actual_value,
            reward=reward,
            confidence=prediction_result['confidence'],
            market_regime=prediction_result['decision_info']['market_regime'],
            is_peak_hour=prediction_result['decision_info']['is_peak_hour'],
            prediction_time_ms=prediction_result['prediction_time_ms']
        )

        # Record in performance tracker
        self.performance_tracker.record_prediction(record)

        # Update earnings
        self.total_earnings += reward

        logger.info(f"Recorded prediction result - Reward: {reward:.6f}")
    def get_status(self) -> dict:
        """Get current miner status"""
        current_time = datetime.now(timezone.utc)

        # Get performance report
        performance_report = self.performance_tracker.get_performance_report()

        # Get peak hour info
        next_peak = self.peak_optimizer.get_next_peak_window(current_time)

        status = {
            'is_running': self.is_running,
            'total_predictions': self.prediction_count,
            'total_earnings': self.total_earnings,
            'uptime_hours': 0.0,  # Would be tracked separately
            'current_performance': performance_report['current_metrics'],
            'alerts': performance_report['alerts'],
            'recommendations': performance_report['recommendations'],
            'next_peak_window': next_peak,
            'current_regime': self.adaptive_system['regime_detector'].detect_regime()[0],
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'timestamp': current_time.isoformat()
        }

        return status

    def print_status(self):
        """Print formatted status to console"""
        status = self.get_status()

        print("\n" + "="*60)
        print("🎯 PRECOG DOMINATION MINER STATUS")
        print("="*60)

        print("\n📊 PERFORMANCE:")
        print(f"  Total Predictions: {status['total_predictions']}")
        print(f"  Avg Prediction Time: {status['avg_prediction_time']:.6f}ms")
        print(f"  Memory Usage: {status['memory_usage']:.4f}GB")
        print(f"  GPU Usage: {status['gpu_usage']:.1f}%")
        print(f"  CPU Usage: {status['cpu_usage']:.1f}%")
        print(f"  Current Regime: {status['current_regime']}")

        if status['alerts']:
            print("\n🚨 ALERTS:")
            for alert in status['alerts']:
                severity_icon = "🔴" if alert['severity'] == 'high' else "🟡"
                print(f"  {severity_icon} {alert['message']}")

        next_peak = status['next_peak_window']
        print("\n⏰ NEXT PEAK WINDOW:")
        print(f"  Time: {next_peak['next_peak_start'].strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Wait: {next_peak['wait_minutes']} minutes")
        print(f"  Currently Peak: {next_peak['is_currently_peak']}")

        if status['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in status['recommendations']:
                print(f"  • {rec}")

        print(f"\n⏰ Status Time: {status['timestamp']}")
        print("="*60)

    def start_mining(self):
        """Start the domination mining operation"""
        logger.info("🚀 Starting Domination Miner")
        self.is_running = True

        # Load system if not already loaded
        if self.model is None:
            self.load_domination_system()

        # Print initial status
        self.print_status()

        # In a real implementation, this would integrate with the Bittensor miner
        # For now, we'll just log that it's ready
        logger.info("✅ Domination Miner ready for mainnet deployment")
        logger.info("💡 Integrate this with your Bittensor miner forward function")

    def stop_mining(self):
        """Stop the mining operation"""
        logger.info("🛑 Stopping Domination Miner")
        self.is_running = False

        # Save performance data
        self.performance_tracker.save_performance_data('domination_performance.json')

        # Print final status
        self.print_status()


def main():
    """Main function for running the domination miner"""
    parser = argparse.ArgumentParser(description='Start Precog #1 Miner Domination System')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--timezone_offset', type=int, default=0, help='Timezone offset in hours')
    parser.add_argument('--save_dir', type=str, default='domination_system', help='Domination system directory')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length for model input')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with mock predictions')

    args = parser.parse_args()

    # Configuration
    config = {
        'device': args.device,
        'timezone_offset': args.timezone_offset,
        'save_dir': args.save_dir,
        'seq_len': args.seq_len
    }

    print("🎯 Starting Precog #1 Miner Domination System")
    print("=" * 50)

    try:
        # Create domination miner
        miner = DominationMiner(config)

        # Start mining
        miner.start_mining()

        if args.demo:
            print("\n🎮 Running in demo mode...")
            print("Simulating predictions for testing...")

            # Generate some mock market data and predictions
            for i in range(5):
                # Mock market features
                mock_features = np.random.randn(24).tolist()

                # Make prediction
                prediction = miner.make_prediction(mock_features)

                # Mock actual value and reward
                actual = prediction['prediction'] + np.random.normal(0, 0.002)
                reward = np.random.random() * 0.001

                # Record result
                miner.record_prediction_result(prediction, actual, reward)

                print(f"Demo prediction {i+1}: {prediction['prediction']:.6f} -> {actual:.6f} (reward: {reward:.6f})")

            # Show final status
            miner.print_status()

        else:
            print("\n✅ Domination Miner is ready!")
            print("💡 This miner should be integrated with your Bittensor miner code")
            print("💡 Use the prediction functions in your forward() method")

        # Keep running until interrupted
        try:
            while miner.is_running:
                import time
                time.sleep(60)  # Check every minute

                # Print periodic status updates
                if miner.prediction_count > 0 and miner.prediction_count % 10 == 0:
                    miner.print_status()

        except KeyboardInterrupt:
            print("\n🛑 Received shutdown signal...")

        finally:
            miner.stop_mining()

    except Exception as e:
        logger.error(f"Miner failed: {e}")
        raise


if __name__ == "__main__":
    main()
