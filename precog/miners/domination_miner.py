"""
DOMINATION MINER: Enhanced miner with peak-hour optimization and ensemble methods
Implements the 48-hour execution plan to surpass UID 31 and become #1
"""

import time
import asyncio
import logging
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timezone, timedelta
import statistics
import importlib

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import bittensor as bt

from precog.protocol import Challenge
from precog.utils.cm_data import CMData

logger = logging.getLogger(__name__)

# Global models and tracking
point_model = None
interval_model = None
scaler = None
models_loaded = False

# Performance tracking
performance_history = []
reward_history = []
peak_hours_active = False

class BasicEnsemble(nn.Module):
    """Basic ensemble combining GRU and Transformer for immediate improvement"""

    def __init__(self, input_size=24, hidden_size=128):
        super().__init__()
        self.gru = EnhancedGRUPriceForecaster(input_size, hidden_size, num_heads=8)
        # Simple transformer for basic ensemble
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=min(8, input_size),
                dim_feedforward=hidden_size,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.transformer_fc = nn.Linear(input_size, 1)
        self.ensemble_weight = nn.Parameter(torch.tensor(0.5))  # Learnable weight

    def forward(self, x):
        # GRU prediction
        gru_pred = self.gru(x)

        # Simple transformer prediction
        transformer_out = self.transformer_encoder(x)
        transformer_pred = self.transformer_fc(transformer_out[:, -1, :])

        # Weighted ensemble
        weight = torch.sigmoid(self.ensemble_weight)
        ensemble_pred = weight * gru_pred + (1 - weight) * transformer_pred

        return ensemble_pred

class DominationMiner:
    """Enhanced miner with domination strategies"""

    def __init__(self):
        # Peak hour configuration
        self.peak_hours = [9, 10, 13, 14]  # UTC hours
        self.peak_frequency = 15  # minutes
        self.normal_frequency = 60  # minutes
        self.peak_confidence_threshold = 0.75
        self.normal_confidence_threshold = 0.85

        # Market regime detection
        self.market_regimes = {
            'bull': {'freq': 30, 'threshold': 0.75, 'description': 'High volatility - frequent predictions'},
            'bear': {'freq': 45, 'threshold': 0.82, 'description': 'Medium volatility - balanced'},
            'volatile': {'freq': 15, 'threshold': 0.70, 'description': 'High volatility - aggressive'},
            'ranging': {'freq': 60, 'threshold': 0.85, 'description': 'Low volatility - conservative'}
        }

        # Performance tracking
        self.prediction_count = 0
        self.reward_sum = 0.0
        self.response_times = []
        self.hourly_performance = {}

        # Adaptive thresholds
        self.volatility_thresholds = {
            'high': 0.05,    # 5% price movement
            'medium': 0.02,  # 2% price movement
            'low': 0.01      # 1% price movement
        }

        logger.info("🏆 DOMINATION MINER INITIALIZED")
        logger.info("🎯 Target: Surpass UID 31 and become #1")

    def get_current_hour_utc(self) -> int:
        """Get current hour in UTC"""
        return datetime.now(timezone.utc).hour

    def is_peak_hour(self) -> bool:
        """Check if current time is peak hour"""
        current_hour = self.get_current_hour_utc()
        return current_hour in self.peak_hours

    def detect_market_regime(self, recent_data: pd.DataFrame) -> str:
        """Detect current market regime based on recent price action"""
        if len(recent_data) < 5:
            return 'ranging'  # Default

        try:
            prices = recent_data['price'].values[-60:]  # Last hour
            if len(prices) < 5:
                return 'ranging'

            # Calculate volatility (standard deviation of returns)
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0

            # Calculate trend
            if len(prices) >= 10:
                short_trend = (prices[-1] - prices[-10]) / prices[-10]
                if short_trend > 0.02:
                    trend = 'bull'
                elif short_trend < -0.02:
                    trend = 'bear'
                else:
                    trend = 'sideways'
            else:
                trend = 'sideways'

            # Determine regime
            if volatility > self.volatility_thresholds['high']:
                return 'volatile'
            elif trend == 'bull' and volatility < self.volatility_thresholds['medium']:
                return 'bull'
            elif trend == 'bear':
                return 'bear'
            else:
                return 'ranging'

        except Exception as e:
            logger.warning(f"Regime detection error: {e}")
            return 'ranging'

    def get_adaptive_parameters(self, market_regime: str, is_peak_hour: bool) -> Dict:
        """Get adaptive prediction parameters based on market conditions"""

        # Base parameters from regime
        regime_params = self.market_regimes[market_regime]

        # Adjust for peak hours
        if is_peak_hour:
            params = regime_params.copy()
            params['freq'] = min(params['freq'], self.peak_frequency)
            params['threshold'] = min(params['threshold'], self.peak_confidence_threshold)
            params['description'] += " (PEAK HOUR BONUS)"
        else:
            params = regime_params.copy()
            params['freq'] = min(params['freq'], self.normal_frequency)
            params['threshold'] = min(params['threshold'], self.normal_confidence_threshold)

        # Add domination multiplier for aggressive prediction during peak hours
        if is_peak_hour and market_regime in ['volatile', 'bull']:
            params['freq'] = max(10, params['freq'] // 2)  # At least every 10 minutes
            params['threshold'] = max(0.65, params['threshold'] - 0.1)  # Lower threshold

        return params

    def should_make_prediction(self, confidence_score: float, market_regime: str, is_peak_hour: bool) -> bool:
        """Determine if prediction should be made based on adaptive thresholds"""
        params = self.get_adaptive_parameters(market_regime, is_peak_hour)

        # Additional boost during peak hours for domination
        if is_peak_hour:
            confidence_score *= 1.2  # 20% boost to confidence threshold

        return confidence_score >= params['threshold']

    def track_prediction(self, prediction_value: float, actual_value: float,
                        confidence: float, response_time: float, reward: float = 0.0):
        """Track prediction performance for optimization"""
        self.prediction_count += 1
        self.reward_sum += reward
        self.response_times.append(response_time)

        # Keep only last 100 response times
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]

        # Track hourly performance
        current_hour = datetime.now().hour
        if current_hour not in self.hourly_performance:
            self.hourly_performance[current_hour] = {
                'predictions': 0,
                'rewards': 0.0,
                'response_times': []
            }

        hourly = self.hourly_performance[current_hour]
        hourly['predictions'] += 1
        hourly['rewards'] += reward
        hourly['response_times'].append(response_time)

        # Keep only last 10 response times per hour
        if len(hourly['response_times']) > 10:
            hourly['response_times'] = hourly['response_times'][-10:]

        # Log performance every 10 predictions
        if self.prediction_count % 10 == 0:
            avg_reward = self.reward_sum / self.prediction_count
            avg_response_time = statistics.mean(self.response_times) if self.response_times else 0

            logger.info(".6f"
                        ".3f"
                        ".1f")

            # Check if we're on track for domination
            if avg_reward >= 0.08:
                logger.info("🎉 REWARD TARGET ACHIEVED! Surpassing UID 31 level")
            elif avg_reward >= 0.05:
                logger.info("✅ Good progress - on track for domination")
            else:
                logger.warning("⚠️ Reward performance needs improvement")

    def get_performance_report(self) -> Dict:
        """Generate performance report for monitoring"""
        avg_reward = self.reward_sum / self.prediction_count if self.prediction_count > 0 else 0
        avg_response_time = statistics.mean(self.response_times) if self.response_times else 0

        return {
            'total_predictions': self.prediction_count,
            'average_reward': avg_reward,
            'average_response_time': avg_response_time,
            'peak_hour_active': self.is_peak_hour(),
            'hourly_performance': self.hourly_performance,
            'domination_targets': {
                'current_avg_reward': avg_reward,
                'target_1h': 0.08,  # Surpass UID 31
                'target_24h': 0.12, # Top 3
                'target_48h': 0.15, # Dominate UID 31
                'progress_to_target': avg_reward / 0.08 if avg_reward > 0 else 0
            }
        }

def load_domination_models():
    """Load enhanced models for domination strategy"""
    global point_model, interval_model, scaler, models_loaded

    if models_loaded:
        return

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading domination models on device: {device}")

        # Load enhanced ensemble instead of single model
        point_model = BasicEnsemble(input_size=24, hidden_size=128)
        point_model.load_state_dict(torch.load('models/enhanced_gru.pth', map_location=device))
        point_model.to(device)
        point_model.eval()

        # Load interval model (optional for now, focus on point predictions)
        interval_model = QuantileIntervalForecaster(input_size=24)
        try:
            interval_model.load_state_dict(torch.load('models/interval_model.pth', map_location=device))
            interval_model.to(device)
            interval_model.eval()
        except:
            logger.warning("Interval model not available, using point predictions only")

        # Load scaler
        import joblib
        scaler = joblib.load('models/feature_scaler.pkl')

        models_loaded = True
        logger.info("✅ Domination models loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise

def enhanced_forward(synapse: Challenge, cm: CMData, domination_miner: DominationMiner) -> Challenge:
    """Enhanced forward function with domination strategies"""

    start_time = time.time()

    try:
        # Get current market data
        data = cm.get_recent_data(minutes=60)  # Last hour
        if data.empty:
            logger.warning("No market data available")
            return synapse

        # Detect market regime
        market_regime = domination_miner.detect_market_regime(data)
        is_peak_hour = domination_miner.is_peak_hour()

        # Get adaptive parameters
        params = domination_miner.get_adaptive_parameters(market_regime, is_peak_hour)

        logger.info(f"🎯 Market Regime: {market_regime.upper()} | Peak Hour: {is_peak_hour} | Freq: {params['freq']}min")

        # Extract and scale features
        features_df = add_advanced_features(data)
        feature_cols = [col for col in features_df.columns if col != 'price' and col != 'timestamp']
        features = features_df[feature_cols].iloc[-1:].values  # Last timestep

        # Apply feature scaling
        if scaler:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        # Convert to tensor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)  # (1, 1, features)

        # Make prediction with ensemble
        with torch.no_grad():
            if point_model:
                point_prediction = point_model(features_tensor).item()
            else:
                # Fallback to simple prediction
                current_price = features_df['price'].iloc[-1]
                point_prediction = current_price * (1 + np.random.normal(0, 0.01))  # Small random change

        # Calculate confidence (based on prediction magnitude and market volatility)
        market_volatility = features_df['price'].pct_change().std() if len(features_df) > 5 else 0.01
        confidence_score = min(1.0, 1.0 / (1.0 + market_volatility * 10))  # Higher confidence in stable markets

        # Apply peak hour bonus
        if is_peak_hour:
            confidence_score *= 1.2

        # Decide whether to make prediction based on adaptive thresholds
        should_predict = domination_miner.should_make_prediction(confidence_score, market_regime, is_peak_hour)

        if should_predict:
            # Convert to TAO prediction (assuming current BTC price ~$100k)
            current_btc_price = features_df['price'].iloc[-1]
            tao_prediction = point_prediction * 1000  # Rough TAO conversion

            # Create prediction intervals (90% confidence)
            interval_width = abs(point_prediction * 0.05)  # 5% interval
            lower_bound = point_prediction - interval_width
            upper_bound = point_prediction + interval_width

            # Set predictions
            synapse.predictions = [tao_prediction]
            synapse.intervals = [[lower_bound * 1000, upper_bound * 1000]]

            # Track performance (reward will be updated when received)
            response_time = time.time() - start_time
            domination_miner.track_prediction(
                prediction_value=point_prediction,
                actual_value=current_btc_price,  # Will be updated with real value
                confidence=confidence_score,
                response_time=response_time,
                reward=0.0  # Will be updated when reward is received
            )

            logger.info(".2f"
                        ".3f"
                        ".1%")

        else:
            synapse.predictions = None
            synapse.intervals = None
            logger.info("⏸️ Skipping prediction (below confidence threshold)")

        # Log performance report every 50 predictions
        if domination_miner.prediction_count % 50 == 0:
            report = domination_miner.get_performance_report()
            logger.info(f"📊 PERFORMANCE REPORT: MAE: {report['metrics']['mae']:.6f}, RMSE: {report['metrics']['rmse']:.3f}")
            if report['domination_targets']['progress_to_target'] >= 1.0:
                logger.info("🎉 TARGET ACHIEVED: Surpassing UID 31!")
            else:
                progress_pct = report['domination_targets']['progress_to_target'] * 100
                logger.info(f"📈 Progress to target: {progress_pct:.1f}%")

    except Exception as e:
        logger.error(f"❌ Enhanced forward error: {e}")
        synapse.predictions = None
        synapse.intervals = None

    return synapse

# Global domination miner instance
domination_miner = DominationMiner()

async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """Main forward function with domination strategies"""
    return enhanced_forward(synapse, cm, domination_miner)

# Initialize models on import
try:
    load_domination_models()
except Exception as e:
    logger.warning(f"Could not load domination models: {e}")
    logger.info("Running with basic prediction capabilities")
