"""
Market Regime Detection and Adaptive Prediction Strategies for Precog #1 Miner
Detects bull/bear/volatile/ranging markets and adapts prediction strategies accordingly
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timezone, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple indicators
    Detects: Bull, Bear, Volatile, Ranging markets
    """

    def __init__(self, lookback_period: int = 200, confidence_threshold: float = 0.7):
        self.lookback_period = lookback_period
        self.confidence_threshold = confidence_threshold

        # Historical data storage
        self.price_history = []
        self.volume_history = []
        self.regime_history = []
        self.confidence_history = []

        # Regime-specific statistics
        self.regime_stats = {
            'bull': {'accuracy': [], 'returns': [], 'volatility': []},
            'bear': {'accuracy': [], 'returns': [], 'volatility': []},
            'volatile': {'accuracy': [], 'returns': [], 'volatility': []},
            'ranging': {'accuracy': [], 'returns': [], 'volatility': []}
        }

    def update_market_data(self, price: float, volume: float = None, timestamp: Optional[datetime] = None):
        """Update market data for regime detection"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.price_history.append({
            'price': price,
            'volume': volume or 1.0,
            'timestamp': timestamp
        })

        # Keep only recent data
        if len(self.price_history) > self.lookback_period * 2:
            self.price_history = self.price_history[-self.lookback_period * 2:]

    def detect_regime(self, current_price: float = None,
                     recent_prices: List[float] = None) -> Tuple[str, float]:
        """
        Detect current market regime with confidence score
        Returns: (regime, confidence)
        """
        if recent_prices is None and len(self.price_history) < self.lookback_period:
            return 'unknown', 0.0

        # Get price data
        if recent_prices is not None:
            prices = np.array(recent_prices[-self.lookback_period:])
        else:
            prices = np.array([p['price'] for p in self.price_history[-self.lookback_period:]])

        if len(prices) < 50:  # Minimum data requirement
            return 'unknown', 0.0

        # Calculate multiple indicators
        indicators = self._calculate_indicators(prices)

        # Determine regime based on indicators
        regime, confidence = self._classify_regime(indicators)

        # Store result
        self.regime_history.append({
            'regime': regime,
            'confidence': confidence,
            'timestamp': datetime.now(timezone.utc),
            'indicators': indicators
        })

        return regime, confidence

    def _calculate_indicators(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate technical indicators for regime detection"""
        returns = np.diff(prices) / prices[:-1]

        indicators = {}

        # Trend indicators
        indicators['momentum'] = np.mean(returns[-20:])  # Short-term momentum
        indicators['trend_strength'] = np.mean(returns[-50:])  # Medium-term trend
        indicators['long_trend'] = np.mean(returns[-100:]) if len(returns) >= 100 else indicators['trend_strength']

        # Volatility indicators
        indicators['volatility'] = np.std(returns)
        indicators['recent_volatility'] = np.std(returns[-20:])
        indicators['volatility_ratio'] = indicators['recent_volatility'] / max(indicators['volatility'], 1e-6)

        # Range indicators
        indicators['price_range'] = (np.max(prices[-50:]) - np.min(prices[-50:])) / np.mean(prices[-50:])
        indicators['average_range'] = np.mean(np.abs(np.diff(prices[-50:])) / prices[-50:][:-1])

        # Oscillator indicators
        if len(prices) >= 14:
            indicators['rsi'] = self._calculate_rsi(prices)
        else:
            indicators['rsi'] = 50.0

        # Volume indicators (if available)
        if len(self.price_history) >= self.lookback_period:
            volumes = np.array([p['volume'] for p in self.price_history[-self.lookback_period:]])
            indicators['volume_trend'] = np.mean(np.diff(volumes[-20:])) if len(volumes) >= 20 else 0.0
        else:
            indicators['volume_trend'] = 0.0

        return indicators

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0

        gains = []
        losses = []

        for i in range(1, min(len(prices), period + 1)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _classify_regime(self, indicators: Dict[str, float]) -> Tuple[str, float]:
        """Classify market regime based on indicators"""
        # Regime scoring system
        scores = {
            'bull': 0.0,
            'bear': 0.0,
            'volatile': 0.0,
            'ranging': 0.0
        }

        # Bull market criteria
        if indicators['trend_strength'] > 0.001 and indicators['momentum'] > 0.0005:
            scores['bull'] += 0.4
        if indicators['rsi'] > 60:
            scores['bull'] += 0.3
        if indicators['volume_trend'] > 0:
            scores['bull'] += 0.2

        # Bear market criteria
        if indicators['trend_strength'] < -0.001 and indicators['momentum'] < -0.0005:
            scores['bear'] += 0.4
        if indicators['rsi'] < 40:
            scores['bear'] += 0.3
        if indicators['volume_trend'] < 0:
            scores['bear'] += 0.2

        # Volatile market criteria
        if indicators['volatility'] > 0.02 or indicators['volatility_ratio'] > 1.5:
            scores['volatile'] += 0.5
        if indicators['price_range'] > 0.03:
            scores['volatile'] += 0.3

        # Ranging market criteria
        if abs(indicators['trend_strength']) < 0.0005 and indicators['volatility'] < 0.015:
            scores['ranging'] += 0.4
        if 40 <= indicators['rsi'] <= 60:
            scores['ranging'] += 0.3
        if indicators['volatility_ratio'] < 1.2:
            scores['ranging'] += 0.2

        # Determine winning regime
        max_score = max(scores.values())
        if max_score < 0.3:  # Minimum confidence threshold
            return 'unknown', 0.0

        winning_regime = max(scores, key=scores.get)
        confidence = max_score / sum(scores.values()) if sum(scores.values()) > 0 else 0.0

        # Boost confidence if multiple indicators agree
        if confidence >= self.confidence_threshold:
            return winning_regime, confidence
        else:
            return 'unknown', confidence

    def get_regime_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get detailed regime statistics"""
        if not self.regime_history:
            return {}

        # Analyze regime performance
        stats = {}
        for regime in ['bull', 'bear', 'volatile', 'ranging']:
            regime_data = [r for r in self.regime_history if r['regime'] == regime]

            if regime_data:
                confidence_avg = np.mean([r['confidence'] for r in regime_data])
                frequency = len(regime_data) / len(self.regime_history)

                # Calculate indicator averages
                indicator_avgs = {}
                for indicator in ['momentum', 'trend_strength', 'volatility', 'rsi']:
                    values = [r['indicators'].get(indicator, 0) for r in regime_data if indicator in r['indicators']]
                    if values:
                        indicator_avgs[indicator] = np.mean(values)

                stats[regime] = {
                    'frequency': frequency,
                    'avg_confidence': confidence_avg,
                    'indicators': indicator_avgs,
                    'count': len(regime_data)
                }

        return stats

    def get_optimal_strategy(self, regime: str) -> Dict[str, Union[str, float]]:
        """Get optimal prediction strategy for current regime"""
        strategies = {
            'bull': {
                'prediction_frequency': 'high',
                'confidence_threshold': 0.6,
                'model_weights': {'gru': 0.5, 'transformer': 0.3, 'lstm': 0.2},
                'risk_multiplier': 1.2,
                'description': 'High frequency predictions, moderate confidence, momentum-focused'
            },
            'bear': {
                'prediction_frequency': 'medium',
                'confidence_threshold': 0.7,
                'model_weights': {'gru': 0.3, 'transformer': 0.4, 'lstm': 0.3},
                'risk_multiplier': 0.8,
                'description': 'Conservative approach, high confidence required'
            },
            'volatile': {
                'prediction_frequency': 'low',
                'confidence_threshold': 0.8,
                'model_weights': {'gru': 0.2, 'transformer': 0.3, 'lstm': 0.5},
                'risk_multiplier': 0.6,
                'description': 'Very conservative, only high-confidence predictions'
            },
            'ranging': {
                'prediction_frequency': 'medium',
                'confidence_threshold': 0.65,
                'model_weights': {'gru': 0.4, 'transformer': 0.4, 'lstm': 0.2},
                'risk_multiplier': 1.0,
                'description': 'Balanced approach, steady prediction frequency'
            },
            'unknown': {
                'prediction_frequency': 'low',
                'confidence_threshold': 0.75,
                'model_weights': {'gru': 0.33, 'transformer': 0.33, 'lstm': 0.34},
                'risk_multiplier': 0.7,
                'description': 'Conservative default strategy'
            }
        }

        return strategies.get(regime, strategies['unknown'])


class AdaptivePredictor:
    """
    Adaptive prediction system that adjusts strategy based on market regime
    """

    def __init__(self, base_model, regime_detector: MarketRegimeDetector):
        self.base_model = base_model
        self.regime_detector = regime_detector
        self.current_regime = 'unknown'
        self.current_strategy = {}

        # Performance tracking
        self.prediction_history = []
        self.regime_performance = {}

    def predict_adaptive(self, market_data: torch.Tensor,
                        current_price: float = None) -> Tuple[torch.Tensor, Dict[str, Union[str, float]]]:
        """
        Make adaptive predictions based on current market regime
        Returns: (predictions, strategy_info)
        """
        # Detect current regime
        regime, confidence = self.regime_detector.detect_regime(current_price)
        self.current_regime = regime

        # Get optimal strategy for regime
        strategy = self.regime_detector.get_optimal_strategy(regime)
        self.current_strategy = strategy

        # Adjust model predictions based on strategy
        with torch.no_grad():
            predictions, uncertainties = self.base_model(market_data)

            # Apply confidence filtering based on strategy
            confidence_threshold = strategy['confidence_threshold']
            confidence_scores = 1 - uncertainties.squeeze()

            # Only predict when confidence meets threshold
            valid_predictions = confidence_scores >= confidence_threshold

            # Apply risk multiplier
            risk_multiplier = strategy['risk_multiplier']
            predictions = predictions * risk_multiplier

        strategy_info = {
            'regime': regime,
            'regime_confidence': confidence,
            'strategy': strategy,
            'valid_predictions': valid_predictions.sum().item(),
            'total_predictions': predictions.shape[0],
            'risk_multiplier': risk_multiplier
        }

        return predictions, strategy_info

    def update_performance(self, prediction: float, actual: float,
                          regime: str, reward: float = 0.0):
        """Update performance tracking by regime"""
        error = abs(prediction - actual) / max(abs(actual), 1e-6)

        self.prediction_history.append({
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'regime': regime,
            'reward': reward,
            'timestamp': datetime.now(timezone.utc)
        })

        # Update regime-specific performance
        if regime not in self.regime_performance:
            self.regime_performance[regime] = []

        self.regime_performance[regime].append({
            'error': error,
            'reward': reward
        })

    def get_regime_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by regime"""
        stats = {}
        for regime, performances in self.regime_performance.items():
            if performances:
                errors = [p['error'] for p in performances]
                rewards = [p['reward'] for p in performances]

                stats[regime] = {
                    'mape': np.mean(errors),
                    'accuracy': 1 - np.mean(errors),
                    'avg_reward': np.mean(rewards),
                    'total_predictions': len(performances)
                }

        return stats


class PeakHourOptimizer:
    """
    Ultra-precise peak hour detection and optimization for Precog rewards
    """

    def __init__(self, timezone_offset: int = 0):  # UTC offset in hours
        self.timezone_offset = timezone_offset

        # Peak hour analysis
        self.reward_history = []
        self.hourly_performance = {}
        self.peak_hours = set()

        # Bittensor peak hours (UTC): typically 9-11, 13-15
        self.known_peak_hours = {9, 10, 11, 13, 14, 15}

    def update_reward_data(self, reward: float, timestamp: Optional[datetime] = None):
        """Update reward data for analysis"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Adjust for timezone
        local_hour = (timestamp.hour + self.timezone_offset) % 24

        self.reward_history.append({
            'reward': reward,
            'hour': local_hour,
            'timestamp': timestamp
        })

        # Keep last 1000 entries
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

    def analyze_peak_hours(self) -> Dict[str, Union[List[int], Dict[int, float]]]:
        """Analyze historical data to identify optimal prediction hours"""
        if len(self.reward_history) < 50:
            return {'peak_hours': list(self.known_peak_hours), 'hourly_rewards': {}}

        # Calculate hourly statistics
        hourly_stats = {}
        for hour in range(24):
            hour_rewards = [r['reward'] for r in self.reward_history if r['hour'] == hour]
            if hour_rewards:
                hourly_stats[hour] = {
                    'avg_reward': np.mean(hour_rewards),
                    'total_reward': sum(hour_rewards),
                    'count': len(hour_rewards),
                    'std_reward': np.std(hour_rewards) if len(hour_rewards) > 1 else 0
                }

        # Identify peak hours (top 6 hours by average reward)
        sorted_hours = sorted(hourly_stats.items(),
                            key=lambda x: x[1]['avg_reward'], reverse=True)

        peak_hours = [hour for hour, _ in sorted_hours[:6]]

        # Calculate peak hour multiplier
        non_peak_avg = np.mean([stats['avg_reward']
                               for hour, stats in hourly_stats.items()
                               if hour not in peak_hours])
        peak_avg = np.mean([hourly_stats[hour]['avg_reward'] for hour in peak_hours])

        peak_multiplier = peak_avg / max(non_peak_avg, 1e-6)

        self.peak_hours = set(peak_hours)

        return {
            'peak_hours': peak_hours,
            'hourly_rewards': {h: stats['avg_reward'] for h, stats in hourly_stats.items()},
            'peak_multiplier': peak_multiplier,
            'optimal_hours': sorted_hours[:6]
        }

    def should_predict_now(self, current_hour: Optional[int] = None) -> Tuple[bool, float]:
        """
        Determine if current time is optimal for predictions
        Returns: (should_predict, confidence_multiplier)
        """
        if current_hour is None:
            now = datetime.now(timezone.utc)
            current_hour = (now.hour + self.timezone_offset) % 24

        if not self.peak_hours:
            # Use known peak hours if no analysis available
            is_peak = current_hour in self.known_peak_hours
            confidence = 0.8 if is_peak else 0.4
        else:
            is_peak = current_hour in self.peak_hours
            confidence = 0.9 if is_peak else 0.3

        return is_peak, confidence

    def get_prediction_schedule(self) -> Dict[str, Union[List[int], Dict[str, int]]]:
        """Get optimal prediction schedule"""
        if not self.peak_hours:
            peak_hours = list(self.known_peak_hours)
        else:
            peak_hours = list(self.peak_hours)

        # Calculate predictions per hour based on peak status
        schedule = {}
        for hour in range(24):
            if hour in peak_hours:
                schedule[hour] = {
                    'predictions_per_hour': 15,  # High frequency during peaks
                    'priority': 'high'
                }
            else:
                schedule[hour] = {
                    'predictions_per_hour': 3,   # Low frequency off-peak
                    'priority': 'low'
                }

        return {
            'peak_hours': peak_hours,
            'schedule': schedule,
            'total_daily_predictions': sum([s['predictions_per_hour'] for s in schedule.values()])
        }


def create_adaptive_prediction_system(base_model, timezone_offset: int = 0):
    """Create complete adaptive prediction system"""
    regime_detector = MarketRegimeDetector()
    adaptive_predictor = AdaptivePredictor(base_model, regime_detector)
    peak_optimizer = PeakHourOptimizer(timezone_offset)

    system = {
        'regime_detector': regime_detector,
        'adaptive_predictor': adaptive_predictor,
        'peak_optimizer': peak_optimizer,
        'performance_tracker': adaptive_predictor.regime_performance
    }

    return system


if __name__ == "__main__":
    # Test market regime detection
    print("🧪 Testing Market Regime Detection and Adaptive Strategies")
    print("=" * 60)

    # Create regime detector
    detector = MarketRegimeDetector()

    # Simulate different market conditions
    print("\n📊 Testing Regime Detection:")

    # Bull market data
    bull_prices = np.cumsum(np.random.normal(0.001, 0.01, 200)) + 100
    regime, confidence = detector.detect_regime(recent_prices=bull_prices.tolist())
    print(f"Bull market: {regime} (confidence: {confidence:.2f})")

    # Bear market data
    bear_prices = np.cumsum(np.random.normal(-0.001, 0.01, 200)) + 100
    regime, confidence = detector.detect_regime(recent_prices=bear_prices.tolist())
    print(f"Bear market: {regime} (confidence: {confidence:.2f})")

    # Volatile market data
    volatile_prices = np.cumsum(np.random.normal(0, 0.03, 200)) + 100
    regime, confidence = detector.detect_regime(recent_prices=volatile_prices.tolist())
    print(f"Volatile market: {regime} (confidence: {confidence:.2f})")

    # Ranging market data
    ranging_prices = 100 + np.sin(np.linspace(0, 4*np.pi, 200)) * 2
    regime, confidence = detector.detect_regime(recent_prices=ranging_prices.tolist())
    print(f"Ranging market: {regime} (confidence: {confidence:.2f})")

    # Test peak hour optimizer
    print("\n⏰ Testing Peak Hour Optimization:")
    peak_optimizer = PeakHourOptimizer()

    # Simulate reward data
    for hour in range(24):
        # Higher rewards during known peak hours
        base_reward = 0.001
        if hour in [9, 10, 11, 13, 14, 15]:
            reward = base_reward * (1 + np.random.random() * 0.5)  # 50% bonus
        else:
            reward = base_reward * (0.3 + np.random.random() * 0.4)  # Lower off-peak

        timestamp = datetime.now(timezone.utc).replace(hour=hour)
        peak_optimizer.update_reward_data(reward, timestamp)

    # Analyze peak hours
    analysis = peak_optimizer.analyze_peak_hours()
    print(f"Detected peak hours: {analysis['peak_hours']}")
    print(".2f")
    print(f"Total daily predictions recommended: {peak_optimizer.get_prediction_schedule()['total_daily_predictions']}")

    print("\n✅ Adaptive Prediction System Ready!")
