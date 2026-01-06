#!/usr/bin/env python3
"""
Comprehensive backtesting script for the domination model
Evaluates simple working model performance on historical data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model and utilities
from simple_working_model import load_simple_model, predict_with_simple_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_domination.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DominationBacktester:
    """Backtesting framework for domination model evaluation"""

    def __init__(self, model_path: str = 'simple_working_model.pth'):
        self.model = None
        self.scaler = None
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load the model for backtesting"""
        logger.info(f"Loading model from {model_path}")
        try:
            self.model = load_simple_model(model_path, 'cpu')
            if hasattr(self.model, 'feature_scaler'):
                self.scaler = self.model.feature_scaler
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def create_features(self, df: pd.DataFrame, lookback: int = 24) -> List[List[float]]:
        """Create technical indicators and features for prediction"""
        features_list = []

        # Ensure we have enough data
        if len(df) < lookback:
            logger.warning(f"Insufficient data: {len(df)} < {lookback}")
            return features_list

        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i]

            # Basic price features
            close_prices = window['close'].values
            high_prices = window['high'].values
            low_prices = window['low'].values
            volume = window['volume'].values

            features = []

            # Price-based features
            features.append(close_prices[-1])  # Current close
            features.append(np.mean(close_prices))  # SMA
            features.append(np.std(close_prices))  # Volatility
            features.append((close_prices[-1] - close_prices[0]) / close_prices[0])  # Return
            features.append(np.max(high_prices) / np.min(low_prices) - 1)  # Range

            # Volume features
            features.append(volume[-1])  # Current volume
            features.append(np.mean(volume))  # Average volume
            features.append(volume[-1] / np.mean(volume) if np.mean(volume) > 0 else 0)  # Volume ratio

            # Momentum indicators
            if len(close_prices) >= 5:
                features.append((close_prices[-1] - close_prices[-5]) / close_prices[-5])  # 5-period return
            else:
                features.append(0.0)

            # RSI-like indicator (simplified)
            gains = np.maximum(np.diff(close_prices), 0)
            losses = np.maximum(-np.diff(close_prices), 0)
            if len(losses) > 0 and np.sum(losses) > 0:
                rs = np.sum(gains) / np.sum(losses)
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi / 100.0)  # Normalize
            else:
                features.append(0.5)

            # Bollinger Bands (simplified)
            sma = np.mean(close_prices)
            std = np.std(close_prices)
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            features.append((close_prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5)

            # MACD-like (simplified)
            if len(close_prices) >= 12:
                ema12 = pd.Series(close_prices).ewm(span=12).mean().iloc[-1]
                ema26 = pd.Series(close_prices).ewm(span=26).mean().iloc[-1]
                macd = ema12 - ema26
                features.append(macd / close_prices[-1])  # Normalize
            else:
                features.append(0.0)

            # Pad to ensure we have 24 features
            while len(features) < 24:
                features.append(0.0)

            features_list.append(features[:24])  # Ensure exactly 24 features

        return features_list

    def run_backtest(self, data_path: str, symbol: str = 'BTC', prediction_hours: int = 1) -> Dict:
        """Run backtest on historical data"""
        logger.info(f"Starting backtest for {symbol} with {prediction_hours}h predictions")
        logger.info(f"Loading data from {data_path}")

        # Load data
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        df = df[df['symbol'] == symbol].copy()

        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")

        logger.info(f"Loaded {len(df)} data points for {symbol}")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Create features
        features_list = self.create_features(df)
        logger.info(f"Created {len(features_list)} feature sets")

        if not features_list:
            raise ValueError("No features created - insufficient data")

        # Make predictions and collect results
        predictions = []
        actuals = []
        timestamps = []

        for i, features in enumerate(features_list):
            try:
                # Make prediction
                point_pred, lower_bound, upper_bound = predict_with_simple_model(self.model, features, self.scaler)

                # Get actual future price (prediction_hours ahead)
                current_idx = i + 24  # Offset by lookback window
                future_idx = current_idx + prediction_hours * 4  # 4 data points per hour (15min intervals)

                if future_idx < len(df):
                    actual_price = df.iloc[future_idx]['close']
                    predicted_price = point_pred

                    predictions.append(predicted_price)
                    actuals.append(actual_price)
                    timestamps.append(df.iloc[current_idx]['timestamp'])

            except Exception as e:
                logger.warning(f"Error making prediction {i}: {e}")
                continue

        logger.info(f"Completed {len(predictions)} predictions")

        # Calculate metrics
        if not predictions:
            raise ValueError("No predictions generated")

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Basic metrics
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        mae = np.mean(np.abs(actuals - predictions))

        # Directional accuracy
        pred_direction = np.sign(predictions[1:] - predictions[:-1])
        actual_direction = np.sign(actuals[1:] - actuals[:-1])
        directional_accuracy = np.mean(pred_direction == actual_direction)

        # Additional metrics
        mean_prediction = np.mean(predictions)
        mean_actual = np.mean(actuals)
        prediction_std = np.std(predictions)
        actual_std = np.std(actuals)

        # TAO earnings estimation (simplified)
        # Based on typical Precog subnet scoring
        tao_per_prediction = max(0, 0.5 - mape/200)  # Rough approximation
        daily_predictions = 24 * 6  # 24 hours * 6 predictions per hour (every 10 min)
        daily_tao_estimate = tao_per_prediction * daily_predictions

        results = {
            'symbol': symbol,
            'total_predictions': len(predictions),
            'test_period_days': (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600),
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'mean_prediction': mean_prediction,
            'mean_actual': mean_actual,
            'prediction_std': prediction_std,
            'actual_std': actual_std,
            'tao_per_prediction': tao_per_prediction,
            'daily_tao_estimate': daily_tao_estimate,
            'timestamp_range': [timestamps[0], timestamps[-1]],
            'model_performance': self._assess_performance(mape, directional_accuracy, daily_tao_estimate)
        }

        return results

    def _assess_performance(self, mape: float, directional_accuracy: float, daily_tao: float) -> Dict:
        """Assess overall model performance"""
        # Performance tiers based on typical Precog requirements
        if mape < 1.0 and directional_accuracy > 0.75 and daily_tao > 1.0:
            rating = "EXCELLENT"
            competitiveness = "Top tier - ready for mainnet domination"
            score = 95
        elif mape < 2.0 and directional_accuracy > 0.65 and daily_tao > 0.5:
            rating = "VERY GOOD"
            competitiveness = "Competitive - good for mainnet"
            score = 85
        elif mape < 5.0 and directional_accuracy > 0.55 and daily_tao > 0.2:
            rating = "GOOD"
            competitiveness = "Viable for mainnet with monitoring"
            score = 75
        elif mape < 10.0 and directional_accuracy > 0.50:
            rating = "FAIR"
            competitiveness = "Needs improvement but workable"
            score = 60
        else:
            rating = "NEEDS IMPROVEMENT"
            competitiveness = "Not competitive - requires significant optimization"
            score = 40

        return {
            'rating': rating,
            'competitiveness': competitiveness,
            'score': score,
            'recommendations': self._get_recommendations(mape, directional_accuracy, daily_tao)
        }

    def _get_recommendations(self, mape: float, directional_accuracy: float, daily_tao: float) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        if mape > 5.0:
            recommendations.append("Improve accuracy: MAPE is too high for competitive performance")
        if directional_accuracy < 0.55:
            recommendations.append("Enhance directional prediction: accuracy below 55%")
        if daily_tao < 0.5:
            recommendations.append("Increase earning potential: current TAO/day too low")
        if len(recommendations) == 0:
            recommendations.append("Model is performing well - focus on fine-tuning and monitoring")

        return recommendations

    def print_results(self, results: Dict):
        """Print formatted backtest results"""
        print("\n" + "="*80)
        print("🎯 DOMINATION MODEL BACKTEST RESULTS")
        print("="*80)

        print(f"\n📊 Asset: {results['symbol']}")
        print(f"📅 Test Period: {results['timestamp_range'][0]} to {results['timestamp_range'][1]}")
        print(f"📈 Days Tested: {results['test_period_days']:.1f}")
        print(f"🎯 Predictions Made: {results['total_predictions']}")

        print(f"\n📈 ACCURACY METRICS:")
        print(f"   MAPE: {results['mape']:.2f}%")
        print(f"   RMSE: ${results['rmse']:.6f}")
        print(f"   MAE:  ${results['mae']:.6f}")
        print(f"   Directional Acc: {results['directional_accuracy']:.1%}")
        print(f"   Mean Prediction: ${results['mean_prediction']:.2f}")
        print(f"   Mean Actual:     ${results['mean_actual']:.2f}")
        print(f"   Prediction Std:  ${results['prediction_std']:.2f}")
        print(f"   Actual Std:      ${results['actual_std']:.2f}")

        print(f"\n💰 EARNINGS ESTIMATION:")
        print(f"   TAO/Prediction: {results['tao_per_prediction']:.4f}")
        print(f"   Daily TAO Est: {results['daily_tao_estimate']:.2f}")
        print(f"\n🏆 PERFORMANCE RATING:")
        perf = results['model_performance']
        print(f"   Rating: {perf['rating']} ({perf['score']}/100)")
        print(f"   Competitiveness: {perf['competitiveness']}")

        if perf['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in perf['recommendations']:
                print(f"   • {rec}")

        print("\n" + "="*80)


def main():
    """Main backtesting function"""
    logger.info("🚀 Starting Domination Model Backtest")

    # Initialize backtester
    backtester = DominationBacktester()

    # Test on BTC data
    data_files = [
        ('backtest_data.csv', 'BTC'),
        ('final_backtest_data.csv', 'BTC'),
        ('final_test_BTC.csv', 'BTC')
    ]

    all_results = []

    for data_file, symbol in data_files:
        if os.path.exists(data_file):
            logger.info(f"\n🔍 Testing on {data_file} ({symbol})")
            try:
                results = backtester.run_backtest(data_file, symbol)
                backtester.print_results(results)
                all_results.append(results)
            except Exception as e:
                logger.error(f"❌ Backtest failed for {data_file}: {e}")
        else:
            logger.warning(f"⚠️  Data file {data_file} not found, skipping")

    # Summary across all tests
    if all_results:
        print("\n" + "="*80)
        print("📊 BACKTEST SUMMARY ACROSS ALL DATASETS")
        print("="*80)

        avg_mape = np.mean([r['mape'] for r in all_results])
        avg_dir_acc = np.mean([r['directional_accuracy'] for r in all_results])
        avg_daily_tao = np.mean([r['daily_tao_estimate'] for r in all_results])

        print(f"   Average MAPE: {avg_mape:.2f}%")
        print(f"   Average Directional Acc: {avg_dir_acc:.1%}")
        print(f"   Average Daily TAO: {avg_daily_tao:.2f}")
        # Overall assessment
        if avg_mape < 3.0 and avg_dir_acc > 0.60 and avg_daily_tao > 0.3:
            print("🎉 OVERALL: Model shows COMPETITIVE potential!")
            print("💡 Ready for mainnet deployment with monitoring")
        else:
            print("⚠️  OVERALL: Model needs improvement before mainnet")
            print("💡 Focus on accuracy and directional prediction")

        print("="*80)


if __name__ == "__main__":
    main()
