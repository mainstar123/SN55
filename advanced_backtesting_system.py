#!/usr/bin/env python3
"""
Advanced Backtesting System for Precog #1 Miner Domination System
Test all advanced features on historical data before mainnet deployment
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all our advanced systems
from advanced_ensemble_model import create_advanced_ensemble
from advanced_attention_mechanisms import create_enhanced_attention_ensemble
from temporal_convolutional_networks import create_advanced_tcn_ensemble
from graph_neural_networks import create_advanced_gnn_ensemble
from advanced_feature_engineering import create_comprehensive_feature_set
from market_regime_detector import create_adaptive_prediction_system
from peak_hour_optimizer import create_ultra_precise_prediction_system
from reinforcement_learning_timing import create_rl_prediction_system
from performance_tracking_system import create_performance_tracking_system

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtesting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AdvancedBacktestingEngine:
    """
    Comprehensive backtesting engine for advanced Precog mining strategies
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Models to test
        self.models = {}
        self.model_results = {}

        # Market data
        self.price_data = None
        self.features = None
        self.targets = None

        # Performance tracking
        self.backtest_results = {}

        logger.info(f"Advanced Backtesting Engine initialized on {self.device}")

    def load_historical_data(self, data_path: Optional[str] = None) -> bool:
        """Load historical price data for backtesting"""
        logger.info("Loading historical data for backtesting...")

        if data_path and os.path.exists(data_path):
            # Load from file
            self.price_data = pd.read_csv(data_path)
            logger.info(f"Loaded {len(self.price_data)} historical records from {data_path}")
        else:
            # Generate synthetic historical data
            logger.info("Generating synthetic historical data (30 days, 5-min intervals)")
            self._generate_synthetic_historical_data()

        # Preprocess data
        self._preprocess_data()

        return True

    def _generate_synthetic_historical_data(self):
        """Generate realistic synthetic historical data"""
        # 30 days of 5-minute data = 30 * 24 * 12 = 8640 data points
        n_points = 8640
        start_date = datetime.now(timezone.utc) - timedelta(days=30)

        # Generate timestamps
        timestamps = [start_date + timedelta(minutes=5*i) for i in range(n_points)]

        # Generate realistic price series with trends, volatility, and seasonality
        np.random.seed(42)

        # Base price around $100
        base_price = 100.0

        # Long-term trend (slight upward)
        trend = np.linspace(0, 2.0, n_points)  # +2% over 30 days

        # Daily seasonality (higher volatility during peak hours)
        hours = np.array([ts.hour for ts in timestamps])
        daily_pattern = np.where((hours >= 9) & (hours <= 15), 1.2, 0.8)  # Peak hours

        # Weekly pattern (higher volatility mid-week)
        weekdays = np.array([ts.weekday() for ts in timestamps])
        weekly_pattern = np.where((weekdays >= 0) & (weekdays <= 4), 1.1, 0.9)

        # Market regime changes (bull/bear/volatile periods)
        regime_changes = np.zeros(n_points)
        regime_changes[1000:2000] = 1.5  # Bull market
        regime_changes[3000:4000] = -1.2  # Bear market
        regime_changes[5000:6000] = 2.5  # Volatile market

        # Generate price returns
        base_volatility = 0.001  # 0.1% per 5 minutes
        volatility = base_volatility * daily_pattern * weekly_pattern

        # Add regime-specific volatility
        regime_volatility = np.abs(regime_changes) * 0.002
        total_volatility = volatility + regime_volatility

        # Generate returns
        returns = np.random.normal(trend * 0.0001, total_volatility, n_points)
        returns += regime_changes * 0.0005  # Add regime directional bias

        # Convert to prices
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        high_prices = prices * (1 + np.abs(np.random.normal(0, total_volatility * 2, n_points)))
        low_prices = prices * (1 - np.abs(np.random.normal(0, total_volatility * 2, n_points)))
        open_prices = np.roll(prices, 1)
        open_prices[0] = base_price

        # Volume correlated with volatility
        base_volume = 10000
        volume = base_volume * (1 + total_volatility * 1000) * np.random.lognormal(0, 0.5, n_points)

        # Create DataFrame
        self.price_data = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': prices,
            'volume': volume.astype(int)
        })

        logger.info(f"Generated synthetic data: {len(self.price_data)} points, {prices.min():.2f}-{prices.max():.2f} price range")

    def _preprocess_data(self):
        """Preprocess price data into features and targets"""
        logger.info("Preprocessing data into features and targets...")

        # Convert to numpy arrays for feature engineering
        price_dict = {
            'open': self.price_data['open'].values,
            'high': self.price_data['high'].values,
            'low': self.price_data['low'].values,
            'close': self.price_data['close'].values,
            'volume': self.price_data['volume'].values
        }

        # Create comprehensive features
        features_array, feature_names = create_comprehensive_feature_set(price_dict)
        self.features = features_array
        self.feature_names = feature_names

        # Create targets (next period price change)
        close_prices = self.price_data['close'].values
        next_prices = np.roll(close_prices, -1)
        next_prices[-1] = close_prices[-1]  # Last value stays the same

        # Target: percentage price change
        self.targets = (next_prices - close_prices) / close_prices

        logger.info(f"Created {self.features.shape[1]} features and {len(self.targets)} targets")
        logger.info(f"Features: {self.feature_names[:10]}...")

    def initialize_models(self):
        """Initialize all models for backtesting"""
        logger.info("Initializing models for backtesting...")

        # 1. Original Ensemble (baseline)
        self.models['original_ensemble'] = create_advanced_ensemble()

        # 2. Attention-Enhanced Ensemble
        self.models['attention_ensemble'] = create_enhanced_attention_ensemble()

        # 3. TCN Ensemble
        self.models['tcn_ensemble'] = create_advanced_tcn_ensemble()

        # 4. GNN Ensemble
        self.models['gnn_ensemble'] = create_advanced_gnn_ensemble()

        # 5. Full Advanced System (combination)
        self.models['full_advanced'] = create_enhanced_attention_ensemble()  # Using attention as base

        logger.info(f"Initialized {len(self.models)} models for backtesting")

    def run_backtest(self, model_name: str, prediction_system: Optional[str] = None) -> Dict:
        """Run backtest for a specific model"""
        logger.info(f"Running backtest for {model_name}...")

        model = self.models[model_name]
        model.eval()

        # Initialize systems
        market_system = None
        peak_system = None
        performance_tracker = None

        if prediction_system:
            if 'adaptive' in prediction_system.lower():
                market_system = create_adaptive_prediction_system(model)
            if 'peak' in prediction_system.lower():
                peak_system = create_ultra_precise_prediction_system()
            if 'performance' in prediction_system.lower():
                performance_tracker, _ = create_performance_tracking_system(model)

        # Backtesting parameters
        seq_len = self.config.get('seq_len', 60)
        step_size = self.config.get('step_size', 5)  # Predict every 5 periods
        confidence_threshold = self.config.get('confidence_threshold', 0.6)

        # Results storage
        predictions = []
        actuals = []
        confidences = []
        decisions = []
        rewards = []

        # Simulate trading
        for i in range(seq_len, len(self.features) - 1, step_size):
            # Prepare input sequence
            input_seq = self.features[i-seq_len:i]  # (seq_len, n_features)
            current_features = self.features[i]  # Current market state

            # Make prediction
            with torch.no_grad():
                if model_name in ['tcn_ensemble']:
                    # TCN expects different input format
                    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
                else:
                    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)

                pred, uncertainty = model(input_tensor)
                prediction = pred.squeeze().cpu().numpy()
                confidence = 1 - uncertainty.squeeze().cpu().numpy()

            # Decision logic
            should_predict = confidence > confidence_threshold

            # Additional system checks
            if market_system and should_predict:
                decision_info = market_system['adaptive_predictor'].should_make_prediction(
                    market_data=current_features.tolist()
                )
                should_predict = decision_info['should_predict']

            if peak_system and should_predict:
                is_peak, _, _ = peak_system.should_predict_now(self.price_data.iloc[i]['timestamp'])
                should_predict = should_predict and is_peak

            # Record results
            actual = self.targets[i]
            predictions.append(prediction)
            actuals.append(actual)
            confidences.append(confidence)
            decisions.append(should_predict)

            # Calculate reward (simplified)
            if should_predict:
                # Reward based on prediction accuracy and confidence
                accuracy_reward = 1 - abs(prediction - actual)
                confidence_bonus = confidence * 0.1
                total_reward = accuracy_reward + confidence_bonus
                rewards.append(total_reward)
            else:
                rewards.append(0.0)  # No reward for not predicting

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        confidences = np.array(confidences)
        decisions = np.array(decisions)
        rewards = np.array(rewards)

        # Metrics
        mape = np.mean(np.abs((predictions - actuals) / (np.abs(actuals) + 1e-6)))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))

        # Directional accuracy
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        directional_acc = np.mean(pred_direction == actual_direction)

        # Confidence-weighted metrics
        high_conf_mask = confidences > 0.8
        if np.any(high_conf_mask):
            high_conf_mape = np.mean(np.abs((predictions[high_conf_mask] - actuals[high_conf_mask]) /
                                          (np.abs(actuals[high_conf_mask]) + 1e-6)))
        else:
            high_conf_mape = mape

        # Trading metrics
        prediction_rate = np.mean(decisions)
        total_rewards = np.sum(rewards)
        avg_reward_per_prediction = total_rewards / max(1, np.sum(decisions))
        reward_per_period = total_rewards / len(rewards)

        results = {
            'model_name': model_name,
            'prediction_system': prediction_system,
            'total_predictions': len(predictions),
            'prediction_rate': prediction_rate,

            # Accuracy metrics
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_acc,
            'high_conf_mape': high_conf_mape,

            # Reward metrics
            'total_rewards': total_rewards,
            'avg_reward_per_prediction': avg_reward_per_prediction,
            'reward_per_period': reward_per_period,

            # Confidence metrics
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),

            # Detailed data
            'predictions': predictions.tolist()[:100],  # First 100 for analysis
            'actuals': actuals.tolist()[:100],
            'confidences': confidences.tolist()[:100],
            'decisions': decisions.tolist()[:100],
            'rewards': rewards.tolist()[:100]
        }

        self.model_results[model_name] = results
        logger.info(".4f"
        return results

    def run_comprehensive_backtest(self) -> Dict:
        """Run comprehensive backtesting across all models and systems"""
        logger.info("🚀 Starting comprehensive backtesting...")

        # Test scenarios
        scenarios = [
            ('original_ensemble', None),
            ('attention_ensemble', None),
            ('tcn_ensemble', None),
            ('gnn_ensemble', None),
            ('attention_ensemble', 'adaptive'),
            ('attention_ensemble', 'adaptive+peak'),
            ('full_advanced', 'adaptive+peak+performance')
        ]

        all_results = {}

        for model_name, prediction_system in scenarios:
            logger.info(f"\n🎯 Testing: {model_name} + {prediction_system}")
            try:
                results = self.run_backtest(model_name, prediction_system)
                scenario_key = f"{model_name}_{prediction_system or 'basic'}"
                all_results[scenario_key] = results
            except Exception as e:
                logger.error(f"Failed to test {model_name} + {prediction_system}: {e}")
                continue

        # Generate comparative analysis
        self.backtest_results = all_results
        analysis = self._analyze_backtest_results(all_results)

        return analysis

    def _analyze_backtest_results(self, results: Dict) -> Dict:
        """Analyze and compare backtest results"""
        logger.info("📊 Analyzing backtest results...")

        # Extract key metrics
        summary = {}
        for scenario, data in results.items():
            summary[scenario] = {
                'mape': data['mape'],
                'directional_accuracy': data['directional_accuracy'],
                'total_rewards': data['total_rewards'],
                'reward_per_period': data['reward_per_period'],
                'prediction_rate': data['prediction_rate'],
                'avg_confidence': data['avg_confidence']
            }

        # Create comparison DataFrame
        df = pd.DataFrame(summary).T

        # Calculate improvements over baseline
        baseline_rewards = df.loc['original_ensemble_basic', 'reward_per_period']
        df['improvement_over_baseline'] = (df['reward_per_period'] - baseline_rewards) / abs(baseline_rewards)

        # Rank models
        df['rank'] = df['reward_per_period'].rank(ascending=False)

        # Best performers
        best_model = df['reward_per_period'].idxmax()
        best_reward = df.loc[best_model, 'reward_per_period']
        baseline_reward = df.loc['original_ensemble_basic', 'reward_per_period']
        improvement = (best_reward - baseline_reward) / abs(baseline_reward) * 100

        analysis = {
            'summary': df.to_dict(),
            'best_model': best_model,
            'best_reward_per_period': best_reward,
            'baseline_reward_per_period': baseline_reward,
            'improvement_percentage': improvement,
            'rankings': df.sort_values('rank')['rank'].to_dict(),
            'recommendation': self._generate_recommendation(df)
        }

        return analysis

    def _generate_recommendation(self, df: pd.DataFrame) -> str:
        """Generate deployment recommendation based on results"""
        best_model = df['reward_per_period'].idxmax()

        if df.loc[best_model, 'improvement_over_baseline'] > 0.5:  # 50% improvement
            return f"🚀 DEPLOY {best_model} - {df.loc[best_model, 'improvement_over_baseline']:.1%} improvement over baseline"
        elif df.loc[best_model, 'improvement_over_baseline'] > 0.2:  # 20% improvement
            return f"✅ CONSIDER {best_model} - {df.loc[best_model, 'improvement_over_baseline']:.1%} improvement, worth deploying"
        else:
            return f"⚠️  STICK WITH BASELINE - limited improvement from advanced features"

    def save_backtest_results(self, filename: str = 'backtest_results.json'):
        """Save comprehensive backtest results"""
        results = {
            'config': self.config,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_results': self.model_results,
            'backtest_analysis': self.backtest_results,
            'data_stats': {
                'total_periods': len(self.price_data) if self.price_data is not None else 0,
                'features_count': self.features.shape[1] if self.features is not None else 0,
                'price_range': {
                    'min': float(self.price_data['close'].min()) if self.price_data is not None else None,
                    'max': float(self.price_data['close'].max()) if self.price_data is not None else None
                }
            }
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Backtest results saved to {filename}")

    def generate_report(self) -> str:
        """Generate comprehensive backtest report"""
        if not self.backtest_results:
            return "No backtest results available. Run backtesting first."

        analysis = self._analyze_backtest_results(self.backtest_results)

        report = ".1f"".1f"".1f"".4f"".4f"".4f"".4f"".4f"".4f"f"""
🎯 PRECOG DOMINATION SYSTEM BACKTEST REPORT
{'='*60}

📊 BACKTEST SUMMARY
Total Test Periods: {len(self.price_data) if self.price_data is not None else 'N/A'}
Features Used: {self.features.shape[1] if self.features is not None else 'N/A'}

🏆 BEST PERFORMER: {analysis['best_model']}
Reward/Period: {analysis['best_reward_per_period']:.4f}
vs Baseline: {analysis['improvement_percentage']:+.1f}%

📈 MODEL COMPARISON:
"""

        df = pd.DataFrame(analysis['summary']).T
        df_sorted = df.sort_values('reward_per_period', ascending=False)

        for idx, row in df_sorted.iterrows():
            report += f"{idx:<30} | {row['reward_per_period']:.4f} | {row['improvement_over_baseline']:+.1%}\n"

        report += f"""
💡 RECOMMENDATION:
{analysis['recommendation']}

🎯 DEPLOYMENT READY:
• Best model: {analysis['best_model']}
• Expected improvement: {analysis['improvement_percentage']:.1f}%
• Confidence: {'HIGH' if analysis['improvement_percentage'] > 50 else 'MEDIUM' if analysis['improvement_percentage'] > 20 else 'LOW'}
"""

        return report


def main():
    """Main backtesting function"""
    import argparse

    parser = argparse.ArgumentParser(description='Advanced Backtesting for Precog Domination System')
    parser.add_argument('--data_path', type=str, help='Path to historical data CSV')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length for models')
    parser.add_argument('--step_size', type=int, default=5, help='Prediction step size')
    parser.add_argument('--confidence_threshold', type=float, default=0.6, help='Prediction confidence threshold')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--output', type=str, default='backtest_results.json', help='Output file')

    args = parser.parse_args()

    config = {
        'data_path': args.data_path,
        'seq_len': args.seq_len,
        'step_size': args.step_size,
        'confidence_threshold': args.confidence_threshold,
        'device': args.device
    }

    print("🎯 PRECOG DOMINATION SYSTEM - PHASE 2 BACKTESTING")
    print("=" * 60)

    # Initialize backtesting engine
    engine = AdvancedBacktestingEngine(config)

    try:
        # Load data
        print("\n📊 Loading historical data...")
        engine.load_historical_data(config['data_path'])

        # Initialize models
        print("\n🧠 Initializing models...")
        engine.initialize_models()

        # Run comprehensive backtesting
        print("\n🚀 Running comprehensive backtesting...")
        print("This may take several minutes...")

        analysis = engine.run_comprehensive_backtest()

        # Save results
        print("\n💾 Saving results...")
        engine.save_backtest_results(args.output)

        # Generate and display report
        print("\n📋 BACKTEST REPORT:")
        print("=" * 60)
        report = engine.generate_report()
        print(report)

        print("
✅ BACKTESTING COMPLETE!"        print(f"Results saved to: {args.output}")
        print("
🎯 READY FOR DEPLOYMENT DECISION!"        # Exit with success
        return 0

    except Exception as e:
        print(f"\n❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

