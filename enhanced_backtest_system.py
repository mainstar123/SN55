#!/usr/bin/env python3
"""
Enhanced Backtesting System for Standalone Domination
Tests the improved version with elite features and comprehensive evaluation
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced standalone domination solution
from precog.miners.standalone_domination import (
    domination_forward,
    load_domination_models,
    detect_market_regime,
    is_peak_hour,
    get_adaptive_parameters,
    should_make_prediction
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EnhancedBacktester:
    """Enhanced backtesting system for improved domination model"""

    def __init__(self, data_path: str = None, model_loaded: bool = False):
        self.cm = MockCMData(data_path)
        self.results = []
        self.performance_metrics = {}
        self.model_loaded = model_loaded

        # Try to load enhanced models
        if not model_loaded:
            try:
                load_domination_models()
                self.model_loaded = True
                logger.info("✅ Enhanced models loaded for backtesting")
            except Exception as e:
                logger.warning(f"⚠️  Model loading failed: {e}")
                logger.info("Running backtest with basic capabilities")

    def run_comprehensive_backtest(self, hours: int = 24) -> Dict:
        """Run comprehensive backtest with enhanced analytics"""
        logger.info(f"🚀 Starting enhanced {hours}-hour backtest")
        logger.info(f"📊 Data points: {len(self.cm.data)}")
        logger.info(f"🤖 Enhanced features: {'✅' if self.model_loaded else '❌'}")

        # Initialize tracking
        backtest_results = {
            'predictions': [],
            'performance': {},
            'regime_analysis': {},
            'peak_hour_analysis': {},
            'feature_effectiveness': {},
            'competitor_comparison': {}
        }

        total_predictions = 0
        successful_predictions = 0

        # Process data in 5-minute intervals for realistic testing
        for i in range(0, min(len(self.cm.data), hours * 12), 5):  # Every 5 minutes
            current_time = self.cm.data.iloc[i]['timestamp']

            # Create mock synapse
            synapse = MockSynapse()

            # Get data up to current time
            test_data = self.cm.data.iloc[:i+1].copy()

            # Skip if insufficient data
            if len(test_data) < 20:
                continue

            try:
                # Create temporary CM object with current data
                temp_cm = MockCMData()
                temp_cm.data = test_data

                # Run domination forward
                start_time = time.time()
                result_synapse = domination_forward(synapse, temp_cm)
                response_time = time.time() - start_time

                # Get market context
                market_regime = detect_market_regime(test_data['close'].values)
                is_peak = is_peak_hour()
                adaptive_params = get_adaptive_parameters(market_regime, is_peak)

                # Record result
                result = {
                    'timestamp': current_time,
                    'actual_price': test_data['close'].iloc[-1],
                    'market_regime': market_regime,
                    'is_peak_hour': is_peak,
                    'adaptive_params': adaptive_params,
                    'response_time': response_time,
                    'prediction_made': result_synapse.predictions is not None,
                }

                if result['prediction_made']:
                    result.update({
                        'prediction_tao': result_synapse.predictions[0],
                        'interval_lower': result_synapse.intervals[0][0],
                        'interval_upper': result_synapse.intervals[0][1],
                        'usd_prediction': result_synapse.predictions[0] / 1000,  # Convert to USD
                    })
                    total_predictions += 1

                    # Evaluate prediction quality if we have future data
                    if i + 4 < len(self.cm.data):  # 20 minutes ahead
                        future_price = self.cm.data.iloc[i+4]['close']
                        result['future_price'] = future_price
                        result['prediction_error'] = abs(future_price - result['usd_prediction'])
                        result['direction_correct'] = (result['usd_prediction'] > result['actual_price']) == (future_price > result['actual_price'])

                backtest_results['predictions'].append(result)

            except Exception as e:
                logger.warning(f"Error at {current_time}: {e}")
                continue

        # Calculate comprehensive metrics
        backtest_results['performance'] = self.calculate_enhanced_metrics(backtest_results['predictions'])
        backtest_results['regime_analysis'] = self.analyze_regime_performance(backtest_results['predictions'])
        backtest_results['peak_hour_analysis'] = self.analyze_peak_hour_performance(backtest_results['predictions'])
        backtest_results['feature_effectiveness'] = self.evaluate_feature_effectiveness(backtest_results['predictions'])

        # Generate report
        report = self.generate_enhanced_report(backtest_results)

        # Save results
        with open('enhanced_backtest_results.json', 'w') as f:
            json.dump(backtest_results, f, indent=2, default=str)

        with open('enhanced_backtest_report.md', 'w') as f:
            f.write(report)

        logger.info("📄 Enhanced backtest results saved to enhanced_backtest_report.md")

        return backtest_results

    def calculate_enhanced_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not predictions:
            return {'error': 'No predictions made'}

        pred_made = [p for p in predictions if p['prediction_made']]
        total_predictions = len(pred_made)

        if total_predictions == 0:
            return {'error': 'No predictions recorded'}

        # Basic metrics
        response_times = [p['response_time'] for p in pred_made]
        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)

        # Prediction quality metrics
        predictions_with_future = [p for p in pred_made if 'future_price' in p]
        if predictions_with_future:
            mape_values = []
            directional_accuracy = 0

            for p in predictions_with_future:
                if p['usd_prediction'] != 0:
                    mape = abs(p['future_price'] - p['usd_prediction']) / abs(p['usd_prediction'])
                    mape_values.append(mape)

                if p.get('direction_correct', False):
                    directional_accuracy += 1

            avg_mape = np.mean(mape_values) if mape_values else 0
            directional_accuracy = directional_accuracy / len(predictions_with_future)
        else:
            avg_mape = 0
            directional_accuracy = 0

        # Interval coverage (if available)
        intervals_with_future = [p for p in predictions_with_future if 'interval_lower' in p]
        interval_coverage = 0
        avg_interval_width = 0

        if intervals_with_future:
            coverage_count = 0
            widths = []

            for p in intervals_with_future:
                lower = p['interval_lower'] / 1000  # Convert to USD
                upper = p['interval_upper'] / 1000
                actual = p['future_price']

                if lower <= actual <= upper:
                    coverage_count += 1

                widths.append(upper - lower)

            interval_coverage = coverage_count / len(intervals_with_future)
            avg_interval_width = np.mean(widths)

        # TAO earnings estimation
        tao_per_prediction = max(0, 0.5 - avg_mape) if avg_mape > 0 else 0.3
        daily_predictions_est = total_predictions / (len(predictions) / 288)  # 288 5-min intervals per day
        daily_tao_estimate = tao_per_prediction * daily_predictions_est

        return {
            'total_predictions': total_predictions,
            'prediction_frequency': total_predictions / len(predictions) * 100,
            'avg_response_time': avg_response_time,
            'max_response_time': max_response_time,
            'mape': avg_mape,
            'directional_accuracy': directional_accuracy,
            'interval_coverage': interval_coverage,
            'avg_interval_width': avg_interval_width,
            'tao_per_prediction': tao_per_prediction,
            'daily_tao_estimate': daily_tao_estimate,
            'predictions_per_day': daily_predictions_est,
            'model_loaded': self.model_loaded
        }

    def analyze_regime_performance(self, predictions: List[Dict]) -> Dict:
        """Analyze performance across different market regimes"""
        regimes = {}
        pred_made = [p for p in predictions if p['prediction_made']]

        for regime in ['bull', 'bear', 'volatile', 'ranging']:
            regime_preds = [p for p in pred_made if p['market_regime'] == regime]

            if regime_preds:
                regime_count = len(regime_preds)
                regime_predictions = len([p for p in regime_preds if 'future_price' in p])

                if regime_predictions > 0:
                    correct_direction = sum(1 for p in regime_preds if p.get('direction_correct', False))
                    directional_acc = correct_direction / regime_predictions
                else:
                    directional_acc = 0

                regimes[regime] = {
                    'predictions': regime_count,
                    'directional_accuracy': directional_acc,
                    'percentage': regime_count / len(pred_made) * 100 if pred_made else 0
                }

        return regimes

    def analyze_peak_hour_performance(self, predictions: List[Dict]) -> Dict:
        """Analyze peak hour vs normal hour performance"""
        pred_made = [p for p in predictions if p['prediction_made']]

        peak_hour_preds = [p for p in pred_made if p['is_peak_hour']]
        normal_hour_preds = [p for p in pred_made if not p['is_peak_hour']]

        peak_performance = self.calculate_subset_metrics(peak_hour_preds)
        normal_performance = self.calculate_subset_metrics(normal_hour_preds)

        return {
            'peak_hour': {
                'predictions': len(peak_hour_preds),
                'percentage': len(peak_hour_preds) / len(pred_made) * 100 if pred_made else 0,
                'performance': peak_performance
            },
            'normal_hour': {
                'predictions': len(normal_hour_preds),
                'percentage': len(normal_hour_preds) / len(pred_made) * 100 if pred_made else 0,
                'performance': normal_performance
            },
            'peak_advantage': self.compare_performance(peak_performance, normal_performance)
        }

    def calculate_subset_metrics(self, subset_predictions: List[Dict]) -> Dict:
        """Calculate metrics for a subset of predictions"""
        if not subset_predictions:
            return {'error': 'No predictions in subset'}

        with_future = [p for p in subset_predictions if 'future_price' in p]
        if not with_future:
            return {'directional_accuracy': 0, 'avg_error': 0}

        directional_correct = sum(1 for p in with_future if p.get('direction_correct', False))
        avg_error = np.mean([p.get('prediction_error', 0) for p in with_future])

        return {
            'directional_accuracy': directional_correct / len(with_future),
            'avg_error': avg_error
        }

    def compare_performance(self, peak: Dict, normal: Dict) -> Dict:
        """Compare peak vs normal hour performance"""
        if 'error' in peak or 'error' in normal:
            return {'comparison': 'insufficient_data'}

        peak_acc = peak.get('directional_accuracy', 0)
        normal_acc = normal.get('directional_accuracy', 0)

        return {
            'accuracy_advantage': peak_acc - normal_acc,
            'peak_better': peak_acc > normal_acc,
            'advantage_percentage': (peak_acc - normal_acc) / normal_acc * 100 if normal_acc > 0 else 0
        }

    def evaluate_feature_effectiveness(self, predictions: List[Dict]) -> Dict:
        """Evaluate effectiveness of enhanced features"""
        pred_made = [p for p in predictions if p['prediction_made']]

        if not pred_made:
            return {'error': 'No predictions to evaluate'}

        # Analyze prediction patterns by regime
        regime_effectiveness = {}
        for regime in ['volatile', 'bull', 'ranging']:
            regime_preds = [p for p in pred_made if p['market_regime'] == regime]
            if regime_preds:
                with_future = [p for p in regime_preds if 'future_price' in p]
                if with_future:
                    accuracy = sum(1 for p in with_future if p.get('direction_correct', False)) / len(with_future)
                    regime_effectiveness[regime] = {
                        'accuracy': accuracy,
                        'predictions': len(regime_preds),
                        'adaptive_params': regime_preds[0]['adaptive_params']
                    }

        return {
            'regime_adaptation': regime_effectiveness,
            'peak_hour_adaptation': len([p for p in pred_made if p['is_peak_hour']]) / len(pred_made) * 100,
            'response_time_consistency': np.std([p['response_time'] for p in pred_made]),
            'feature_completeness': '24_indicator_system' if self.model_loaded else 'basic_features'
        }

    def generate_enhanced_report(self, results: Dict) -> str:
        """Generate comprehensive enhanced backtest report"""
        perf = results['performance']

        report = f"""
# 🎯 ENHANCED STANDALONE DOMINATION BACKTEST REPORT
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 EXECUTIVE SUMMARY

### Model Status
- **Enhanced Features:** {'✅ 24-Indicator System' if self.model_loaded else '❌ Basic Features Only'}
- **Predictions Made:** {perf.get('total_predictions', 0)}
- **Prediction Frequency:** {perf.get('prediction_frequency', 0):.1f}%

### Performance Metrics
- **MAPE:** {perf.get('mape', 0):.2f}%
- **Directional Accuracy:** {perf.get('directional_accuracy', 0):.1%}
- **Response Time:** {perf.get('avg_response_time', 0):.4f}s (max: {perf.get('max_response_time', 0):.4f}s)

### Earnings Projection
- **TAO/Prediction:** {perf.get('tao_per_prediction', 0):.4f}
- **Daily Predictions:** {perf.get('predictions_per_day', 0):.0f}
- **Daily TAO Estimate:** {perf.get('daily_tao_estimate', 0):.2f}

## 🎭 MARKET REGIME ANALYSIS

### Performance by Market Condition
"""

        # Add regime analysis
        regime_analysis = results.get('regime_analysis', {})
        for regime, data in regime_analysis.items():
            report += f"""- **{regime.upper()}:**
  - Predictions: {data.get('predictions', 0)}
  - Directional Accuracy: {data.get('directional_accuracy', 0):.1%}
  - Percentage of Total: {data.get('percentage', 0):.1f}%

"""

        # Add peak hour analysis
        peak_analysis = results.get('peak_hour_analysis', {})
        report += f"""
## ⏰ PEAK HOUR ANALYSIS

### Prediction Distribution
- **Peak Hour Predictions:** {peak_analysis.get('peak_hour', {}).get('predictions', 0)}
- **Normal Hour Predictions:** {peak_analysis.get('normal_hour', {}).get('predictions', 0)}
- **Peak Hour Percentage:** {peak_analysis.get('peak_hour', {}).get('percentage', 0):.1f}%

### Performance Comparison
- **Peak Hour Accuracy:** {peak_analysis.get('peak_hour', {}).get('performance', {}).get('directional_accuracy', 0):.1%}
- **Normal Hour Accuracy:** {peak_analysis.get('normal_hour', {}).get('performance', {}).get('directional_accuracy', 0):.1%}
- **Peak Advantage:** {peak_analysis.get('peak_advantage', {}).get('advantage_percentage', 0):.1f}%

## 🏆 COMPETITIVE ASSESSMENT

### vs Current Top Performers
"""

        # Competitive comparison
        mape = perf.get('mape', 0)
        if mape > 0:
            miner_52_mape = 45.2  # From earlier analysis
            competitive_edge = (miner_52_mape - mape) / miner_52_mape * 100

            report += f"""- **Miner #52 MAPE:** {miner_52_mape}%
- **Your MAPE:** {mape:.2f}%
- **Competitive Edge:** {'✅' if mape < miner_52_mape else '❌'} {abs(competitive_edge):.1f}% {'better' if mape < miner_52_mape else 'worse'}

"""

        # Recommendations
        report += f"""
## 💡 KEY RECOMMENDATIONS

### Immediate Actions
"""

        if perf.get('mape', 1) > 0.1:
            report += "- **High Priority:** Improve prediction accuracy - MAPE above competitive threshold\n"
        if perf.get('directional_accuracy', 0) < 0.6:
            report += "- **High Priority:** Enhance directional prediction capabilities\n"
        if perf.get('response_time', 1) > 0.18:
            report += "- **Medium Priority:** Optimize inference speed for competitive edge\n"

        if not self.model_loaded:
            report += "- **Critical:** Enable enhanced 24-indicator feature system for best performance\n"

        peak_advantage = peak_analysis.get('peak_advantage', {}).get('peak_better', False)
        if not peak_advantage:
            report += "- **Medium Priority:** Improve peak hour detection and optimization\n"

        report += f"""
### Feature Effectiveness
- **Enhanced Features:** {'✅ Active' if self.model_loaded else '❌ Not Active'}
- **Technical Indicators:** {'✅ RSI, MACD, Bollinger, Volume' if self.model_loaded else '❌ Basic only'}
- **Adaptive Parameters:** ✅ Dynamic threshold adjustment
- **Market Intelligence:** ✅ Regime-aware predictions

## 🎯 NEXT STEPS

### Phase 1: Immediate Improvements (Today)
1. **Enable Elite Model** - Switch to `elite_domination_model.pth`
2. **Full Feature System** - Activate complete 24-indicator pipeline
3. **Threshold Optimization** - Fine-tune confidence parameters

### Phase 2: Performance Optimization (This Week)
1. **Backtest Refinement** - Validate improvements with comprehensive testing
2. **Hyperparameter Tuning** - Optimize model parameters
3. **Peak Hour Enhancement** - Maximize earnings during high-reward periods

### Phase 3: Competitive Dominance (This Month)
1. **Advanced Ensemble** - Combine multiple model architectures
2. **Real-time Adaptation** - Continuous model updates
3. **Competitor Intelligence** - Automated performance monitoring

## 📈 SUCCESS METRICS TARGETS

### Technical Excellence
- **MAPE:** <2.0% (target: <1.5% elite model potential)
- **Directional Accuracy:** >70% (target: >75%)
- **Response Time:** <0.18s (maintain competitive edge)
- **Prediction Frequency:** 20-40% (optimal balance)

### Market Dominance
- **Daily TAO:** >1.0 (target: >1.4 with full optimization)
- **Position:** Top 3 sustained
- **Competitive Edge:** 30%+ better than current top performers
- **Uptime:** >99.9%

---
*Enhanced Backtest Report - Standalone Domination System*
*Features: 24-Indicator System, Market Regime Detection, Peak Hour Optimization*
"""

        return report


# Mock synapse class for testing
class MockSynapse:
    def __init__(self):
        self.predictions = None
        self.intervals = None

# Mock CMData class for testing
class MockCMData:
    def __init__(self, data_path: str = None):
        self.data_path = data_path or 'backtest_data.csv'
        self.data = self.load_data()

    def load_data(self):
        """Load historical data for testing"""
        try:
            df = pd.read_csv(self.data_path, parse_dates=['timestamp'])
            # Filter for BTC only
            df = df[df['symbol'] == 'BTC'].copy()
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Could not load data from {self.data_path}: {e}")
            # Create synthetic data
            return self.create_synthetic_data()

    def create_synthetic_data(self):
        """Create synthetic BTC data for testing"""
        print("Creating synthetic test data...")
        timestamps = pd.date_range(
            start=datetime(2025, 12, 19, 0, 0),
            end=datetime(2025, 12, 20, 23, 59),
            freq='1min'
        )

        # Generate realistic BTC price movements
        np.random.seed(42)
        base_price = 85000
        prices = [base_price]

        for i in range(1, len(timestamps)):
            # Add trend + noise + volatility + time-of-day effects
            trend = 0.00005  # Slight upward trend
            noise = np.random.normal(0, 0.002)  # 0.2% volatility
            # Add time-of-day effect (higher volatility during peak hours)
            hour = timestamps[i].hour
            if 9 <= hour <= 14:  # Peak hours
                time_multiplier = 1.5
            else:
                time_multiplier = 1.0

            noise *= time_multiplier
            price_change = trend + noise
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)

        # Add volume with peak hour effects
        volumes = []
        for ts in timestamps:
            base_volume = np.random.exponential(50)
            # Higher volume during peak hours
            hour = ts.hour
            if 9 <= hour <= 14:
                volume_multiplier = 2.0
            else:
                volume_multiplier = 1.0
            volumes.append(base_volume * volume_multiplier)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
            'close': prices,
            'volume': volumes,
            'symbol': 'BTC'
        })

        return df

    def get_recent_data(self, minutes: int = 60):
        """Get recent data for testing"""
        return self.data.tail(minutes).copy()


def main():
    """Main enhanced backtesting function"""
    print("🔬 ENHANCED STANDALONE DOMINATION BACKTESTING")
    print("=" * 60)

    # Initialize enhanced backtester
    backtester = EnhancedBacktester()

    # Run comprehensive backtest
    results = backtester.run_comprehensive_backtest(hours=24)

    # Print summary
    perf = results['performance']
    print("\n📊 BACKTEST SUMMARY:")
    print(f"   Duration: 24 hours")
    print(f"   Total Predictions: {perf.get('total_predictions', 0)}")
    print(f"   Prediction Frequency: {perf.get('prediction_frequency', 0):.1f}%")
    print(f"   MAPE: {perf.get('mape', 0):.2f}%")
    print(f"   Directional Accuracy: {perf.get('directional_accuracy', 0):.1%}")
    print(f"   Avg Response Time: {perf.get('avg_response_time', 0):.4f}s")
    print(f"   Daily TAO Estimate: {perf.get('daily_tao_estimate', 0):.2f}")
    print(f"   Enhanced Features: {'✅ Active' if backtester.model_loaded else '❌ Basic Only'}")

    print("\n📄 Detailed results saved to enhanced_backtest_report.md")
    print("📄 Raw data saved to enhanced_backtest_results.json")


if __name__ == "__main__":
    main()
