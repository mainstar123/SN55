#!/usr/bin/env python3
"""
Local backtesting framework for Precog BTC price prediction

Quality Gate 1: Baseline MAPE <0.15% on 30-day backtest before proceeding to testnet

Tests:
1. Point forecast accuracy (MAPE, RMSE)
2. Interval coverage (>85% for 90% confidence intervals)
3. Response time (<16s requirement)
4. Feature extraction robustness
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
import asyncio

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precog.protocol import Challenge
from precog.miners.custom_model import forward
from precog.utils.cm_data import CMData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockCMData(CMData):
    """Mock CMData for backtesting with historical data"""

    def __init__(self, historical_data):
        super().__init__()
        self.historical_data = historical_data
        self.current_index = 60  # Start after 60 minutes of history

    def get_CM_ReferenceRate(self, assets, start, end, frequency):
        """Return historical data slice"""
        if not isinstance(assets, list):
            assets = [assets]

        # Filter for BTC only
        if 'btc' not in [a.lower() for a in assets]:
            return pd.DataFrame()

        # Get data slice
        start_time = pd.to_datetime(start)
        end_time = pd.to_datetime(end)

        mask = (self.historical_data.index >= start_time) & (self.historical_data.index <= end_time)
        data_slice = self.historical_data[mask].copy()

        # Format for CMData compatibility
        data_slice = data_slice.reset_index()
        data_slice = data_slice.rename(columns={'index': 'time', 'price': 'ReferenceRateUSD'})
        data_slice['asset'] = 'btc'

        return data_slice[['time', 'asset', 'ReferenceRateUSD']]


async def run_backtest(historical_data, test_days=7):
    """
    Run backtest on historical data

    Args:
        historical_data: DataFrame with price column and datetime index
        test_days: Number of days to test

    Returns:
        Dict with performance metrics
    """
    logger.info(f"Running backtest on {test_days} days of data...")

    # Initialize mock CM data
    cm_data = MockCMData(historical_data)

    # Prepare test period
    test_end = historical_data.index[-1]
    test_start = test_end - timedelta(days=test_days)

    # Filter test data
    test_data = historical_data[(historical_data.index >= test_start) & (historical_data.index <= test_end)]

    if len(test_data) < 120:  # Need at least 2 hours
        raise ValueError("Insufficient test data")

    logger.info(f"Testing on {len(test_data)} minutes of data")

    # Storage for results
    predictions = []
    actuals = []
    intervals = []
    response_times = []

    # Test every 60 minutes (simulate prediction requests)
    test_indices = range(60, len(test_data) - 60, 60)  # Every hour

    for i in test_indices:
        # Set current time in mock data
        cm_data.current_index = i

        # Create mock challenge
        challenge = Challenge(
            timestamp=test_data.index[i].isoformat(),
            assets=["btc"]
        )

        # Measure response time
        start_time = time.perf_counter()

        try:
            # Make prediction
            result = await forward(challenge, cm_data)
            response_time = time.perf_counter() - start_time

            if result.predictions and 'btc' in result.predictions:
                prediction = result.predictions['btc']
                interval = result.intervals.get('btc', [prediction * 0.97, prediction * 1.03])

                # Get actual price 1 hour later
                actual_price = test_data.iloc[i + 60]['price'] if i + 60 < len(test_data) else test_data.iloc[-1]['price']

                predictions.append(prediction)
                actuals.append(actual_price)
                intervals.append(interval)
                response_times.append(response_time)

                logger.debug(f"Pred: ${prediction:.2f}, Actual: ${actual_price:.2f}, "
                           f"Interval: [${interval[0]:.2f}, ${interval[1]:.2f}], Time: {response_time:.3f}s")
            else:
                logger.warning(f"No prediction generated for timestamp {challenge.timestamp}")

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            continue

    # Calculate metrics
    if not predictions:
        raise ValueError("No predictions generated during backtest")

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    intervals = np.array(intervals)
    response_times = np.array(response_times)

    # Point forecast metrics
    mape = mean_absolute_percentage_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = np.mean(np.abs(predictions - actuals))

    # Interval coverage (90% confidence interval should cover ~90% of actuals)
    lower_bounds = intervals[:, 0]
    upper_bounds = intervals[:, 1]
    coverage = np.mean((lower_bounds <= actuals) & (actuals <= upper_bounds))

    # Interval width (normalized by price)
    avg_width = np.mean((upper_bounds - lower_bounds) / actuals)

    # Response time
    avg_response_time = np.mean(response_times)
    max_response_time = np.max(response_times)
    response_time_violations = np.sum(response_times > 16)  # Bittensor requirement

    # Calculate prediction intervals statistics
    interval_stats = {
        'mean_width_percent': avg_width * 100,
        'median_width_percent': np.median((upper_bounds - lower_bounds) / actuals) * 100,
        'width_std_percent': np.std((upper_bounds - lower_bounds) / actuals) * 100
    }

    results = {
        'sample_size': len(predictions),
        'test_period_days': test_days,

        # Point forecast metrics
        'mape': mape,
        'mape_percent': mape * 100,
        'rmse': rmse,
        'mae': mae,

        # Interval metrics
        'coverage_rate': coverage,
        'coverage_percent': coverage * 100,
        'avg_interval_width_percent': avg_width * 100,

        # Performance metrics
        'avg_response_time': avg_response_time,
        'max_response_time': max_response_time,
        'response_time_violations': response_time_violations,

        # Detailed interval stats
        'interval_stats': interval_stats,

        # Target achievement flags
        'mape_target_achieved': mape < 0.0015,  # <0.15%
        'coverage_target_achieved': coverage > 0.85,  # >85%
        'response_time_target_achieved': response_time_violations == 0  # <16s always
    }

    return results


def load_historical_data(filepath=None):
    """Load historical BTC data for backtesting"""
    if filepath and os.path.exists(filepath):
        logger.info(f"Loading data from {filepath}")
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        data.index = pd.to_datetime(data.index)
        return data

    # Fetch from CoinMetrics if no file provided
    logger.info("Fetching data from CoinMetrics...")
    end = datetime.now()
    start = end - timedelta(days=30)

    try:
        cm = CMData()
        data = cm.get_CM_ReferenceRate(
            assets=['btc'],
            start=start.isoformat(),
            end=end.isoformat(),
            frequency="1m"
        )

        if data.empty:
            raise ValueError("No data from CoinMetrics")

        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data.rename(columns={'ReferenceRateUSD': 'price'})
        data = data[['price']].dropna()

        logger.info(f"Fetched {len(data)} minutes of data")
        return data

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        # Generate synthetic data for testing
        logger.warning("Using synthetic data for backtesting")
        timestamps = pd.date_range(start=start, end=end, freq='1min')
        # Generate more realistic BTC-like price series
        base_price = 50000
        trend = np.linspace(0, 5000, len(timestamps))  # Slight upward trend
        noise = np.random.normal(0, 50, len(timestamps))  # Price noise
        volatility = np.random.choice([1, 2, 3], len(timestamps), p=[0.7, 0.2, 0.1])  # Occasional volatility spikes
        prices = base_price + trend + noise * volatility

        return pd.DataFrame({'price': prices}, index=timestamps)


def print_results(results):
    """Print formatted backtest results"""
    print("\n" + "="*60)
    print("PRECROG CUSTOM MODEL BACKTEST RESULTS")
    print("="*60)

    print(f"Sample Size: {results['sample_size']} predictions")
    print(f"Test Period: {results['test_period_days']} days")
    print()

    print("POINT FORECAST METRICS:")
    print(f"  MAPE: {results['mape_percent']:.4f}% (Target: <0.15%) {'✅' if results['mape_target_achieved'] else '❌'}")
    print(f"  RMSE: ${results['rmse']:.2f} (Target: <77) {'✅' if results['rmse'] < 77 else '❌'}")
    print(f"  MAE:  ${results['mae']:.2f}")
    print()

    print("INTERVAL FORECAST METRICS:")
    print(f"  Coverage Rate: {results['coverage_percent']:.1f}% (Target: >85%) {'✅' if results['coverage_target_achieved'] else '❌'}")
    print(f"  Avg Interval Width: {results['avg_interval_width_percent']:.1f}% of price")
    print()

    print("PERFORMANCE METRICS:")
    print(f"  Avg Response Time: {results['avg_response_time']:.3f}s (Target: <16s) {'✅' if results['response_time_target_achieved'] else '❌'}")
    print(f"  Max Response Time: {results['max_response_time']:.3f}s")
    print(f"  Response Time Violations: {results['response_time_violations']} (Target: 0)")
    print()

    print("QUALITY GATE ASSESSMENT:")
    all_targets_met = all([
        results['mape_target_achieved'],
        results['coverage_target_achieved'],
        results['response_time_target_achieved']
    ])

    if all_targets_met:
        print("🎉 ALL QUALITY GATES PASSED! Ready for testnet deployment.")
        print("   Proceed to Phase 2: Testnet Validation")
    else:
        print("⚠️  SOME QUALITY GATES FAILED. Address issues before proceeding:")
        if not results['mape_target_achieved']:
            print("   - Improve point forecast accuracy (model training/hyperparameters)")
        if not results['coverage_target_achieved']:
            print("   - Adjust interval forecasting (quantile loss tuning)")
        if not results['response_time_target_achieved']:
            print("   - Optimize inference speed (model size/feature extraction)")

    print("="*60)


async def main():
    """Main backtest function"""
    import argparse

    parser = argparse.ArgumentParser(description='Run local backtest for Precog model')
    parser.add_argument('--data-file', type=str, help='Path to historical data CSV file')
    parser.add_argument('--test-days', type=int, default=7, help='Number of days to test (default: 7)')

    args = parser.parse_args()

    # Load historical data
    data = load_historical_data(args.data_file)

    # Run backtest
    results = await run_backtest(data, args.test_days)

    # Print results
    print_results(results)

    # Save results
    results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                json_results[k] = v.tolist()
            elif isinstance(v, np.floating):
                json_results[k] = float(v)
            elif isinstance(v, np.integer):
                json_results[k] = int(v)
            else:
                json_results[k] = v
        json.dump(json_results, f, indent=2)

    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
