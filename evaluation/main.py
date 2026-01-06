"""
Main Orchestration Module

Coordinates the shadow evaluation system:
- Periodic data collection
- Ground truth resolution
- Rolling evaluation
- Summary reporting
"""

import os
import time
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict
import argparse

from collector import DataCollector
from ground_truth import GroundTruthResolver
from evaluator import Evaluator


class ShadowEvaluator:
    """Main orchestrator for the shadow evaluation system."""

    def __init__(self,
                 my_model_predict: Callable,
                 collection_interval_minutes: int = 5,
                 evaluation_interval_hours: int = 24,
                 csv_path: str = "forecast_data.csv"):
        """
        Initialize the shadow evaluator.

        Args:
            my_model_predict: User's forecasting model function
            collection_interval_minutes: How often to collect data (minutes)
            evaluation_interval_hours: How often to run evaluation (hours)
            csv_path: Path to CSV data file
        """
        self.my_model_predict = my_model_predict
        self.collection_interval = collection_interval_minutes * 60  # Convert to seconds
        self.evaluation_interval = evaluation_interval_hours * 3600  # Convert to seconds
        self.csv_path = csv_path

        # Initialize components
        self.collector = DataCollector(csv_path)
        self.ground_truth = GroundTruthResolver(csv_path)
        self.evaluator = Evaluator(csv_path)

        # Track timing
        self.last_evaluation = datetime.now(timezone.utc)
        self.running = False

    def run_collection_cycle(self):
        """Run one data collection cycle."""
        try:
            self.collector.run_collection_cycle(self.my_model_predict)
        except Exception as e:
            print(f"[{datetime.now()}] Collection cycle failed: {str(e)}")

    def run_ground_truth_update(self):
        """Update ground truth for resolved forecasts."""
        try:
            self.ground_truth.update_csv_with_ground_truth()
        except Exception as e:
            print(f"[{datetime.now()}] Ground truth update failed: {str(e)}")

    def run_evaluation(self):
        """Run evaluation and print summary."""
        try:
            self.evaluator.print_evaluation_summary()
            self.last_evaluation = datetime.now(timezone.utc)
        except Exception as e:
            print(f"[{datetime.now()}] Evaluation failed: {str(e)}")

    def should_run_evaluation(self) -> bool:
        """Check if it's time to run evaluation."""
        time_since_last_eval = datetime.now(timezone.utc) - self.last_evaluation
        return time_since_last_eval >= timedelta(hours=self.evaluation_interval / 3600)

    def run_continuous(self):
        """Run the evaluation system continuously."""
        print(f"[{datetime.now()}] Starting shadow evaluation system...")
        print("Press Ctrl+C to stop")

        self.running = True

        # Run initial evaluation
        self.run_evaluation()

        try:
            while self.running:
                cycle_start = time.time()

                # Run collection cycle
                self.run_collection_cycle()

                # Update ground truth
                self.run_ground_truth_update()

                # Run evaluation if due
                if self.should_run_evaluation():
                    self.run_evaluation()

                # Sleep until next collection cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.collection_interval - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n[{datetime.now()}] Stopping shadow evaluation system...")
            self.running = False

    def run_once(self):
        """Run a single collection and evaluation cycle."""
        print(f"[{datetime.now()}] Running single evaluation cycle...")

        self.run_collection_cycle()
        self.run_ground_truth_update()
        self.run_evaluation()


def example_model_predict(spot_price: float, timestamp: datetime) -> Dict[str, float]:
    """
    Example forecasting model.

    Replace this with your actual model implementation.
    This is just a simple baseline that predicts no change with wide intervals.
    """
    # Simple baseline: predict current price with ±2% interval
    margin = spot_price * 0.02

    return {
        'point': spot_price,
        'low': spot_price - margin,
        'high': spot_price + margin
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Shadow Evaluation Harness for Subnet 55')
    parser.add_argument('--mode', choices=['continuous', 'once'], default='continuous',
                       help='Run mode: continuous collection or single run')
    parser.add_argument('--csv', default='forecast_data.csv',
                       help='Path to CSV data file')
    parser.add_argument('--collection-interval', type=int, default=5,
                       help='Collection interval in minutes (default: 5)')
    parser.add_argument('--evaluation-interval', type=int, default=24,
                       help='Evaluation interval in hours (default: 24)')

    args = parser.parse_args()

    # Check for API key
    if not os.getenv('COINMETRICS_API_KEY'):
        print("ERROR: COINMETRICS_API_KEY environment variable not set")
        print("Please set it with: export COINMETRICS_API_KEY='your_key_here'")
        sys.exit(1)

    # Initialize evaluator with example model
    # TODO: Replace example_model_predict with your actual model
    evaluator = ShadowEvaluator(
        my_model_predict=example_model_predict,
        collection_interval_minutes=args.collection_interval,
        evaluation_interval_hours=args.evaluation_interval,
        csv_path=args.csv
    )

    if args.mode == 'continuous':
        evaluator.run_continuous()
    else:
        evaluator.run_once()


if __name__ == "__main__":
    main()
