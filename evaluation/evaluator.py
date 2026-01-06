"""
Evaluator Module

Computes validator-like metrics for forecast evaluation:
- Point error (MAE)
- Interval hit rate
- Interval width
- Combined score

Provides rolling evaluation over different time windows.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    mae: float
    interval_hit_rate: float
    avg_interval_width: float
    avg_combined_score: float
    sample_count: int


@dataclass
class ComparisonResult:
    """Container for model comparison results."""
    my_model: EvaluationMetrics
    top_miner: EvaluationMetrics
    improvement_pct: Dict[str, float]
    verdict: str  # "LIKELY BETTER", "UNCLEAR", "WORSE"


class Evaluator:
    """Evaluates forecast performance using validator-like metrics."""

    def __init__(self, csv_path: str = "forecast_data.csv", alpha: float = 10.0, beta: float = 0.1):
        """
        Initialize the evaluator.

        Args:
            csv_path: Path to CSV file containing forecast and ground truth data
            alpha: Weight for interval miss penalty in combined score
            beta: Weight for interval width in combined score
        """
        self.csv_path = csv_path
        self.alpha = alpha
        self.beta = beta

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the forecast data."""
        df = pd.read_csv(self.csv_path)

        # Convert timestamp columns to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        if 'resolved_at' in df.columns and df['resolved_at'].notna().any():
            df['resolved_at'] = pd.to_datetime(df['resolved_at'], utc=True)

        # Filter to only resolved forecasts
        resolved_df = df.dropna(subset=['actual_price_1h', 'actual_low', 'actual_high']).copy()

        return resolved_df

    def calculate_point_error(self, predicted: float, actual: float) -> float:
        """Calculate absolute point error (MAE component)."""
        return abs(predicted - actual)

    def calculate_interval_hit(self, predicted_low: float, predicted_high: float,
                             actual_low: float, actual_high: float) -> int:
        """
        Calculate whether the prediction interval contains the actual range.

        Returns 1 if interval covers actual range, 0 otherwise.
        """
        return 1 if (actual_low >= predicted_low) and (actual_high <= predicted_high) else 0

    def calculate_interval_width(self, low: float, high: float) -> float:
        """Calculate prediction interval width."""
        return high - low

    def calculate_interval_miss_penalty(self, predicted_low: float, predicted_high: float,
                                      actual_low: float, actual_high: float) -> float:
        """
        Calculate penalty for missing parts of the actual range.

        Penalty is the total amount by which the interval misses the actual range.
        """
        miss_below = max(0, predicted_low - actual_low)
        miss_above = max(0, actual_high - predicted_high)
        return miss_below + miss_above

    def calculate_combined_score(self, point_error: float, interval_hit: int,
                               interval_miss_penalty: float, interval_width: float) -> float:
        """
        Calculate combined forecast score.

        Lower scores are better. Combines point error, interval coverage, and width.
        """
        return (point_error +
                self.alpha * (1 - interval_hit) * interval_miss_penalty +
                self.beta * interval_width)

    def evaluate_model(self, df: pd.DataFrame, model_prefix: str) -> EvaluationMetrics:
        """
        Evaluate a single model's performance.

        Args:
            df: DataFrame with forecast and ground truth data
            model_prefix: 'my' or 'top' to specify which model to evaluate

        Returns:
            EvaluationMetrics for the model
        """
        # Extract model predictions
        point_col = f'{model_prefix}_point'
        low_col = f'{model_prefix}_low'
        high_col = f'{model_prefix}_high'

        # Calculate metrics for each forecast
        point_errors = []
        interval_hits = []
        interval_widths = []
        combined_scores = []

        for _, row in df.iterrows():
            # Point error
            point_error = self.calculate_point_error(row[point_col], row['actual_price_1h'])
            point_errors.append(point_error)

            # Interval metrics
            interval_hit = self.calculate_interval_hit(
                row[low_col], row[high_col], row['actual_low'], row['actual_high']
            )
            interval_hits.append(interval_hit)

            interval_width = self.calculate_interval_width(row[low_col], row[high_col])
            interval_widths.append(interval_width)

            # Miss penalty
            miss_penalty = self.calculate_interval_miss_penalty(
                row[low_col], row[high_col], row['actual_low'], row['actual_high']
            )

            # Combined score
            combined_score = self.calculate_combined_score(
                point_error, interval_hit, miss_penalty, interval_width
            )
            combined_scores.append(combined_score)

        # Aggregate metrics
        mae = np.mean(point_errors)
        interval_hit_rate = np.mean(interval_hits)
        avg_interval_width = np.mean(interval_widths)
        avg_combined_score = np.mean(combined_scores)

        return EvaluationMetrics(
            mae=mae,
            interval_hit_rate=interval_hit_rate,
            avg_interval_width=avg_interval_width,
            avg_combined_score=avg_combined_score,
            sample_count=len(df)
        )

    def compare_models(self, df: pd.DataFrame) -> ComparisonResult:
        """
        Compare my model vs top miner performance.

        Args:
            df: DataFrame with forecast and ground truth data

        Returns:
            ComparisonResult with metrics and verdict
        """
        my_metrics = self.evaluate_model(df, 'my')
        top_metrics = self.evaluate_model(df, 'top')

        # Calculate percentage improvements (positive = my model better)
        improvement_pct = {
            'mae': ((top_metrics.mae - my_metrics.mae) / top_metrics.mae) * 100,
            'interval_hit_rate': ((my_metrics.interval_hit_rate - top_metrics.interval_hit_rate) /
                                top_metrics.interval_hit_rate) * 100,
            'avg_combined_score': ((top_metrics.avg_combined_score - my_metrics.avg_combined_score) /
                                 top_metrics.avg_combined_score) * 100
        }

        # Determine verdict based on combined score improvement and sample size
        verdict = self._determine_verdict(improvement_pct, my_metrics.sample_count)

        return ComparisonResult(
            my_model=my_metrics,
            top_miner=top_metrics,
            improvement_pct=improvement_pct,
            verdict=verdict
        )

    def _determine_verdict(self, improvement_pct: Dict[str, float], sample_count: int) -> str:
        """
        Determine confidence verdict based on improvements and sample size.

        Args:
            improvement_pct: Dictionary of percentage improvements
            sample_count: Number of samples evaluated

        Returns:
            Verdict string: "LIKELY BETTER", "UNCLEAR", "WORSE"
        """
        combined_improvement = improvement_pct['avg_combined_score']

        # Require minimum sample size for confidence
        if sample_count < 10:
            return "UNCLEAR"

        # Strong improvement threshold
        if combined_improvement > 15:
            return "LIKELY BETTER"

        # Moderate improvement but check consistency
        elif combined_improvement > 5:
            # Check if point accuracy and interval coverage are both improved
            if (improvement_pct['mae'] > 0 and improvement_pct['interval_hit_rate'] > 0):
                return "LIKELY BETTER"
            else:
                return "UNCLEAR"

        # Degradation
        elif combined_improvement < -5:
            return "WORSE"

        # Marginal results
        else:
            return "UNCLEAR"

    def evaluate_time_window(self, hours: int) -> Optional[ComparisonResult]:
        """
        Evaluate performance over a rolling time window.

        Args:
            hours: Number of hours to look back

        Returns:
            ComparisonResult if sufficient data available, None otherwise
        """
        try:
            df = self.load_data()

            # Filter to recent data
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_df = df[df['timestamp'] >= cutoff_time].copy()

            if len(recent_df) < 5:  # Require minimum samples
                return None

            return self.compare_models(recent_df)

        except Exception as e:
            print(f"Error evaluating {hours}h window: {str(e)}")
            return None

    def print_evaluation_summary(self):
        """Print evaluation summary for all time windows."""
        windows = [24, 72, 168]  # 24h, 3d, 7d

        print(f"\n{'='*60}")
        print("SHADOW EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Evaluation time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

        for hours in windows:
            result = self.evaluate_time_window(hours)
            if result is None:
                print(f"\n--- Last {hours}h: INSUFFICIENT DATA ---")
                continue

            days = hours / 24
            window_name = f"{days:.0f} day{'s' if days != 1 else ''}"

            print(f"\n--- Last {window_name} ({result.my_model.sample_count} samples) ---")
            print(f"My Model:    MAE={result.my_model.mae:.2f}, Hit={result.my_model.interval_hit_rate:.1%}, Score={result.my_model.avg_combined_score:.2f}")
            print(f"Top Miner:   MAE={result.top_miner.mae:.2f}, Hit={result.top_miner.interval_hit_rate:.1%}, Score={result.top_miner.avg_combined_score:.2f}")
            print(f"Combined Score Improvement: {result.improvement_pct['avg_combined_score']:+.1f}%")
            print(f"Verdict: {result.verdict}")

        print(f"\n{'='*60}")
