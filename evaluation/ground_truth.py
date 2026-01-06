"""
Ground Truth Resolver Module

Resolves actual price outcomes 1 hour after each forecast timestamp.
Computes actual price at t+1h and min/max prices during (t, t+1h).
"""

import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
import time


class GroundTruthResolver:
    """Resolves ground truth prices for forecast evaluation."""

    def __init__(self, csv_path: str = "forecast_data.csv", api_key_env: str = "COINMETRICS_API_KEY"):
        """
        Initialize the ground truth resolver.

        Args:
            csv_path: Path to CSV file containing forecast data
            api_key_env: Environment variable name for Coin Metrics API key
        """
        self.csv_path = csv_path
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")

        self.coinmetrics_base_url = "https://api.coinmetrics.io/v4"

    def get_price_at_time(self, timestamp: datetime) -> float:
        """
        Get BTC Reference Rate at a specific timestamp.

        Args:
            timestamp: Target timestamp

        Returns:
            Price at that timestamp
        """
        price, _ = self._fetch_reference_rate(timestamp, timestamp)
        return price

    def get_price_range(self, start_time: datetime, end_time: datetime) -> Tuple[float, float, float]:
        """
        Get price statistics for a time range.

        Args:
            start_time: Start of interval (exclusive)
            end_time: End of interval (inclusive)

        Returns:
            Tuple of (price_at_end, min_price, max_price)
        """
        prices = self._fetch_price_series(start_time, end_time)

        if not prices:
            raise ValueError(f"No price data available for range {start_time} to {end_time}")

        # Price at t+1h is the last price in the series
        price_at_end = prices[-1]

        # Min and max during the interval (excluding start, including end)
        min_price = min(prices)
        max_price = max(prices)

        return price_at_end, min_price, max_price

    def _fetch_reference_rate(self, start_time: datetime, end_time: datetime,
                            frequency: str = "1s") -> Tuple[float, datetime]:
        """
        Fetch reference rate for a specific time point.

        Args:
            start_time: Start time
            end_time: End time
            frequency: Data frequency

        Returns:
            Tuple of (price, timestamp)
        """
        params = {
            'assets': 'btc',
            'metrics': 'ReferenceRateUSD',
            'start_time': start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'end_time': end_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'frequency': frequency,
            'page_size': 1
        }

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        }

        try:
            response = requests.get(
                f"{self.coinmetrics_base_url}/timeseries/asset-metrics",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            if not data.get('data'):
                raise ValueError("No data returned from Coin Metrics API")

            record = data['data'][0]
            price = float(record['ReferenceRateUSD'])
            timestamp = datetime.fromisoformat(record['time'].replace('Z', '+00:00'))

            return price, timestamp

        except Exception as e:
            raise Exception(f"Failed to fetch reference rate: {str(e)}")

    def _fetch_price_series(self, start_time: datetime, end_time: datetime) -> List[float]:
        """
        Fetch series of prices for a time range.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            List of prices in chronological order
        """
        params = {
            'assets': 'btc',
            'metrics': 'ReferenceRateUSD',
            'start_time': start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'end_time': end_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'frequency': '1m',  # 1-minute intervals
            'page_size': 1000   # Should cover 1 hour
        }

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        }

        try:
            response = requests.get(
                f"{self.coinmetrics_base_url}/timeseries/asset-metrics",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            prices = []
            for record in data.get('data', []):
                if record.get('ReferenceRateUSD') is not None:
                    prices.append(float(record['ReferenceRateUSD']))

            return prices

        except Exception as e:
            raise Exception(f"Failed to fetch price series: {str(e)}")

    def resolve_ground_truth(self, forecast_timestamp: datetime) -> Dict[str, float]:
        """
        Resolve ground truth for a forecast timestamp.

        Args:
            forecast_timestamp: The timestamp when forecast was made

        Returns:
            Dict with actual_price_1h, actual_low, actual_high
        """
        # Calculate the target time (t+1h)
        target_time = forecast_timestamp + timedelta(hours=1)

        # Get price at t+1h and min/max during (t, t+1h)
        # Note: We exclude the start time and include the end time
        actual_price_1h, actual_low, actual_high = self.get_price_range(
            forecast_timestamp, target_time
        )

        return {
            'actual_price_1h': actual_price_1h,
            'actual_low': actual_low,
            'actual_high': actual_high,
            'resolved_at': datetime.now(timezone.utc).isoformat()
        }

    def update_csv_with_ground_truth(self):
        """Update CSV file with resolved ground truth for all unresolved entries."""
        try:
            df = pd.read_csv(self.csv_path)

            # Find rows that haven't been resolved yet
            unresolved_mask = df['actual_price_1h'].isna()

            if not unresolved_mask.any():
                print(f"[{datetime.now()}] No unresolved forecasts to process")
                return

            unresolved_rows = df[unresolved_mask].copy()

            for idx, row in unresolved_rows.iterrows():
                try:
                    forecast_time = datetime.fromisoformat(row['timestamp'])

                    # Check if enough time has passed (at least 1 hour + some buffer)
                    time_diff = datetime.now(timezone.utc) - forecast_time
                    if time_diff < timedelta(hours=1, minutes=5):  # 1h + 5min buffer
                        continue

                    # Resolve ground truth
                    ground_truth = self.resolve_ground_truth(forecast_time)

                    # Update the row
                    df.loc[idx, 'actual_price_1h'] = ground_truth['actual_price_1h']
                    df.loc[idx, 'actual_low'] = ground_truth['actual_low']
                    df.loc[idx, 'actual_high'] = ground_truth['actual_high']
                    df.loc[idx, 'resolved_at'] = ground_truth['resolved_at']

                    print(f"[{datetime.now()}] Resolved ground truth for {forecast_time}")

                except Exception as e:
                    print(f"[{datetime.now()}] Error resolving ground truth for row {idx}: {str(e)}")
                    continue

            # Save updated CSV
            df.to_csv(self.csv_path, index=False)
            print(f"[{datetime.now()}] Updated CSV with ground truth data")

        except Exception as e:
            print(f"[{datetime.now()}] Error updating CSV with ground truth: {str(e)}")
            raise
