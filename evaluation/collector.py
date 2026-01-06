"""
Data Collector Module

Handles periodic collection of:
- Current BTC Reference Rate from Coin Metrics API v4
- Top miner forecast from Precog API
- Local model predictions
- Stores data to CSV for later evaluation
"""

import os
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import time


class DataCollector:
    """Collects forecast data every 5 minutes for shadow evaluation."""

    def __init__(self, csv_path: str = "forecast_data.csv", api_key_env: str = "COINMETRICS_API_KEY"):
        """
        Initialize the data collector.

        Args:
            csv_path: Path to CSV file for storing collected data
            api_key_env: Environment variable name for Coin Metrics API key
        """
        self.csv_path = csv_path
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")

        self.coinmetrics_base_url = "https://api.coinmetrics.io/v4"
        self.precog_url = "https://precog.coinmetrics.io"

        # Ensure CSV exists with headers
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            columns = [
                'timestamp', 'spot_price',
                'top_point', 'top_low', 'top_high',
                'my_point', 'my_low', 'my_high',
                'actual_price_1h', 'actual_low', 'actual_high',
                'resolved_at'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_path, index=False)

    def get_btc_reference_rate(self, timestamp: Optional[datetime] = None) -> Tuple[float, datetime]:
        """
        Fetch BTC Reference Rate from Coin Metrics API v4.

        Args:
            timestamp: Specific timestamp to fetch (default: current time)

        Returns:
            Tuple of (price, timestamp)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Format timestamp for API
        time_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        params = {
            'assets': 'btc',
            'metrics': 'ReferenceRateUSD',
            'start_time': time_str,
            'end_time': time_str,
            'frequency': '1s',  # Get the exact timestamp
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

            # Extract price and timestamp
            record = data['data'][0]
            price = float(record['ReferenceRateUSD'])
            api_timestamp = datetime.fromisoformat(record['time'].replace('Z', '+00:00'))

            return price, api_timestamp

        except Exception as e:
            raise Exception(f"Failed to fetch BTC reference rate: {str(e)}")

    def get_top_miner_forecast(self) -> Dict[str, float]:
        """
        Fetch current forecast from top miner via Precog API.

        Returns:
            Dict with keys: point, low, high
        """
        try:
            response = requests.get(self.precog_url)
            response.raise_for_status()
            data = response.json()

            # Assuming the API returns the forecast in this format
            # Adjust based on actual API response structure
            forecast = {
                'point': float(data['point']),
                'low': float(data['low']),
                'high': float(data['high'])
            }

            return forecast

        except Exception as e:
            raise Exception(f"Failed to fetch top miner forecast: {str(e)}")

    def collect_data_point(self, my_model_predict) -> Dict:
        """
        Collect one complete data point including all forecasts.

        Args:
            my_model_predict: Function that takes (spot_price, timestamp) and returns
                            {'point': float, 'low': float, 'high': float}

        Returns:
            Dict containing all collected data
        """
        # Get current timestamp and spot price
        spot_price, timestamp = self.get_btc_reference_rate()

        # Get top miner forecast
        top_forecast = self.get_top_miner_forecast()

        # Get my model forecast
        my_forecast = my_model_predict(spot_price, timestamp)

        data_point = {
            'timestamp': timestamp.isoformat(),
            'spot_price': spot_price,
            'top_point': top_forecast['point'],
            'top_low': top_forecast['low'],
            'top_high': top_forecast['high'],
            'my_point': my_forecast['point'],
            'my_low': my_forecast['low'],
            'my_high': my_forecast['high'],
            'actual_price_1h': None,  # To be filled by ground truth resolver
            'actual_low': None,        # To be filled by ground truth resolver
            'actual_high': None,       # To be filled by ground truth resolver
            'resolved_at': None        # To be filled by ground truth resolver
        }

        return data_point

    def append_to_csv(self, data_point: Dict):
        """Append a data point to the CSV file."""
        df = pd.DataFrame([data_point])
        df.to_csv(self.csv_path, mode='a', header=False, index=False)

    def run_collection_cycle(self, my_model_predict) -> Dict:
        """
        Run one complete collection cycle.

        Args:
            my_model_predict: User's forecasting model function

        Returns:
            The collected data point
        """
        try:
            data_point = self.collect_data_point(my_model_predict)
            self.append_to_csv(data_point)
            print(f"[{datetime.now()}] Collected data point at {data_point['timestamp']}")
            return data_point
        except Exception as e:
            print(f"[{datetime.now()}] Error collecting data: {str(e)}")
            raise
