#!/usr/bin/env python3
"""
Fetch training data from CoinMetrics for model training
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precog.utils.cm_data import CMData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Fetch and save training data"""
    import argparse

    parser = argparse.ArgumentParser(description='Fetch training data from CoinMetrics')
    parser.add_argument('--days', type=int, default=30, help='Number of days of data to fetch')
    parser.add_argument('--output', type=str, default='data/btc_1m_train.csv', help='Output file path')

    args = parser.parse_args()

    # Create data directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Fetch data
    end = datetime.now()
    start = end - timedelta(days=args.days)

    logger.info(f"Fetching BTC data from {start.date()} to {end.date()}")

    try:
        cm = CMData()
        data = cm.get_CM_ReferenceRate(
            assets=['btc'],
            start=start.isoformat(),
            end=end.isoformat(),
            frequency="1m"
        )

        if data.empty:
            raise ValueError("No data received from CoinMetrics")

        # Process data
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data.rename(columns={'ReferenceRateUSD': 'price'})
        data = data[['price']].dropna()

        # Save data
        data.to_csv(args.output)
        logger.info(f"Saved {len(data)} minutes of data to {args.output}")

        # Print summary
        print(f"Data Summary:")
        print(f"  Start: {data.index[0]}")
        print(f"  End: {data.index[-1]}")
        print(f"  Samples: {len(data)}")
        print(f"  Price range: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
        print(f"  Avg price: ${data['price'].mean():.2f}")

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
