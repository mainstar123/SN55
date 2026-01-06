#!/usr/bin/env python3
"""Test data fetching for multi-asset training"""

from precog.utils.cm_data import CMData
import pandas as pd

def test_data_fetch():
    cm = CMData()
    assets = ['btc', 'eth', 'tao_bittensor']

    for asset in assets:
        print(f"\n🧪 Testing {asset.upper()} data fetch:")
        try:
            # Test with shorter time range first
            data = cm.get_recent_data(minutes=60, asset=asset)  # 1 hour instead of 24
            print(f"  ✅ Fetched {len(data)} rows")
            if len(data) > 0:
                print(".2f")
                print(f"  📅 Time range: {data['time'].min()} to {data['time'].max()}")
                print(f"  🏷️  Asset column: {data['asset'].unique() if 'asset' in data.columns else 'No asset column'}")
            else:
                print("  ❌ Empty dataframe")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_data_fetch()
