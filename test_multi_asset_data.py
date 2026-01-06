#!/usr/bin/env python3
"""
Test multi-asset data fetching for subnet 55
"""

from precog.utils.cm_data import CMData

def test_multi_asset_data():
    """Test fetching data for BTC, ETH, and TAO"""

    assets = ['btc', 'eth', 'tao_bittensor']
    cm = CMData()

    print("🔄 TESTING MULTI-ASSET DATA FETCHING:\n")

    for asset in assets:
        print(f"🧪 Testing {asset.upper()}:")
        try:
            # First test the raw API call
            from datetime import datetime, timedelta
            import pytz
            end_time = datetime.now(pytz.UTC)
            start_time = end_time - timedelta(minutes=10)

            asset_mapping = {
                "btc": "btc",
                "eth": "eth",
                "tao_bittensor": "tao"
            }
            cm_asset = asset_mapping.get(asset.lower(), asset.lower())

            print(f"  🔍 Fetching raw data for {cm_asset}...")
            raw_df = cm.get_CM_ReferenceRate(
                assets=[cm_asset],
                start=start_time,
                end=end_time
            )
            print(f"  📊 Raw data shape: {raw_df.shape}")
            if not raw_df.empty:
                print(f"  🏷️  Raw asset column unique values: {raw_df['asset'].unique() if 'asset' in raw_df.columns else 'No asset column'}")
                print(f"  💰 Raw price range: ${raw_df['ReferenceRateUSD'].min():.2f} - ${raw_df['ReferenceRateUSD'].max():.2f}")

            # Now test our processed method
            data = cm.get_recent_data(minutes=10, asset=asset)
            if not data.empty:
                print(f"  ✅ SUCCESS: {len(data)} data points")
                print(".2f")
                print(f"  📅 Time range: {data['time'].min()} to {data['time'].max()}")
                print(".2f")
                unique_assets = data['asset'].unique() if 'asset' in data.columns else ['No asset column']
                print(f"  🏷️  Processed asset values: {unique_assets}")
            else:
                print("  ❌ FAILED: No processed data returned")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
        print()

if __name__ == "__main__":
    test_multi_asset_data()
