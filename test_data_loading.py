#!/usr/bin/env python3
"""Quick test of multi-asset data loading"""

import os
os.environ['TRAINING_MODE'] = 'true'

from train_multi_asset_domination import MultiAssetDominationDataset

print("🧪 Testing multi-asset data loading...")

try:
    dataset = MultiAssetDominationDataset(
        assets=['btc', 'eth', 'tao_bittensor'],
        samples_per_asset=100  # Just 100 samples for testing
    )

    print(f"✅ Dataset created with {len(dataset)} samples")
    print(f"📏 Feature means shape: {dataset.feature_means.shape}")
    print(f"📏 Feature stds shape: {dataset.feature_stds.shape}")

    # Test a few samples
    for i in range(min(3, len(dataset))):
        features, target = dataset[i]
        print(f"Sample {i}: features shape {features.shape}, target {target:.6f}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
