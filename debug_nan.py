#!/usr/bin/env python3
"""Debug NaN issue in training"""

import os
os.environ['TRAINING_MODE'] = 'true'

from train_multi_asset_domination import MultiAssetDominationDataset
import numpy as np

print('🔍 DEBUGGING NaN TRAINING ISSUE')

# Create small dataset
dataset = MultiAssetDominationDataset(assets=['btc'], samples_per_asset=5)

print(f'Dataset size: {len(dataset)}')
print(f'Feature means NaN: {np.isnan(dataset.feature_means).any()}')
print(f'Feature stds NaN: {np.isnan(dataset.feature_stds).any()}')
print(f'Feature stds min: {dataset.feature_stds.min()}')

# Check samples
for i in range(min(2, len(dataset))):
    features, target = dataset[i]
    print(f'Sample {i}: features NaN: {np.isnan(features).any()}, target NaN: {np.isnan(target)}')
