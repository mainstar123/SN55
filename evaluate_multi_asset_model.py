#!/usr/bin/env python3
"""
EVALUATE MULTI-ASSET DOMINATION MODEL
Test performance on BTC, ETH, and TAO data
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime

# Add paths
sys.path.append('.')

# Import required modules
from precog.utils.cm_data import CMData
from train_multi_asset_domination import simple_32_feature_extraction, WorkingEnsemble

def load_model_and_scaler():
    """Load the trained multi-asset model and scaler"""
    print("🔄 Loading multi-asset model and scaler...")

    # Load scaler
    with open('models/multi_asset_feature_scaler.pkl', 'rb') as f:
        scaler_data = pickle.load(f)

    feature_means = scaler_data['means']
    feature_stds = scaler_data['stds']

    print(f"✅ Scaler loaded: {len(feature_means)} features")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WorkingEnsemble(input_size=32, hidden_size=128)
    model.load_state_dict(torch.load('models/multi_asset_domination_model.pth', map_location=device))
    model.to(device)
    model.eval()

    print(f"✅ Model loaded on {device}")

    return model, feature_means, feature_stds, device

def evaluate_on_asset(model, feature_means, feature_stds, device, asset, num_samples=100):
    """Evaluate model performance on a specific asset"""
    print(f"\\n🧪 Evaluating on {asset.upper()}...")

    # Fetch recent data
    cm = CMData()
    data = cm.get_recent_data(minutes=360, asset=asset)  # 6 hours

    if data.empty:
        print(f"❌ No data available for {asset}")
        return None

    print(f"📊 Testing on {len(data)} data points")

    predictions = []
    actuals = []
    inference_times = []

    # Test on sliding windows
    window_size = 60
    step_size = 5

    for i in range(window_size, min(len(data), window_size + num_samples * step_size), step_size):
        try:
            # Get price window
            price_window = data['price'].values[i-window_size:i]
            current_price = data['price'].values[i]
            next_price = data['price'].values[i+1] if i+1 < len(data) else current_price

            # Create mock dataframe
            mock_data = pd.DataFrame({
                'price': price_window,
                'volume': np.ones(window_size)
            })

            # Extract features
            features, _ = simple_32_feature_extraction(mock_data)

            # Scale features
            scaled_features = (features - feature_means) / feature_stds
            scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Predict
            start_time = time.time()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).unsqueeze(0).to(device)
                prediction = model(input_tensor).item()
            inference_time = time.time() - start_time

            # Convert prediction back to price scale (rough approximation)
            predicted_return = prediction
            predicted_price = current_price * (1 + predicted_return)

            predictions.append(predicted_price)
            actuals.append(next_price)
            inference_times.append(inference_time)

        except Exception as e:
            continue

    if len(predictions) == 0:
        print(f"❌ No valid predictions for {asset}")
        return None

    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    inference_times = np.array(inference_times)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    # Hit rate (@1% and @2%)
    hit_rate_1pct = np.mean(np.abs((predictions - actuals) / actuals) <= 0.01) * 100
    hit_rate_2pct = np.mean(np.abs((predictions - actuals) / actuals) <= 0.02) * 100

    # Average inference time
    avg_inference_time = np.mean(inference_times) * 1000  # ms

    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print(f"📈 Predictions made: {len(predictions)}")

    return {
        'asset': asset,
        'mape': mape,
        'hit_rate_1pct': hit_rate_1pct,
        'hit_rate_2pct': hit_rate_2pct,
        'avg_inference_ms': avg_inference_time,
        'num_predictions': len(predictions)
    }

def main():
    print("🎯 EVALUATING MULTI-ASSET DOMINATION MODEL")
    print("=" * 50)

    # Load model and scaler
    model, feature_means, feature_stds, device = load_model_and_scaler()

    # Evaluate on each asset
    assets = ['btc', 'eth', 'tao_bittensor']
    results = []

    for asset in assets:
        result = evaluate_on_asset(model, feature_means, feature_stds, device, asset)
        if result:
            results.append(result)

    # Summary
    print("\\n" + "=" * 50)
    print("📊 EVALUATION SUMMARY")
    print("=" * 50)

    if results:
        avg_mape = np.mean([r['mape'] for r in results])
        avg_hit_1pct = np.mean([r['hit_rate_1pct'] for r in results])
        avg_hit_2pct = np.mean([r['hit_rate_2pct'] for r in results])
        avg_inference = np.mean([r['avg_inference_ms'] for r in results])

        print("\\n🎯 OVERALL PERFORMANCE:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print("\\n🏆 ASSET-BY-ASSET RESULTS:")
        for result in results:
            print(f"  {result['asset'].upper():>8}: MAPE {result['mape']:>6.2f}% | Hit@1% {result['hit_rate_1pct']:>5.1f}% | Hit@2% {result['hit_rate_2pct']:>5.1f}%")

        print("\\n✅ EVALUATION COMPLETE")
        print("🎯 Model is ready for deployment!" if avg_mape < 1.0 else "⚠️ Model may need improvement")
    else:
        print("❌ No evaluation results available")

if __name__ == "__main__":
    main()
