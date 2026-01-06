#!/usr/bin/env python3
"""Test model generalization across different time periods and market conditions"""

import os
os.environ['TRAINING_MODE'] = 'true'

import torch
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from train_multi_asset_domination import simple_32_feature_extraction, WorkingEnsemble

def test_temporal_generalization():
    """Test model performance across different time periods"""
    print("🧪 TESTING TEMPORAL GENERALIZATION")
    print("=" * 50)

    # Load model and scaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WorkingEnsemble(input_size=32, hidden_size=128)
    model.load_state_dict(torch.load('models/multi_asset_domination_model.pth', map_location=device))
    model.to(device)
    model.eval()

    with open('models/multi_asset_feature_scaler.pkl', 'rb') as f:
        scaler_data = pickle.load(f)
    feature_means = scaler_data['means']
    feature_stds = scaler_data['stds']

    print("✅ Model and scaler loaded")

    # Test on different time windows
    test_periods = [
        ("Recent (last 1 hour)", 60),
        ("Last 6 hours", 360),
        ("Last 24 hours", 1440),
    ]

    results = []

    for period_name, minutes in test_periods:
        print(f"\n📅 Testing: {period_name}")

        try:
            from precog.utils.cm_data import CMData
            cm = CMData()
            data = cm.get_recent_data(minutes=minutes, asset='btc')

            if data.empty or len(data) < 70:
                print(f"  ❌ Insufficient data for {period_name}")
                continue

            print(f"  📊 Data points: {len(data)}")

            # Evaluate on this time period
            predictions = []
            actuals = []

            window_size = 60
            step_size = 10  # Test every 10 minutes

            for i in range(window_size, min(len(data), window_size + 50 * step_size), step_size):
                try:
                    price_window = data['price'].values[i-window_size:i]
                    current_price = data['price'].values[i]
                    next_price = data['price'].values[i+1] if i+1 < len(data) else current_price

                    mock_data = pd.DataFrame({'price': price_window, 'volume': np.ones(window_size)})
                    features, _ = simple_32_feature_extraction(mock_data)

                    scaled_features = (features - feature_means) / feature_stds
                    scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).unsqueeze(0).to(device)
                        prediction_return = model(input_tensor).item()
                        predicted_price = current_price * (1 + prediction_return)

                    predictions.append(predicted_price)
                    actuals.append(next_price)

                except Exception as e:
                    continue

            if len(predictions) < 5:
                print(f"  ❌ Too few predictions for {period_name}")
                continue

            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)

            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            hit_rate_1pct = np.mean(np.abs((predictions - actuals) / actuals) <= 0.01) * 100
            hit_rate_2pct = np.mean(np.abs((predictions - actuals) / actuals) <= 0.02) * 100

            print(".2f")
            print(".2f")
            print(".2f")

            results.append({
                'period': period_name,
                'mape': mape,
                'hit_rate_1pct': hit_rate_1pct,
                'hit_rate_2pct': hit_rate_2pct,
                'predictions': len(predictions)
            })

        except Exception as e:
            print(f"  ❌ Error testing {period_name}: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 GENERALIZATION SUMMARY")
    print("=" * 50)

    if results:
        print("\\n🕐 TEMPORAL GENERALIZATION RESULTS:")
        for result in results:
            print(f"  {result['period']:<20}: {result['hit_rate_1pct']:>5.1f}% @1% | {result['mape']:>5.2f}% MAPE")

        avg_hit_rate = np.mean([r['hit_rate_1pct'] for r in results])
        avg_mape = np.mean([r['mape'] for r in results])

        print("\\n📈 AVERAGE PERFORMANCE:")
        print(".1f")
        print(".2f")
        # Assessment
        if avg_hit_rate >= 90 and avg_mape <= 0.1:
            print("\\n🎯 ASSESSMENT: EXCELLENT GENERALIZATION")
            print("  ✅ Model performs consistently across time periods")
            print("  ✅ Not significantly overfitted")
        elif avg_hit_rate >= 70 and avg_mape <= 0.5:
            print("\\n🎯 ASSESSMENT: GOOD GENERALIZATION")
            print("  ✅ Reasonable performance across time periods")
            print("  ⚠️ Some overfitting possible but manageable")
        else:
            print("\\n🎯 ASSESSMENT: POOR GENERALIZATION")
            print("  ⚠️ Significant overfitting detected")
            print("  ❌ Performance degrades across time periods")
    else:
        print("❌ No generalization test results available")

if __name__ == "__main__":
    test_temporal_generalization()
