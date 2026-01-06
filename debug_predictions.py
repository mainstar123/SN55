#!/usr/bin/env python3
"""Debug model predictions to understand 100% hit rate"""

import os
os.environ['TRAINING_MODE'] = 'true'

import torch
import numpy as np
import pandas as pd
from train_multi_asset_domination import simple_32_feature_extraction
from precog.utils.cm_data import CMData

def debug_predictions():
    print("🔍 DEBUGGING MODEL PREDICTIONS")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from train_multi_asset_domination import WorkingEnsemble
    model = WorkingEnsemble(input_size=32, hidden_size=128)
    model.load_state_dict(torch.load('models/multi_asset_domination_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Load scaler
    import pickle
    with open('models/multi_asset_feature_scaler.pkl', 'rb') as f:
        scaler_data = pickle.load(f)
    feature_means = scaler_data['means']
    feature_stds = scaler_data['stds']

    # Get test data
    cm = CMData()
    data = cm.get_recent_data(minutes=60, asset='btc')

    if data.empty:
        print("❌ No test data")
        return

    print(f"📊 Testing on {len(data)} data points")

    # Test first 5 predictions manually
    print("\n🔬 MANUAL PREDICTION ANALYSIS:")
    print("-" * 50)

    for i in range(min(5, len(data) - 60)):
        idx = 60 + i  # Start after window size

        # Get price window
        price_window = data['price'].values[idx-60:idx]
        current_price = data['price'].values[idx]
        next_price = data['price'].values[idx+1] if idx+1 < len(data) else current_price

        # Extract features
        mock_data = pd.DataFrame({'price': price_window, 'volume': np.ones(60)})
        features, _ = simple_32_feature_extraction(mock_data)

        # Scale features
        scaled_features = (features - feature_means) / feature_stds
        scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).unsqueeze(0).to(device)
            prediction_return = model(input_tensor).item()
            predicted_price = current_price * (1 + prediction_return)

        # Calculate accuracy
        error_pct = abs(predicted_price - next_price) / next_price * 100

        print(f"  Current Price: ${current_price:,.2f}")
        print(f"  Next Price: ${next_price:,.2f}")
        print(f"  Predicted Price: ${predicted_price:,.2f}")
        print(f"  Prediction Error: {error_pct:.6f}%")
        print(f"  Within 1% threshold: {'✅ YES' if error_pct <= 1.0 else '❌ NO'}")
        print()

if __name__ == "__main__":
    debug_predictions()
