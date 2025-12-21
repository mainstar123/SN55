#!/usr/bin/env python3
"""
Test Domination Features
Verify all domination components are working before full deployment
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
from datetime import datetime, timezone

def test_domination_components():
    """Test all domination components"""

    print("🧪 TESTING DOMINATION COMPONENTS")
    print("=" * 50)

    # Test 1: Model Loading
    print("\n1️⃣ Testing Model Loading...")
    try:
        sys.path.append('precog/miners')
        from standalone_domination import load_domination_models, WorkingEnsemble
        load_domination_models()
        print("✅ Model loading successful")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

    # Test 2: Peak Hour Detection
    print("\n2️⃣ Testing Peak Hour Detection...")
    from standalone_domination import is_peak_hour, get_current_hour_utc

    current_hour = get_current_hour_utc()
    is_peak = is_peak_hour()
    print(f"   Current UTC hour: {current_hour}")
    print(f"   Is peak hour: {is_peak}")
    print("✅ Peak hour detection working")

    # Test 3: Market Regime Detection
    print("\n3️⃣ Testing Market Regime Detection...")
    from standalone_domination import detect_market_regime

    # Create sample price data (simulate 1 hour of data)
    sample_prices = np.random.normal(88000, 500, 60).tolist()
    regime = detect_market_regime(sample_prices)
    print(f"   Detected regime: {regime.upper()}")
    print("✅ Market regime detection working")

    # Test 4: Adaptive Parameters
    print("\n4️⃣ Testing Adaptive Parameters...")
    from standalone_domination import get_adaptive_parameters

    params = get_adaptive_parameters(regime, is_peak)
    print(f"   Prediction frequency: {params['freq']} minutes")
    print(f"   Confidence threshold: {params['threshold']}")
    print(f"   Strategy: {params['description']}")
    print("✅ Adaptive parameters working")

    # Test 5: Prediction Confidence
    print("\n5️⃣ Testing Prediction Confidence...")
    from standalone_domination import should_make_prediction

    confidence_scores = [0.6, 0.75, 0.85, 0.95]
    for conf in confidence_scores:
        should_predict = should_make_prediction(conf, regime, is_peak)
        status = "✅ PREDICT" if should_predict else "⏸️ SKIP"
        print(f"   Confidence {conf:.2f}: {status}")

    print("✅ Prediction confidence working")

    # Test 6: Ensemble Model Inference
    print("\n6️⃣ Testing Ensemble Model Inference...")
    try:
        # Create sample input (24 features, 60 timesteps)
        sample_input = torch.randn(1, 60, 24)

        # Load model and run inference
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = WorkingEnsemble(input_size=24, hidden_size=64)
        model.load_state_dict(torch.load('models/working_domination_ensemble.pth', map_location=device))
        model.to(device)
        model.eval()

        with torch.no_grad():
            sample_input = sample_input.to(device)
            output = model(sample_input)
            prediction = output.item()

        print(f"   Prediction: {prediction:.2f}")
        print("✅ Ensemble model inference working")

    except Exception as e:
        print(f"❌ Ensemble model inference failed: {e}")
        return False

    # Test 7: Performance Tracking
    print("\n7️⃣ Testing Performance Tracking...")
    from standalone_domination import track_prediction

    # Reset global variables for testing
    import precog.miners.standalone_domination as dom
    dom.prediction_count = 0
    dom.total_reward = 0.0
    dom.response_times = []

    # Track some sample predictions
    track_prediction(88000, 87900, 0.85, 0.15, 0.05)
    track_prediction(88100, 88200, 0.90, 0.12, 0.08)
    track_prediction(87900, 87800, 0.75, 0.18, 0.03)

    avg_reward = dom.total_reward / dom.prediction_count if dom.prediction_count > 0 else 0
    avg_response = sum(dom.response_times) / len(dom.response_times) if dom.response_times else 0

    print("   Sample predictions tracked")
    print(f"   Average reward: {avg_reward:.6f} TAO")
    print(f"   Average response: {avg_response:.3f}s")
    print("✅ Performance tracking working")

    # Summary
    print("\n🎉 DOMINATION COMPONENTS TEST COMPLETE")
    print("=" * 50)
    print("✅ All domination features are working!")
    print()
    print("🚀 READY FOR DEPLOYMENT:")
    print("• Peak hour optimization: ACTIVE")
    print("• Market regime detection: ACTIVE")
    print("• Ensemble predictions: ACTIVE")
    print("• Adaptive thresholds: ACTIVE")
    print("• Performance tracking: ACTIVE")
    print()
    print("🎯 START DOMINATION MINER:")
    print("./start_domination_miner.sh")
    print()
    print("🏆 TARGET: Surpass UID 31 in 48 hours!")

    return True

if __name__ == "__main__":
    success = test_domination_components()
    if not success:
        print("\n❌ Some tests failed. Please fix issues before deployment.")
        sys.exit(1)
    else:
        print("\n✅ All domination components ready for #1 position!")
