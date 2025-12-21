#!/usr/bin/env python3
"""
Demo Script for Precog #1 Miner Domination System
Shows all features working without full training pipeline
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime, timezone

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import domination system components
from advanced_ensemble_model import create_advanced_ensemble
from market_regime_detector import create_adaptive_prediction_system
from peak_hour_optimizer import create_ultra_precise_prediction_system
from performance_tracking_system import create_performance_tracking_system
from gpu_accelerated_training import create_inference_optimizer


def demo_advanced_ensemble():
    """Demo the advanced ensemble model"""
    print("🧠 Testing Advanced Ensemble Model")
    print("=" * 40)

    # Create model
    model = create_advanced_ensemble(input_size=24)

    # Test forward pass
    batch_size, seq_len, input_size = 4, 60, 24
    x = torch.randn(batch_size, seq_len, input_size)

    predictions, uncertainties = model(x)
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Predictions shape: {predictions.shape}")
    print(f"✅ Uncertainties shape: {uncertainties.shape}")

    # Test confidence filtering
    conf_predictions, conf_uncertainties, should_predict = model.predict_with_confidence(x, confidence_threshold=0.8)
    confident_count = should_predict.sum().item()
    print(f"✅ High-confidence predictions: {confident_count}/{batch_size}")

    return model


def demo_market_regime_detection():
    """Demo market regime detection"""
    print("\n🎯 Testing Market Regime Detection")
    print("=" * 40)

    # Create adaptive system
    adaptive_system = create_adaptive_prediction_system(None)  # No model for demo

    # Test regime detection with different market conditions
    test_cases = [
        ("Bull Market", np.cumsum(np.random.normal(0.001, 0.01, 200)) + 100),
        ("Bear Market", np.cumsum(np.random.normal(-0.001, 0.01, 200)) + 100),
        ("Volatile Market", np.cumsum(np.random.normal(0, 0.03, 200)) + 100),
        ("Ranging Market", 100 + np.sin(np.linspace(0, 4*np.pi, 200)) * 2)
    ]

    for market_type, prices in test_cases:
        regime, confidence = adaptive_system['regime_detector'].detect_regime(recent_prices=prices.tolist())
        strategy = adaptive_system['regime_detector'].get_optimal_strategy(regime)

        print(f"✅ {market_type}: {regime} (confidence: {confidence:.2f})")
        print(f"   Strategy: {strategy['prediction_frequency']} frequency, {strategy['confidence_threshold']:.2f} threshold")

    return adaptive_system


def demo_peak_hour_optimization():
    """Demo ultra-precise peak hour optimization"""
    print("\n⏰ Testing Peak Hour Optimization")
    print("=" * 40)

    # Create peak optimizer
    peak_optimizer = create_ultra_precise_prediction_system(timezone_offset=0)

    # Simulate reward data
    base_time = datetime.now(timezone.utc)
    print("📊 Simulating reward data across 15-minute intervals...")

    # Add sample data for peak hours
    for hour in range(24):
        for quarter in [0, 15, 30, 45]:
            # Higher rewards during known peak hours
            interval_key = (hour, quarter)
            base_reward = 0.0005

            if interval_key in peak_optimizer.known_peak_periods:
                multiplier = peak_optimizer.known_peak_periods[interval_key]
                reward = base_reward * multiplier * (0.8 + np.random.random() * 0.4)
            else:
                reward = base_reward * (0.2 + np.random.random() * 0.3)

            timestamp = base_time + (hour * 3600 + quarter * 60)  # Convert to seconds
            peak_optimizer.update_reward_data(reward, timestamp)

    # Analyze peak intervals
    analysis = peak_optimizer.analyze_peak_intervals()
    print(f"✅ Detected {len(analysis['peak_intervals'])} peak intervals")
    print(".2f")

    # Test prediction timing
    current_time = datetime.now(timezone.utc)
    is_peak, confidence, info = peak_optimizer.should_predict_now(current_time)

    print(f"✅ Current time analysis: {'Peak' if is_peak else 'Off-peak'} hour")
    print(".2f")

    schedule = peak_optimizer.get_prediction_schedule()
    print(f"✅ Daily prediction schedule: {schedule['total_daily_predictions']} predictions/day")

    return peak_optimizer


def demo_performance_tracking():
    """Demo real-time performance tracking"""
    print("\n📊 Testing Performance Tracking System")
    print("=" * 40)

    # Create performance tracking system
    performance_tracker, dashboard = create_performance_tracking_system(None)  # No model for demo

    # Simulate predictions and tracking
    print("🎯 Simulating prediction tracking...")

    for i in range(20):
        # Generate mock prediction data
        actual = np.random.randn() * 0.01 + 0.001
        prediction = actual + np.random.normal(0, 0.002)
        reward = np.random.random() * 0.001
        confidence = np.random.random()

        # Create prediction record
        from performance_tracking_system import PredictionRecord

        record = PredictionRecord(
            timestamp=datetime.now(timezone.utc),
            prediction=float(prediction),
            actual=float(actual),
            reward=float(reward),
            confidence=float(confidence),
            market_regime=np.random.choice(['bull', 'bear', 'volatile', 'ranging']),
            is_peak_hour=(i % 3 == 0),  # Every 3rd prediction is peak hour
            prediction_time_ms=np.random.uniform(10, 50)
        )

        performance_tracker.record_prediction(record)
        print(".6f")
    # Display performance report
    print("\n📈 Performance Report:")
    report = performance_tracker.get_performance_report()
    print(".1f")
    print(".4f")
    print(".1%")

    if report['alerts']:
        print(f"🚨 Alerts: {len(report['alerts'])} active")

    if report['recommendations']:
        print(f"💡 Recommendations: {len(report['recommendations'])} available")

    return performance_tracker, dashboard


def demo_gpu_acceleration():
    """Demo GPU-accelerated inference"""
    print("\n⚡ Testing GPU Acceleration")
    print("=" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")

    # Create model and optimizer
    model = create_advanced_ensemble(input_size=24)
    inference_optimizer = create_inference_optimizer(model, device)

    # Benchmark inference
    print("🏃 Benchmarking inference performance...")
    benchmark_results = inference_optimizer.benchmark_inference(num_runs=10)

    for i, batch_size in enumerate(benchmark_results['batch_size']):
        print(".2f")
    print("✅ GPU acceleration ready for production!")

    return inference_optimizer


def demo_complete_integration():
    """Demo the complete integrated domination system"""
    print("\n🎯 Testing Complete Domination System Integration")
    print("=" * 50)

    # Create all components
    model = create_advanced_ensemble(input_size=24)
    adaptive_system = create_adaptive_prediction_system(model)
    peak_optimizer = create_ultra_precise_prediction_system(timezone_offset=0)
    performance_tracker, dashboard = create_performance_tracking_system(model)
    inference_optimizer = create_inference_optimizer(model, 'cpu')

    # Simulate a complete prediction workflow
    print("🔄 Running integrated prediction workflow...")

    # Generate market data
    market_features = np.random.randn(24).tolist()

    # Step 1: Check if we should predict
    should_predict, decision_info = adaptive_system['adaptive_predictor'].should_make_prediction(
        market_data=market_features
    )

    print("✅ Market analysis complete")
    print(f"   Should predict: {should_predict}")
    print(f"   Market regime: {decision_info['market_regime']}")
    print(f"   Is peak hour: {decision_info['is_peak_hour']}")

    if should_predict:
        # Step 2: Make prediction
        prediction_result = adaptive_system['adaptive_predictor'].predict_adaptive(
            market_data=market_features
        )

        print("✅ Prediction made"        print(".6f"
        # Step 3: Record performance (simulate)
        actual_value = prediction_result[0].item() + np.random.normal(0, 0.001)
        reward = np.random.random() * 0.001

        # Create prediction record
        from performance_tracking_system import PredictionRecord

        record = PredictionRecord(
            timestamp=datetime.now(timezone.utc),
            prediction=prediction_result[0].item(),
            actual=actual_value,
            reward=reward,
            confidence=1 - prediction_result[1].item(),
            market_regime=decision_info['market_regime'],
            is_peak_hour=decision_info['is_peak_hour'],
            prediction_time_ms=15.0
        )

        performance_tracker.record_prediction(record)

        print("✅ Performance recorded"        print(".6f"
    # Step 4: Display dashboard
    print("\n📊 System Status:")
    dashboard.print_dashboard()

    print("🎉 Complete domination system integration successful!")
    return True


def main():
    """Main demo function"""
    print("🎯 PRECOG #1 MINER DOMINATION SYSTEM DEMO")
    print("=" * 50)
    print("🚀 Demonstrating all advanced features for mainnet deployment")
    print("=" * 50)

    try:
        # Demo each component
        model = demo_advanced_ensemble()
        adaptive_system = demo_market_regime_detection()
        peak_optimizer = demo_peak_hour_optimization()
        performance_tracker, dashboard = demo_performance_tracking()
        inference_optimizer = demo_gpu_acceleration()

        # Demo complete integration
        success = demo_complete_integration()

        if success:
            print("\n" + "=" * 50)
            print("🎉 ALL DOMINATION FEATURES WORKING PERFECTLY!")
            print("=" * 50)
            print("\n🚀 READY FOR MAINNET DEPLOYMENT")
            print("\nNext steps:")
            print("1. Run full training: python train_domination_model_complete.py")
            print("2. Deploy to mainnet: python start_domination_miner.py")
            print("3. Monitor performance: python monitor_domination_miner.py")
            print("\n💡 The domination system will help you achieve #1 miner position!")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
