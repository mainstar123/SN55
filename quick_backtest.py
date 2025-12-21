#!/usr/bin/env python3
"""
Quick Backtesting for Precog Domination System
Lightweight version that tests core improvements
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime, timezone, timedelta
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core components
from advanced_ensemble_model import create_advanced_ensemble
from advanced_attention_mechanisms import create_enhanced_attention_ensemble

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_data(n_samples=100, seq_len=60, n_features=24):
    """Generate synthetic test data with proper sequences"""
    logger.info(f"Generating {n_samples} test sequences of length {seq_len}...")

    # Generate realistic price data
    np.random.seed(42)
    base_price = 100.0
    total_points = n_samples * seq_len + seq_len  # Extra for targets

    # Create price series with trends and volatility
    returns = np.random.normal(0.0001, 0.02, total_points)
    prices = base_price * np.cumprod(1 + returns)

    # Create sequences
    features = []
    targets = []

    for i in range(n_samples):
        start_idx = i * seq_len
        end_idx = start_idx + seq_len

        # Get price sequence
        price_seq = prices[start_idx:end_idx]

        # Create features for each timestep in the sequence
        sequence_features = []
        for j in range(seq_len):
            # Technical indicators for this timestep
            lookback = min(20, j+1)  # Look back up to 20 periods
            recent_prices = price_seq[max(0, j-lookback+1):j+1]

            timestep_features = [
                np.mean(recent_prices),  # SMA
                np.std(recent_prices) if len(recent_prices) > 1 else 0,  # Volatility
                recent_prices[-1] - recent_prices[-2] if len(recent_prices) > 1 else 0,  # Recent change
                np.max(recent_prices) - np.min(recent_prices),  # Range
                price_seq[j],  # Current price
            ]

            # Pad to required feature count
            while len(timestep_features) < n_features:
                timestep_features.append(np.random.normal(0, 0.1))

            sequence_features.append(timestep_features)

        features.append(sequence_features)

        # Target: price change after the sequence
        target_price = prices[end_idx] if end_idx < len(prices) else prices[-1]
        current_price = price_seq[-1]
        targets.append(target_price - current_price)

    return np.array(features), np.array(targets)


def test_model_performance(model, features, targets, model_name="Model"):
    """Test model performance"""
    logger.info(f"Testing {model_name}...")

    model.eval()
    predictions = []
    actuals = []
    confidences = []

    # Test one sequence at a time (models expect full sequences)
    for i in range(min(10, len(features))):  # Test first 10 sequences for speed
        sequence_features = features[i]  # Shape: (60, 24) - full sequence
        sequence_target = targets[i]      # Shape: scalar

        # Prepare input - full sequence
        if model_name == "TCN":
            # TCN expects (batch=1, features, seq_len)
            input_tensor = torch.FloatTensor(sequence_features).transpose(0, 1).unsqueeze(0)  # (1, 24, 60)
        else:
            # Standard format: (batch=1, seq_len, features)
            input_tensor = torch.FloatTensor(sequence_features).unsqueeze(0)  # (1, 60, 24)

        with torch.no_grad():
            try:
                # Handle both single-output and tuple-output models
                output = model(input_tensor)

                if isinstance(output, tuple):
                    pred, uncertainty = output
                    pred_val = pred.squeeze().cpu().numpy()
                    uncertainty_val = uncertainty.squeeze().cpu().numpy()
                    confidence = 1 - uncertainty_val
                else:
                    # Single output model (no uncertainty)
                    pred_val = output.squeeze().cpu().numpy()
                    confidence = 0.5  # Default confidence for models without uncertainty

                # Ensure scalar prediction
                if np.isscalar(pred_val) or (hasattr(pred_val, 'shape') and pred_val.shape == ()) or pred_val.ndim == 0:
                    pred_scalar = float(pred_val) if np.isscalar(pred_val) else float(pred_val.item())
                    predictions.append(pred_scalar)
                    actuals.append(float(sequence_target))
                    confidences.append(float(confidence))

            except Exception as e:
                print(f"❌ Error in {model_name} at sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        # Silent processing for clean output

    if not predictions:
        return {"mape": 1.0, "mae": 1.0, "total_predictions": 0}

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    mape = np.mean(np.abs((predictions - actuals) / (np.abs(actuals) + 1e-6)))
    mae = np.mean(np.abs(predictions - actuals))

    return {
        "mape": mape,
        "mae": mae,
        "total_predictions": len(predictions),
        "avg_prediction": np.mean(predictions),
        "avg_actual": np.mean(actuals)
    }


def run_quick_backtest():
    """Run quick comparative backtest"""
    print("🎯 PRECOG DOMINATION SYSTEM - QUICK BACKTEST")
    print("=" * 60)

    # Generate test data
    print("\n📊 Generating test data...")
    features, targets = generate_test_data(n_samples=500, seq_len=60, n_features=24)

    print(f"✅ Generated {len(features)} samples with {features.shape[1]} features")

    # Test models
    models_to_test = {
        "Original Ensemble": create_advanced_ensemble(),
        "Attention Enhanced": create_enhanced_attention_ensemble(),
    }

    results = {}

    print("\n🚀 Testing models...")
    for model_name, model in models_to_test.items():
        try:
            result = test_model_performance(model, features, targets, model_name)
            results[model_name] = result

            print(f"\n📈 {model_name}:")
            print(".4f")
            print(".4f")
            print(f"   Predictions: {result['total_predictions']}")

        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            results[model_name] = {"mape": 1.0, "mae": 1.0, "error": str(e)}

    # Compare results
    print("\n" + "=" * 60)
    print("🏆 BACKTEST RESULTS COMPARISON")
    print("=" * 60)

    baseline = results.get("Original Ensemble", {})
    baseline_mape = baseline.get("mape", 1.0)

    for model_name, result in results.items():
        if "error" not in result:
            mape = result["mape"]
            improvement = (baseline_mape - mape) / baseline_mape * 100

            print(f"\n{model_name}:")
            print(".4f")
            print(".4f")
            print(".1f")
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1].get("mape", 1.0))
    best_mape = best_model[1].get("mape", 1.0)
    best_improvement = (baseline_mape - best_mape) / baseline_mape * 100

    print("\n🎯 SUMMARY:")
    print(f"Best Model: {best_model[0]}")
    print(".4f")
    print(".1f")
    if best_improvement > 10:
        print("🚀 STRONG IMPROVEMENT - READY FOR DEPLOYMENT!")
    elif best_improvement > 0:
        print("✅ MODERATE IMPROVEMENT - CONSIDER DEPLOYMENT")
    else:
        print("⚠️  LIMITED IMPROVEMENT - MORE TUNING NEEDED")

    print("\n💡 NEXT STEPS:")
    print("1. Deploy best model to mainnet for real testing")
    print("2. Monitor performance with advanced tracking")
    print("3. Continue optimization based on real market data")

    return results


def main():
    """Main function"""
    try:
        results = run_quick_backtest()
        return 0
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
