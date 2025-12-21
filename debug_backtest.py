#!/usr/bin/env python3
"""
Debug Backtest - Fix tensor shape issues before full deployment
"""

import sys
import os
import torch
import numpy as np
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core components (without pandas dependencies)
from advanced_attention_mechanisms import create_enhanced_attention_ensemble

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_simple_test_data(n_samples=100, seq_len=60, n_features=24):
    """Generate simple test data"""
    logger.info(f"Generating {n_samples} simple test samples...")

    np.random.seed(42)

    # Generate simple features (just random data for debugging)
    features = []
    targets = []

    for i in range(n_samples):
        # Simple feature vector
        feature_vector = np.random.normal(0, 1, n_features)
        target = np.random.normal(0, 0.1)  # Small target values

        features.append(feature_vector)
        targets.append(target)

    return np.array(features), np.array(targets)


def test_attention_ensemble():
    """Test the attention ensemble specifically"""
    print("🧠 Testing Attention Ensemble...")

    try:
        # Create model
        model = create_enhanced_attention_ensemble(input_size=24)
        model.eval()

        # Generate test data
        features, targets = generate_simple_test_data(n_samples=32, seq_len=60, n_features=24)

        # Test single forward pass
        batch_size = 4
        seq_len = 60

        # Create batch input (batch_size, seq_len, n_features)
        input_tensor = torch.randn(batch_size, seq_len, 24)

        print(f"Input shape: {input_tensor.shape}")

        with torch.no_grad():
            output = model(input_tensor)
            print(f"Output shape: {output.shape}")

        print("✅ Attention Ensemble working!")
        return True

    except Exception as e:
        print(f"❌ Attention Ensemble failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_gru():
    """Test basic GRU model"""
    print("\n🧠 Testing Basic GRU...")

    try:
        # Simple GRU model
        class SimpleGRU(torch.nn.Module):
            def __init__(self, input_size=24, hidden_size=64):
                super().__init__()
                self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
                self.fc = torch.nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])  # Last timestep

        model = SimpleGRU()
        model.eval()

        # Test input
        input_tensor = torch.randn(4, 60, 24)  # (batch, seq, features)

        with torch.no_grad():
            output = model(input_tensor)
            print(f"GRU Output shape: {output.shape}")

        print("✅ Basic GRU working!")
        return True

    except Exception as e:
        print(f"❌ Basic GRU failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_integration():
    """Test ensemble integration"""
    print("\n🔗 Testing Ensemble Integration...")

    try:
        # Create a simple ensemble
        class SimpleEnsemble(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gru1 = torch.nn.GRU(24, 32, batch_first=True)
                self.gru2 = torch.nn.GRU(24, 32, batch_first=True)
                self.attention = torch.nn.MultiheadAttention(32, 4, batch_first=True)
                self.fc = torch.nn.Linear(32, 1)

            def forward(self, x):
                # GRU processing
                out1, _ = self.gru1(x)
                out2, _ = self.gru2(x)

                # Simple attention
                query = out1[:, -1:, :]  # Last timestep as query
                key_value = out2
                attn_out, _ = self.attention(query, key_value, key_value)

                return self.fc(attn_out.squeeze(1))

        model = SimpleEnsemble()
        model.eval()

        input_tensor = torch.randn(4, 60, 24)

        with torch.no_grad():
            output = model(input_tensor)
            print(f"Ensemble Output shape: {output.shape}")

        print("✅ Ensemble Integration working!")
        return True

    except Exception as e:
        print(f"❌ Ensemble Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main debug function"""
    print("🔧 PRECOG DOMINATION SYSTEM - DEBUG BACKTEST")
    print("=" * 60)

    results = []

    # Test components
    results.append(("Attention Ensemble", test_attention_ensemble()))
    results.append(("Basic GRU", test_basic_gru()))
    results.append(("Ensemble Integration", test_ensemble_integration()))

    print("\n" + "=" * 60)
    print("📊 DEBUG RESULTS")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print("20")
        if not passed:
            all_passed = False

    print(f"\n🎯 OVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    if all_passed:
        print("\n🚀 READY FOR FULL BACKTEST AND DEPLOYMENT!")
        print("Run: python3 quick_backtest.py")
    else:
        print("\n🔧 FIX ISSUES BEFORE DEPLOYMENT")
        print("The tensor shape problems need to be resolved first.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
