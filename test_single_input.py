#!/usr/bin/env python3
"""Test models with single input"""

import torch
import sys
sys.path.append('.')

from advanced_ensemble_model import create_advanced_ensemble
from advanced_attention_mechanisms import create_enhanced_attention_ensemble

def test_single_input():
    print("🧪 Testing single input processing...")

    # Create models
    original_model = create_advanced_ensemble()
    attention_model = create_enhanced_attention_ensemble()

    # Set to eval mode (important for BatchNorm!)
    original_model.eval()
    attention_model.eval()

    # Create test input (1, 60, 24) - batch=1, seq_len=60, features=24
    x = torch.randn(1, 60, 24)
    print(f"Input shape: {x.shape}")

    # Test original model
    print("\n1. Testing Original Ensemble...")
    try:
        out_orig = original_model(x)
        if isinstance(out_orig, tuple):
            pred, unc = out_orig
            print(f"✅ Original: pred shape {pred.shape}, uncertainty shape {unc.shape}")
        else:
            print(f"✅ Original: output shape {out_orig.shape}")
    except Exception as e:
        print(f"❌ Original failed: {e}")
        import traceback
        traceback.print_exc()

    # Test attention model
    print("\n2. Testing Attention Enhanced...")
    try:
        out_attn = attention_model(x)
        if isinstance(out_attn, tuple):
            pred, unc = out_attn
            print(f"✅ Attention: pred shape {pred.shape}, uncertainty shape {unc.shape}")
        else:
            print(f"✅ Attention: output shape {out_attn.shape}")
    except Exception as e:
        print(f"❌ Attention failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_input()
