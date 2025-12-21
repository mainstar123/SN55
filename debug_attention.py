#!/usr/bin/env python3
"""
Debug script to isolate attention mechanism issues
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from advanced_attention_mechanisms import MultiScaleAttention, EnhancedEnsembleWithAttention

def test_multiscale_attention():
    print("🔍 Testing MultiScaleAttention...")

    embed_dim = 128
    batch_size, seq_len = 1, 60  # Match our backtest

    # Create attention
    attn = MultiScaleAttention(embed_dim, num_heads=8)

    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"Input shape: {x.shape}")

    try:
        out = attn(x, x, x)
        print(f"✅ Output shape: {out.shape}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_ensemble():
    print("\n🔍 Testing EnhancedEnsembleWithAttention...")

    batch_size, seq_len, input_size = 1, 60, 24

    # Create model
    model = EnhancedEnsembleWithAttention(input_size=input_size)

    # Create test input
    x = torch.randn(batch_size, seq_len, input_size)
    print(f"Input shape: {x.shape}")

    try:
        out, uncertainty = model(x)
        print(f"✅ Output shape: {out.shape}, Uncertainty shape: {uncertainty.shape}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_shapes():
    print("🔧 Debugging tensor shapes step by step...")

    embed_dim = 128
    batch_size, seq_len = 1, 60
    num_heads = 8

    # Test MultiheadAttention directly
    print("\n1. Testing PyTorch MultiheadAttention directly...")
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    x = torch.randn(batch_size, seq_len, embed_dim)

    try:
        out, weights = mha(x, x, x)
        print(f"✅ MHA Input: {x.shape}, Output: {out.shape}, Weights: {weights.shape}")
    except Exception as e:
        print(f"❌ MHA Error: {e}")

    # Test our multi-scale attention with different scales
    print("\n2. Testing multi-scale attention components...")

    scale_factors = [1, 2, 4, 8]
    scale_attentions = [
        nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        for _ in scale_factors
    ]

    x = torch.randn(batch_size, seq_len, embed_dim)
    scale_outputs = []

    for i, scale_factor in enumerate(scale_factors):
        print(f"   Scale {scale_factor}:")
        try:
            if scale_factor > 1:
                seq_len_scaled = seq_len // scale_factor
                if seq_len_scaled > 0:
                    indices = torch.arange(0, seq_len, scale_factor)
                    q_scaled = x[:, indices]
                    k_scaled = x[:, indices]
                    v_scaled = x[:, indices]
                    print(f"     Scaled seq_len: {q_scaled.shape[1]}")
                else:
                    q_scaled, k_scaled, v_scaled = x, x, x
            else:
                q_scaled, k_scaled, v_scaled = x, x, x

            attn_output, _ = scale_attentions[i](q_scaled, k_scaled, v_scaled)
            print(f"     Attention output: {attn_output.shape}")
            scale_outputs.append(attn_output)

        except Exception as e:
            print(f"     ❌ Error at scale {scale_factor}: {e}")

    print(f"\n3. Checking scale_outputs alignment...")
    for i, output in enumerate(scale_outputs):
        print(f"   Scale {scale_factors[i]} output shape: {output.shape}")

    # Check if we can align them
    target_seq_len = seq_len
    print(f"   Target seq_len: {target_seq_len}")

    aligned_outputs = []
    for i, output in enumerate(scale_outputs):
        current_seq_len = output.shape[1]
        print(f"   Scale {scale_factors[i]}: current={current_seq_len}, target={target_seq_len}")

        if current_seq_len != target_seq_len:
            try:
                # Interpolate to target sequence length
                output_interp = output.transpose(1, 2)  # (batch, embed_dim, seq_len)
                output_interp = torch.nn.functional.interpolate(output_interp, size=target_seq_len, mode='linear')
                output_interp = output_interp.transpose(1, 2)  # (batch, seq_len, embed_dim)
                aligned_outputs.append(output_interp)
                print(f"     ✅ Interpolated to: {output_interp.shape}")
            except Exception as e:
                print(f"     ❌ Interpolation failed: {e}")
                aligned_outputs.append(output)  # Keep original
        else:
            aligned_outputs.append(output)

    # Check concatenation
    print(f"\n4. Testing concatenation...")
    try:
        concat_output = torch.cat(aligned_outputs, dim=-1)
        print(f"✅ Concatenated shape: {concat_output.shape}")
        print(f"   Expected features: {len(scale_factors) * embed_dim}")
        print(f"   Actual features: {concat_output.shape[-1]}")
    except Exception as e:
        print(f"❌ Concatenation failed: {e}")

if __name__ == "__main__":
    print("🧠 DEBUGGING ATTENTION MECHANISMS")
    print("=" * 50)

    # Test individual components
    multi_scale_ok = test_multiscale_attention()
    ensemble_ok = test_enhanced_ensemble()

    if not multi_scale_ok or not ensemble_ok:
        print("\n🔧 Running detailed shape debugging...")
        debug_shapes()

    print("\n" + "=" * 50)
    if multi_scale_ok and ensemble_ok:
        print("🎉 All attention mechanisms working!")
    else:
        print("⚠️ Issues found - see above for details")
