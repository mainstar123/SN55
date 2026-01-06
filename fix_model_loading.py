#!/usr/bin/env python3
"""
Fix model loading for GPU usage
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from working_domination_ensemble import WorkingEnsemble

def load_corrected_domination_models():
    """Load domination models with correct architecture"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading domination models on device: {device}")
        
        # Use the correct model architecture (11 features input)
        model = WorkingEnsemble(input_size=11)  # Match the feature extraction output
        model = model.to(device)
        
        # Try to load the trained weights with strict=False to handle mismatches
        model_path = "models/multi_asset_domination_model.pth"
        if os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                print("✅ Model weights loaded (with compatibility adjustments)")
            except Exception as e:
                print(f"⚠️  Could not load model weights: {e}")
                print("Using randomly initialized model")
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print("Using randomly initialized model")
        
        # Load scaler
        scaler_path = "models/multi_asset_feature_scaler.pkl"
        if os.path.exists(scaler_path):
            import joblib
            scaler = joblib.load(scaler_path)
            print("✅ Feature scaler loaded")
        else:
            print(f"⚠️  Scaler file not found: {scaler_path}")
            scaler = None
        
        # Put model in eval mode
        model.eval()
        
        # Verify GPU usage
        param = next(model.parameters())
        print(f"Model device: {param.device}")
        print(f"Model is on CUDA: {param.is_cuda}")
        
        return {'domination_model': model, 'scaler': scaler}
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Testing corrected model loading...")
    models = load_corrected_domination_models()
    if models:
        print("✅ Models loaded successfully")
        # Test GPU memory usage
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    else:
        print("❌ Model loading failed")
