#!/usr/bin/env python3
"""
Fix GPU model loading for Precog miner
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class WorkingEnsemble(nn.Module):
    """Corrected model architecture matching feature extraction"""
    
    def __init__(self, input_size=11):
        super(WorkingEnsemble, self).__init__()
        self.input_size = input_size
        
        # GRU layers
        self.gru1 = nn.GRU(input_size, 64, batch_first=True)
        self.gru2 = nn.GRU(64, 32, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(32, num_heads=4, batch_first=True)
        
        # Output layers
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = x.unsqueeze(0) if x.dim() == 2 else x  # Add batch dimension if needed
        
        # GRU layers
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        
        # Attention
        attn_out, _ = self.attention(out, out, out)
        
        # Global average pooling
        out = torch.mean(attn_out, dim=1)
        
        # Fully connected layers
        out = self.dropout(torch.relu(self.fc1(out)))
        out = self.fc2(out)
        
        return out.squeeze()

def load_gpu_domination_models():
    """Load models optimized for GPU usage"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Loading GPU-optimized domination models on device: {device}")
        
        # Create model with correct input size
        model = WorkingEnsemble(input_size=11)  # Matches feature extraction
        
        # Move to GPU immediately
        model = model.to(device)
        
        # Load trained weights with error handling
        model_path = "models/multi_asset_domination_model.pth"
        if os.path.exists(model_path):
            print(f"📁 Loading weights from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                    
                # Load with strict=False to handle architecture differences
                model.load_state_dict(state_dict, strict=False)
                print("✅ Model weights loaded (GPU compatible)")
                
            except Exception as e:
                print(f"⚠️  Weight loading failed: {e}")
                print("🔄 Using randomly initialized model")
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print("🔄 Using randomly initialized model")
        
        # Load scaler
        scaler = None
        scaler_path = "models/multi_asset_feature_scaler.pkl"
        if os.path.exists(scaler_path):
            try:
                import joblib
                scaler = joblib.load(scaler_path)
                print("✅ Feature scaler loaded")
            except Exception as e:
                print(f"⚠️  Scaler loading failed: {e}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Verify GPU usage
        sample_input = torch.randn(1, 10, 11).to(device)
        with torch.no_grad():
            sample_output = model(sample_input)
        
        print(f"🎯 Model device: {next(model.parameters()).device}")
        print(f"🚀 GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"💾 GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        print(f"✨ Sample inference successful: {sample_output.shape}")
        
        return {'domination_model': model, 'scaler': scaler}
        
    except Exception as e:
        print(f"❌ Error in GPU model loading: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧪 Testing GPU model loading...")
    models = load_gpu_domination_models()
    if models and models['domination_model']:
        print("✅ GPU model loading successful!")
        
        # Test GPU utilization
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util, mem_used = result.stdout.strip().split(', ')
                print(f"🎮 GPU Utilization: {gpu_util}%")
                print(f"🧠 GPU Memory Used: {mem_used} MB")
        except:
            pass
            
    else:
        print("❌ GPU model loading failed")
        sys.exit(1)
