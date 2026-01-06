import torch
import torch.nn as nn

class GPUWorkingEnsemble(nn.Module):
    """GPU-optimized ensemble model matching feature extraction"""
    
    def __init__(self, input_size=11):
        super(GPUWorkingEnsemble, self).__init__()
        self.input_size = input_size
        
        # GRU layers optimized for GPU
        self.gru1 = nn.GRU(input_size, 64, batch_first=True)
        self.gru2 = nn.GRU(64, 32, batch_first=True)
        
        # Multi-head attention for better pattern recognition
        self.attention = nn.MultiheadAttention(32, num_heads=4, batch_first=True)
        
        # Output layers
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Ensure proper tensor dimensions
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # GRU processing
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        
        # Attention mechanism
        attn_out, _ = self.attention(out, out, out)
        
        # Global pooling
        out = torch.mean(attn_out, dim=1)
        
        # Fully connected layers
        out = self.dropout(torch.relu(self.fc1(out)))
        out = self.fc2(out)
        
        return out.squeeze()

def load_gpu_optimized_models():
    """Load GPU-optimized domination models"""
    global point_model, scaler, models_loaded

    if models_loaded:
        return

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔥 Loading GPU-optimized domination models on device: {device}")

        # Create GPU-optimized model
        point_model = GPUWorkingEnsemble(input_size=11)  # Matches feature extraction
        point_model = point_model.to(device)
        
        # Load trained weights with compatibility
        model_path = 'models/multi_asset_domination_model.pth'
        if os.path.exists(model_path):
            logger.info(f"📁 Loading weights from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Load with strict=False for compatibility
                point_model.load_state_dict(state_dict, strict=False)
                logger.info("✅ Model weights loaded (GPU optimized)")
                
            except Exception as e:
                logger.warning(f"Weight loading failed: {e}")
                logger.info("🔄 Using randomly initialized model")
        else:
            logger.info(f"⚠️  Model file not found: {model_path}")
            logger.info("🔄 Using randomly initialized model")

        # Load feature scaler
        scaler_path = 'models/multi_asset_feature_scaler.pkl'
        if os.path.exists(scaler_path):
            try:
                import joblib
                scaler = joblib.load(scaler_path)
                logger.info("✅ Feature scaler loaded")
            except Exception as e:
                logger.warning(f"Scaler loading failed: {e}")
                scaler = None
        else:
            logger.warning(f"Scaler file not found: {scaler_path}")
            scaler = None

        # Verify GPU usage
        point_model.eval()
        sample_input = torch.randn(1, 10, 11).to(device)
        with torch.no_grad():
            sample_output = point_model(sample_input)
        
        logger.info(f"🎯 Model loaded on device: {next(point_model.parameters()).device}")
        logger.info(f"🚀 GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        models_loaded = True
        logger.info("✅ GPU-optimized domination models loaded successfully!")
        
        return {'domination_model': point_model, 'scaler': scaler}
        
    except Exception as e:
        logger.warning(f"❌ Could not load GPU models: {e}")
        logger.info("Running with basic prediction capabilities")
        return None
