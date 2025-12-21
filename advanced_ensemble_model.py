"""
Advanced Ensemble Model for Precog #1 Miner Position
Combines GRU, Transformer, and LSTM with Meta-Learning for optimal predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    import datetime
try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Simple replacement for StandardScaler
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None

        def fit(self, X):
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            return self

        def transform(self, X):
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)
import logging

logger = logging.getLogger(__name__)

class AdvancedGRU(nn.Module):
    """Enhanced GRU with attention and residual connections"""
    def __init__(self, input_size: int = 24, hidden_size: int = 128, num_layers: int = 3,
                 dropout: float = 0.1, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Multi-layer GRU with increasing hidden sizes
        self.gru_layers = nn.ModuleList()
        current_input = input_size
        for i in range(num_layers):
            current_hidden = hidden_size * (2 ** i)  # Exponential growth
            self.gru_layers.append(nn.GRU(current_input, current_hidden,
                                        batch_first=True, dropout=dropout if i > 0 else 0))
            current_input = current_hidden

        # Attention mechanism
        self.attention = nn.MultiheadAttention(current_hidden, num_heads=8, dropout=dropout, batch_first=True)

        # Output layers with residual connections
        self.layer_norm1 = nn.LayerNorm(current_hidden)
        self.layer_norm2 = nn.LayerNorm(current_hidden // 4)  # Match fc2 output

        self.fc1 = nn.Linear(current_hidden, current_hidden // 2)
        self.fc2 = nn.Linear(current_hidden // 2, current_hidden // 4)
        self.fc3 = nn.Linear(current_hidden // 4, output_size)

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(current_hidden // 2)

    def forward(self, x):
        # Process through GRU layers
        for i, gru in enumerate(self.gru_layers):
            out, _ = gru(x)
            x = out

        # Attention mechanism
        query = x[:, -1:, :]  # Use last timestep as query
        key_value = x
        attn_out, _ = self.attention(query, key_value, key_value)
        out = self.layer_norm1(attn_out.squeeze(1))

        # Residual output layers
        residual = out
        out = F.relu(self.fc1(out))
        out = self.batch_norm(out)
        out = self.dropout(out)

        if out.shape[-1] == residual.shape[-1]:
            out = out + residual

        out = F.relu(self.fc2(out))
        out = self.layer_norm2(out)
        out = self.dropout(out)

        return self.fc3(out)


class AdvancedTransformer(nn.Module):
    """Transformer encoder with positional encoding and multiple heads"""
    def __init__(self, input_size: int = 24, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1, output_size: int = 1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_size)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use mean pooling for sequence output
        x = x.mean(dim=1)
        return self.output_projection(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AdvancedLSTM(nn.Module):
    """Advanced LSTM with bidirectional processing and attention"""
    def __init__(self, input_size: int = 24, hidden_size: int = 128, num_layers: int = 3,
                 dropout: float = 0.1, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        # Attention for bidirectional outputs
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        # LSTM processing
        outputs, (h_n, c_n) = self.lstm(x)  # outputs: (batch, seq, hidden*2)

        # Attention mechanism
        attention_weights = F.softmax(self.attention(outputs), dim=1)  # (batch, seq, 1)
        attended = torch.sum(outputs * attention_weights, dim=1)  # (batch, hidden*2)

        attended = self.layer_norm(attended)

        # Output layers
        out = F.relu(self.fc1(attended))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)

        return self.fc3(out)


class MetaLearner(nn.Module):
    """Meta-learner that dynamically weights ensemble predictions"""
    def __init__(self, num_models: int = 3, feature_size: int = 24):
        super().__init__()
        self.num_models = num_models

        # Context encoder for dynamic weighting
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_models),
            nn.Softmax(dim=1)
        )

    def forward(self, model_outputs: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model_outputs: (batch, num_models) - outputs from individual models
            context_features: (batch, feature_size) - current market context
        Returns:
            weighted_output: (batch, 1) - ensemble prediction
        """
        # Generate dynamic weights based on context
        weights = self.context_encoder(context_features)  # (batch, num_models)

        # Apply weights to model outputs
        weighted_output = torch.sum(model_outputs * weights, dim=1, keepdim=True)

        return weighted_output


class AdvancedEnsemble(nn.Module):
    """
    Advanced Ensemble combining GRU, Transformer, LSTM with Meta-Learning
    Designed to achieve #1 miner position in Precog subnet
    """
    def __init__(self, input_size: int = 24, hidden_size: int = 128,
                 dropout: float = 0.1, output_size: int = 1):
        super().__init__()
        self.input_size = input_size

        # Individual models
        self.gru_model = AdvancedGRU(input_size, hidden_size, dropout=dropout)
        self.transformer_model = AdvancedTransformer(input_size, hidden_size, dropout=dropout)
        self.lstm_model = AdvancedLSTM(input_size, hidden_size, dropout=dropout)

        # Meta-learner for dynamic weighting
        self.meta_learner = MetaLearner(num_models=3, feature_size=input_size)

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0-1
        )

        # Feature scaler
        self.feature_scaler = StandardScaler()

        # Performance tracking
        self.prediction_history = []
        self.accuracy_history = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with ensemble prediction and uncertainty
        Returns: (prediction, uncertainty)
        """
        batch_size, seq_len, _ = x.shape

        # Get context features (last timestep)
        context_features = x[:, -1, :]  # (batch, input_size)

        # Individual model predictions
        gru_out = self.gru_model(x)  # (batch, 1)
        transformer_out = self.transformer_model(x)  # (batch, 1)
        lstm_out = self.lstm_model(x)  # (batch, 1)

        # Stack model outputs
        model_outputs = torch.cat([gru_out, transformer_out, lstm_out], dim=1)  # (batch, 3)

        # Meta-learner weighting
        ensemble_pred = self.meta_learner(model_outputs, context_features)  # (batch, 1)

        # Uncertainty estimation
        uncertainty = self.uncertainty_head(context_features)  # (batch, 1)

        return ensemble_pred, uncertainty

    def predict_with_confidence(self, x: torch.Tensor,
                              confidence_threshold: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions only when confidence is high enough
        Returns: (predictions, uncertainties, should_predict_mask)
        """
        predictions, uncertainties = self.forward(x)
        confidence_scores = 1 - uncertainties  # Higher confidence = lower uncertainty

        # Only predict when confidence is above threshold
        should_predict = confidence_scores >= confidence_threshold

        return predictions, uncertainties, should_predict

    def update_performance(self, prediction: float, actual: float, reward: float = 0.0):
        """Update model performance tracking"""
        error = abs(prediction - actual) / max(abs(actual), 1e-6)  # MAPE
        self.prediction_history.append({
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'reward': reward,
            'timestamp': datetime.datetime.now() if not HAS_PANDAS else pd.Timestamp.now()
        })

        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]

    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.prediction_history:
            return {'mape': 0.0, 'total_predictions': 0, 'avg_reward': 0.0}

        recent_predictions = self.prediction_history[-100:]  # Last 100 predictions

        mape = np.mean([p['error'] for p in recent_predictions])
        avg_reward = np.mean([p['reward'] for p in recent_predictions])
        total_predictions = len(recent_predictions)

        return {
            'mape': mape,
            'total_predictions': total_predictions,
            'avg_reward': avg_reward,
            'recent_accuracy': 1 - mape
        }


def create_advanced_ensemble(input_size: int = 24) -> AdvancedEnsemble:
    """Factory function to create the advanced ensemble model"""
    model = AdvancedEnsemble(input_size=input_size)
    logger.info(f"Created Advanced Ensemble with input_size={input_size}")
    return model


def load_advanced_ensemble(model_path: str, device: str = 'cpu') -> AdvancedEnsemble:
    """Load trained advanced ensemble model"""
    model = create_advanced_ensemble()
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load scaler if available
        if 'scaler' in checkpoint:
            model.feature_scaler = checkpoint['scaler']

        model.to(device)
        model.eval()
        logger.info(f"Loaded advanced ensemble from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def save_advanced_ensemble(model: AdvancedEnsemble, model_path: str, scaler=None):
    """Save trained advanced ensemble model"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'scaler': scaler or model.feature_scaler
    }
    torch.save(checkpoint, model_path)
    logger.info(f"Saved advanced ensemble to {model_path}")


# Market regime detection for adaptive strategies
class MarketRegimeDetector:
    """Detects market regime (bull, bear, volatile, ranging) for adaptive prediction"""

    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.price_history = []
        self.regime_history = []

    def detect_regime(self, current_price: float, recent_prices: List[float]) -> str:
        """
        Detect current market regime
        Returns: 'bull', 'bear', 'volatile', 'ranging'
        """
        if len(recent_prices) < self.lookback_period:
            return 'unknown'

        # Calculate returns
        returns = np.diff(recent_prices) / recent_prices[:-1]
        returns = returns[-self.lookback_period:]

        # Trend detection (momentum)
        trend_strength = np.mean(returns)

        # Volatility detection
        volatility = np.std(returns)

        # Determine regime
        if trend_strength > 0.001 and volatility < 0.02:
            regime = 'bull'
        elif trend_strength < -0.001 and volatility < 0.02:
            regime = 'bear'
        elif volatility > 0.03:
            regime = 'volatile'
        else:
            regime = 'ranging'

        self.regime_history.append(regime)
        return regime

    def get_regime_stats(self) -> Dict[str, float]:
        """Get regime detection statistics"""
        if not self.regime_history:
            return {}

        regimes = self.regime_history[-100:] if not HAS_PANDAS else pd.Series(self.regime_history[-100:])  # Last 100 detections
        stats = regimes.value_counts(normalize=True).to_dict()
        return stats


if __name__ == "__main__":
    # Test the advanced ensemble
    print("🧠 Testing Advanced Ensemble Model")
    print("=" * 50)

    # Create model
    model = create_advanced_ensemble(input_size=24)

    # Test forward pass
    batch_size, seq_len, input_size = 4, 60, 24
    x = torch.randn(batch_size, seq_len, input_size)

    predictions, uncertainties = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Uncertainties shape: {uncertainties.shape}")

    # Test confidence filtering
    conf_predictions, conf_uncertainties, should_predict = model.predict_with_confidence(x, confidence_threshold=0.8)
    print(f"Should predict: {should_predict.sum().item()}/{batch_size} samples")

    # Test market regime detector
    regime_detector = MarketRegimeDetector()
    test_prices = np.random.randn(150) * 0.01 + 100  # Simulated price series
    regime = regime_detector.detect_regime(test_prices[-1], test_prices.tolist())
    print(f"Detected market regime: {regime}")

    print("✅ Advanced Ensemble Model Ready for Training!")
