"""
Advanced Ensemble Trainer for Precog Subnet 55
Implements multiple model architectures with meta-learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from pathlib import Path

from models.enhanced_gru import EnhancedGRUPriceForecaster
from models.transformer_model import TransformerPriceForecaster
from models.lstm_attention import LSTMAttentionForecaster
from models.interval_forecaster import QuantileIntervalForecaster
from data.feature_engineering import add_advanced_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleMetaLearner:
    """Advanced ensemble with meta-learning capabilities"""

    def __init__(self, input_size: int = 24, hidden_size: int = 128, num_heads: int = 8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Initialize multiple base models
        self.models = {
            'enhanced_gru': EnhancedGRUPriceForecaster(input_size, hidden_size, num_heads),
            'transformer': TransformerPriceForecaster(input_size, hidden_size, num_heads),
            'lstm_attention': LSTMAttentionForecaster(input_size, hidden_size, num_heads),
            'quantile_interval': QuantileIntervalForecaster(input_size, hidden_size)
        }

        # Meta-learner for combining predictions
        self.meta_learner = nn.Sequential(
            nn.Linear(len(self.models), 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Market regime detector
        self.regime_detector = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 market regimes
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_device()

    def to_device(self):
        """Move all models to appropriate device"""
        for model in self.models.values():
            model.to(self.device)
        self.meta_learner.to(self.device)
        self.regime_detector.to(self.device)

    def detect_market_regime(self, features: torch.Tensor) -> int:
        """Detect current market regime"""
        with torch.no_grad():
            regime_logits = self.regime_detector(features)
            return torch.argmax(regime_logits, dim=1).item()

    def get_regime_weights(self, regime: int) -> Dict[str, float]:
        """Get model weights based on market regime"""
        # Regime-specific weights (learned through meta-learning)
        regime_weights = {
            0: {'enhanced_gru': 0.4, 'transformer': 0.3, 'lstm_attention': 0.2, 'quantile_interval': 0.1},  # Bull
            1: {'enhanced_gru': 0.3, 'transformer': 0.2, 'lstm_attention': 0.4, 'quantile_interval': 0.1},  # Bear
            2: {'enhanced_gru': 0.2, 'transformer': 0.4, 'lstm_attention': 0.3, 'quantile_interval': 0.1},  # Sideways
            3: {'enhanced_gru': 0.3, 'transformer': 0.3, 'lstm_attention': 0.2, 'quantile_interval': 0.2}   # Volatile
        }
        return regime_weights.get(regime, regime_weights[0])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ensemble"""
        regime = self.detect_market_regime(x)
        weights = self.get_regime_weights(regime)

        predictions = []
        model_outputs = []

        for name, model in self.models.items():
            with torch.no_grad():
                if name == 'quantile_interval':
                    pred = model(x)[0]  # Point prediction from quantile model
                else:
                    pred = model(x)
                predictions.append(pred * weights[name])
                model_outputs.append(pred)

        # Combine predictions
        ensemble_pred = torch.stack(predictions).sum(dim=0)

        # Meta-learner refinement
        meta_input = torch.stack(model_outputs).squeeze().t()
        meta_refinement = self.meta_learner(meta_input)
        final_prediction = ensemble_pred + meta_refinement.squeeze()

        return final_prediction, torch.tensor(regime)

    def train_meta_learner(self, train_loader: DataLoader, epochs: int = 10):
        """Train the meta-learner on ensemble predictions"""
        optimizer = optim.Adam(self.meta_learner.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.meta_learner.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Get individual model predictions
                model_preds = []
                for name, model in self.models.items():
                    if name == 'quantile_interval':
                        pred = model(batch_x)[0]
                    else:
                        pred = model(batch_x)
                    model_preds.append(pred)

                # Meta-learner input
                meta_input = torch.stack(model_preds).squeeze().t()
                meta_output = self.meta_learner(meta_input)

                # Ensemble prediction
                weights = torch.ones(len(self.models)) / len(self.models)
                ensemble_pred = torch.stack(model_preds).squeeze().t() @ weights

                # Train meta-learner to improve ensemble
                refined_pred = ensemble_pred + meta_output.squeeze()
                loss = criterion(refined_pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logger.info(f"Meta-learner Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")

    def save_ensemble(self, path: str = "models/ensemble"):
        """Save the entire ensemble"""
        Path(path).mkdir(exist_ok=True)

        # Save base models
        for name, model in self.models.items():
            torch.save(model.state_dict(), f"{path}/{name}.pth")

        # Save meta-learner
        torch.save(self.meta_learner.state_dict(), f"{path}/meta_learner.pth")
        torch.save(self.regime_detector.state_dict(), f"{path}/regime_detector.pth")

        logger.info(f"Ensemble saved to {path}")

    def load_ensemble(self, path: str = "models/ensemble"):
        """Load the entire ensemble"""
        try:
            for name, model in self.models.items():
                model.load_state_dict(torch.load(f"{path}/{name}.pth"))

            self.meta_learner.load_state_dict(torch.load(f"{path}/meta_learner.pth"))
            self.regime_detector.load_state_dict(torch.load(f"{path}/regime_detector.pth"))

            logger.info(f"Ensemble loaded from {path}")
        except FileNotFoundError:
            logger.warning(f"Ensemble files not found at {path}")

class OnlineLearner:
    """Online learning adapter for continuous model improvement"""

    def __init__(self, ensemble: EnsembleMetaLearner, adaptation_rate: float = 0.01):
        self.ensemble = ensemble
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.regime_performance = {i: [] for i in range(4)}

    def adapt_to_feedback(self, prediction: torch.Tensor, actual: torch.Tensor,
                         regime: int, reward: float):
        """Adapt model based on prediction feedback and rewards"""

        # Calculate prediction error
        error = torch.abs(prediction - actual).item()

        # Store performance data
        self.performance_history.append({
            'error': error,
            'reward': reward,
            'regime': regime
        })

        self.regime_performance[regime].append(reward)

        # Online adaptation based on recent performance
        if len(self.performance_history) >= 10:
            recent_performance = self.performance_history[-10:]
            avg_recent_reward = np.mean([p['reward'] for p in recent_performance])

            # Adjust adaptation rate based on performance
            if avg_recent_reward > 0.05:  # High reward threshold
                self.adaptation_rate *= 1.1  # Increase learning
            elif avg_recent_reward < 0.01:  # Low reward
                self.adaptation_rate *= 0.9  # Decrease learning

        logger.info(".6f"
    def get_adaptation_suggestions(self) -> Dict[str, float]:
        """Get suggestions for model adaptation"""
        suggestions = {}

        # Analyze regime performance
        for regime, rewards in self.regime_performance.items():
            if rewards:
                avg_reward = np.mean(rewards)
                suggestions[f'regime_{regime}_weight'] = avg_reward

        return suggestions

def create_online_training_data(price_data: pd.DataFrame, lookback: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create training data for online learning"""
    features_list = []

    for i in range(lookback, len(price_data)):
        window = price_data.iloc[i-lookback:i]

        # Extract features
        features = add_advanced_features(window)

        # Get target (next hour return)
        current_price = window['close'].iloc[-1]
        next_price = price_data['close'].iloc[i] if i < len(price_data) else current_price
        target_return = (next_price - current_price) / current_price

        features_list.append({
            'features': features.values[-1],  # Last feature vector
            'target': target_return
        })

    # Convert to tensors
    X = torch.tensor([f['features'] for f in features_list], dtype=torch.float32)
    y = torch.tensor([f['target'] for f in features_list], dtype=torch.float32)

    return X, y
