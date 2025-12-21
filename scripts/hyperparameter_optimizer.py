"""
Advanced Hyperparameter Optimization for Precog Subnet 55
Uses Bayesian optimization and live performance feedback
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from models.enhanced_gru import EnhancedGRUPriceForecaster
from data.feature_engineering import add_advanced_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianHyperparameterOptimizer:
    """Bayesian optimization for model hyperparameters"""

    def __init__(self, input_size: int = 24, max_trials: int = 100):
        self.input_size = input_size
        self.max_trials = max_trials
        self.best_params = {}
        self.optimization_history = []

        # Create study
        self.study = optuna.create_study(
            direction="minimize",  # Minimize validation loss
            sampler=TPESampler(),
            pruner=MedianPruner()
        )

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""

        # Define hyperparameter search space
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 12, 16])
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        # Initialize model with suggested parameters
        model = EnhancedGRUPriceForecaster(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Simulate training with current data
        # In real implementation, this would use actual training data
        train_losses = []
        val_losses = []

        # Mock training loop (replace with real training)
        for epoch in range(10):  # Short training for optimization
            # Generate mock training data
            batch_size_actual = min(batch_size, 64)  # Limit for mock data

            # Mock loss values (would be computed from real training)
            train_loss = np.random.uniform(0.001, 0.01)
            val_loss = train_loss * (0.8 + np.random.random() * 0.4)  # Add some noise

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            # Report intermediate results
            trial.report(val_loss, epoch)

            # Prune if needed
            if trial.should_prune():
                raise optuna.TrialPruned()

        final_val_loss = np.mean(val_losses[-3:])  # Average of last 3 epochs

        # Store optimization result
        self.optimization_history.append({
            'trial': trial.number,
            'params': trial.params,
            'val_loss': final_val_loss,
            'timestamp': datetime.now().isoformat()
        })

        return final_val_loss

    def optimize(self, timeout: int = 3600) -> Dict:
        """Run hyperparameter optimization"""
        logger.info("🎯 Starting Bayesian hyperparameter optimization...")

        self.study.optimize(self.objective, n_trials=self.max_trials, timeout=timeout)

        # Get best parameters
        best_trial = self.study.best_trial
        self.best_params = best_trial.params

        logger.info("✅ Optimization completed!")
        logger.info(f"🏆 Best trial: {best_trial.number}")
        logger.info(f"🎯 Best validation loss: {best_trial.value:.6f}")
        logger.info("📊 Best hyperparameters:"        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")

        return self.best_params

    def save_optimization_results(self, path: str = "optimization_results"):
        """Save optimization results"""
        Path(path).mkdir(exist_ok=True)

        results = {
            'best_params': self.best_params,
            'optimization_history': self.optimization_history,
            'study_stats': {
                'n_trials': len(self.study.trials),
                'best_value': self.study.best_value,
                'best_trial': self.study.best_trial.number
            }
        }

        with open(f"{path}/hyperparameter_optimization.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Optimization results saved to {path}")

class MarketRegimeDetector:
    """Detects and adapts to different market regimes"""

    def __init__(self, lookback_periods: int = 24):
        self.lookback_periods = lookback_periods
        self.regime_history = []

        # Regime classification thresholds
        self.regime_thresholds = {
            'bull': {'returns': 0.02, 'volatility': 0.03},
            'bear': {'returns': -0.02, 'volatility': 0.03},
            'volatile': {'returns': 0.00, 'volatility': 0.05},
            'ranging': {'returns': 0.00, 'volatility': 0.02}
        }

    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """Detect current market regime"""

        if len(price_data) < self.lookback_periods:
            return 'unknown'

        # Calculate returns and volatility
        recent_data = price_data.tail(self.lookback_periods)
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std()
        avg_return = returns.mean()

        # Classify regime
        if avg_return > self.regime_thresholds['bull']['returns'] and volatility < self.regime_thresholds['bull']['volatility']:
            regime = 'bull'
        elif avg_return < self.regime_thresholds['bear']['returns']:
            regime = 'bear'
        elif volatility > self.regime_thresholds['volatile']['volatility']:
            regime = 'volatile'
        else:
            regime = 'ranging'

        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'avg_return': avg_return,
            'volatility': volatility
        })

        return regime

    def get_regime_model_params(self, regime: str) -> Dict:
        """Get optimal model parameters for each regime"""

        regime_params = {
            'bull': {
                'model_type': 'transformer',  # Good at capturing trends
                'learning_rate': 0.001,
                'attention_focus': 'momentum_indicators',
                'prediction_horizon': 1  # Short-term focus
            },
            'bear': {
                'model_type': 'lstm_attention',  # Good at pattern recognition
                'learning_rate': 0.0005,
                'attention_focus': 'support_levels',
                'prediction_horizon': 2  # Medium-term focus
            },
            'volatile': {
                'model_type': 'enhanced_gru',  # Fast adaptation
                'learning_rate': 0.002,
                'attention_focus': 'volatility_indicators',
                'prediction_horizon': 1  # Very short-term
            },
            'ranging': {
                'model_type': 'ensemble',  # Balanced approach
                'learning_rate': 0.0008,
                'attention_focus': 'mean_reversion',
                'prediction_horizon': 3  # Longer-term focus
            }
        }

        return regime_params.get(regime, regime_params['ranging'])

class LiveHyperparameterTuner:
    """Tunes hyperparameters based on live performance"""

    def __init__(self, base_model: EnhancedGRUPriceForecaster):
        self.base_model = base_model
        self.performance_history = []
        self.current_params = {}
        self.tuning_schedule = {
            'daily': ['learning_rate', 'dropout'],
            'weekly': ['hidden_size', 'num_heads'],
            'monthly': ['num_layers', 'attention_mechanism']
        }

    def tune_based_on_performance(self, recent_rewards: List[float],
                                recent_accuracy: List[float]) -> Dict[str, float]:
        """Tune hyperparameters based on recent performance"""

        # Analyze performance trends
        reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        accuracy_trend = np.polyfit(range(len(recent_accuracy)), recent_accuracy, 1)[0]

        tuning_suggestions = {}

        # Adjust learning rate based on performance
        if reward_trend < 0 and accuracy_trend < 0:
            tuning_suggestions['learning_rate'] = self.current_params.get('learning_rate', 0.001) * 0.8
            tuning_suggestions['dropout'] = min(self.current_params.get('dropout', 0.1) + 0.05, 0.3)

        elif reward_trend > 0.001:  # Improving
            tuning_suggestions['learning_rate'] = self.current_params.get('learning_rate', 0.001) * 1.1
            tuning_suggestions['dropout'] = max(self.current_params.get('dropout', 0.1) - 0.02, 0.05)

        # Adjust model capacity
        avg_reward = np.mean(recent_rewards)
        if avg_reward > 0.05:  # High performance
            tuning_suggestions['hidden_size'] = min(self.current_params.get('hidden_size', 128) + 32, 512)
        elif avg_reward < 0.01:  # Low performance
            tuning_suggestions['hidden_size'] = max(self.current_params.get('hidden_size', 128) - 16, 64)

        self.current_params.update(tuning_suggestions)
        return tuning_suggestions

class PerformancePredictor:
    """Predicts future performance based on current hyperparameters"""

    def __init__(self):
        self.performance_model = None
        self.hyperparameter_history = []
        self.performance_history = []

    def train_performance_predictor(self, hyperparams_history: List[Dict],
                                   performance_history: List[float]):
        """Train a model to predict performance from hyperparameters"""

        # Simple linear model for demonstration
        # In practice, use more sophisticated ML model
        self.hyperparameter_history = hyperparams_history
        self.performance_history = performance_history

    def predict_performance(self, new_hyperparams: Dict) -> float:
        """Predict expected performance for new hyperparameters"""

        if not self.performance_history:
            return 0.022  # Default average

        # Simple similarity-based prediction
        similarities = []
        for i, params in enumerate(self.hyperparameter_history):
            similarity = self.calculate_similarity(params, new_hyperparams)
            similarities.append((similarity, self.performance_history[i]))

        # Weight predictions by similarity
        similarities.sort(reverse=True)
        weights = np.array([s[0] for s in similarities[:5]])
        performances = np.array([s[1] for s in similarities[:5]])

        weights = weights / weights.sum()
        predicted_performance = np.sum(weights * performances)

        return predicted_performance

    def calculate_similarity(self, params1: Dict, params2: Dict) -> float:
        """Calculate similarity between two hyperparameter sets"""
        similarity = 0
        count = 0

        for key in set(params1.keys()) & set(params2.keys()):
            if isinstance(params1[key], (int, float)) and isinstance(params2[key], (int, float)):
                # Normalized difference
                diff = abs(params1[key] - params2[key])
                max_val = max(abs(params1[key]), abs(params2[key]), 1e-6)
                similarity += 1 - (diff / max_val)
                count += 1

        return similarity / count if count > 0 else 0

def run_hyperparameter_optimization():
    """Run complete hyperparameter optimization pipeline"""

    logger.info("🚀 Starting comprehensive hyperparameter optimization...")

    # 1. Bayesian optimization
    optimizer = BayesianHyperparameterOptimizer()
    best_params = optimizer.optimize(timeout=1800)  # 30 minutes

    # 2. Save results
    optimizer.save_optimization_results()

    # 3. Market regime detection
    regime_detector = MarketRegimeDetector()

    # 4. Live tuner setup
    # Note: Would need actual model instance for live tuning
    # live_tuner = LiveHyperparameterTuner(model_instance)

    logger.info("✅ Hyperparameter optimization pipeline completed!")
    logger.info(f"🎯 Best parameters found: {best_params}")

    return best_params

if __name__ == "__main__":
    run_hyperparameter_optimization()
