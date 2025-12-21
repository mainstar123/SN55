"""
Automated Hyperparameter Optimization for Precog #1 Miner
Uses Bayesian optimization and evolutionary algorithms for optimal model tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import optuna
from optuna import Trial
import logging
from typing import Dict, List, Tuple, Optional, Callable
import time
import json
import os

from advanced_ensemble_model import create_advanced_ensemble, AdvancedEnsemble

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization using Bayesian optimization
    Targets: MAPE minimization, GPU efficiency, inference speed
    """

    def __init__(self, train_data: Tuple[np.ndarray, np.ndarray],
                 val_data: Tuple[np.ndarray, np.ndarray],
                 device: str = 'auto',
                 study_name: str = 'precog_ensemble_optimization'):
        self.train_x, self.train_y = train_data
        self.val_x, self.val_y = val_data
        self.device = self._setup_device(device)
        self.study_name = study_name

        # Optimization history
        self.best_params = None
        self.best_score = float('inf')
        self.optimization_history = []

        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            direction='minimize',  # Minimize MAPE
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )

    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = 'cpu'

        logger.info(f"Using device: {device}")
        return device

    def objective(self, trial: Trial) -> float:
        """Objective function for optimization"""
        # Sample hyperparameters
        params = {
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
            'num_gru_layers': trial.suggest_int('num_gru_layers', 2, 4),
            'num_transformer_layers': trial.suggest_int('num_transformer_layers', 2, 6),
            'num_lstm_layers': trial.suggest_int('num_lstm_layers', 2, 4),
            'dropout': trial.suggest_float('dropout', 0.05, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'confidence_threshold': trial.suggest_float('confidence_threshold', 0.5, 0.9),
        }

        try:
            # Create and train model
            score = self._evaluate_params(params, trial)

            # Track best parameters
            if score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()

            # Log progress
            logger.info(".4f"
            return score

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float('inf')  # Penalize failed trials

    def _evaluate_params(self, params: Dict, trial: Trial) -> float:
        """Evaluate a set of hyperparameters"""
        # Create model with sampled parameters
        model = AdvancedEnsemble(
            input_size=self.train_x.shape[-1],
            hidden_size=params['hidden_size'],
            dropout=params['dropout']
        )

        # Override model parameters
        model.gru_model.num_layers = params['num_gru_layers']
        model.transformer_model.transformer_encoder.num_layers = params['num_transformer_layers']
        model.lstm_model.num_layers = params['num_lstm_layers']

        model.to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(self.train_x),
            torch.FloatTensor(self.train_y)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.val_x),
            torch.FloatTensor(self.val_y)
        )

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        # Quick training (5 epochs for optimization)
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0

        for epoch in range(5):
            # Training
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                predictions, _ = model(batch_x)
                loss = criterion(predictions.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_actuals = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    predictions, _ = model(batch_x)
                    loss = criterion(predictions.squeeze(), batch_y)
                    val_loss += loss.item()

                    val_predictions.extend(predictions.squeeze().cpu().numpy())
                    val_actuals.extend(batch_y.cpu().numpy())

            val_loss /= len(val_loader)
            train_loss /= len(train_loader)

            # Calculate MAPE
            mape = mean_absolute_percentage_error(val_actuals, val_predictions)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            # Report intermediate results
            trial.report(mape, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return mape

    def optimize(self, n_trials: int = 50, timeout: int = 3600) -> Dict:
        """Run hyperparameter optimization"""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")

        self.study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        # Get best results
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value

        logger.info("
Optimization Complete!"        logger.info(".6f"
        logger.info(f"Best parameters: {best_params}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': self.study,
            'optimization_history': self.optimization_history
        }

    def save_optimization_results(self, save_path: str):
        """Save optimization results"""
        results = {
            'best_params': self.study.best_params,
            'best_score': self.study.best_value,
            'study_name': self.study_name,
            'n_trials': len(self.study.trials),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved optimization results to {save_path}")


class EvolutionaryOptimizer:
    """
    Evolutionary algorithm for ensemble weight optimization
    Optimizes meta-learner weights for different market conditions
    """

    def __init__(self, model: AdvancedEnsemble, population_size: int = 50, generations: int = 100):
        self.model = model
        self.population_size = population_size
        self.generations = generations
        self.device = next(model.parameters()).device

        # Meta-learner weights (3 models: GRU, Transformer, LSTM)
        self.num_weights = 3

    def fitness_function(self, weights: np.ndarray,
                        val_data: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate fitness (negative MAPE) for a set of weights"""
        val_x, val_y = val_data

        # Set meta-learner weights
        with torch.no_grad():
            # Create context features (dummy for now)
            context_features = torch.FloatTensor(val_x[:, -1, :]).to(self.device)

            # Manually compute weighted predictions
            model_outputs = []
            for i in range(val_x.shape[0]):
                batch_x = torch.FloatTensor(val_x[i:i+1]).to(self.device)
                gru_out, _ = self.model.gru_model(batch_x)
                transformer_out, _ = self.model.transformer_model(batch_x)
                lstm_out, _ = self.model.lstm_model(batch_x)
                model_outputs.append(torch.cat([gru_out, transformer_out, lstm_out], dim=1))

            model_outputs = torch.cat(model_outputs, dim=0)
            weights_tensor = torch.FloatTensor(weights).to(self.device).unsqueeze(0)
            predictions = torch.sum(model_outputs * weights_tensor, dim=1, keepdim=True)

            # Calculate MAPE
            predictions_np = predictions.squeeze().cpu().numpy()
            mape = mean_absolute_percentage_error(val_y, predictions_np)

        return -mape  # Negative because we want to maximize fitness

    def evolve_weights(self, val_data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Evolve optimal ensemble weights"""
        logger.info("Starting evolutionary optimization of ensemble weights")

        # Initialize population
        population = np.random.dirichlet(np.ones(self.num_weights), self.population_size)

        best_fitness = float('-inf')
        best_weights = None

        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for weights in population:
                fitness = self.fitness_function(weights, val_data)
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_weights = weights.copy()

            # Selection (tournament selection)
            selected = []
            for _ in range(self.population_size // 2):
                # Tournament of 3
                tournament_indices = np.random.choice(self.population_size, 3, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected.append(population[winner_idx])

            # Crossover
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i+1]

                    # Single point crossover
                    crossover_point = np.random.randint(1, self.num_weights)
                    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

                    offspring.extend([child1, child2])

            # Mutation
            mutated_offspring = []
            for child in offspring:
                if np.random.random() < 0.1:  # 10% mutation rate
                    mutation_idx = np.random.randint(self.num_weights)
                    child[mutation_idx] += np.random.normal(0, 0.1)
                    child = np.clip(child, 0, 1)  # Ensure valid weights
                    child = child / child.sum()  # Renormalize
                mutated_offspring.append(child)

            # Create new population
            population = np.array(selected + mutated_offspring)

            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {-best_fitness:.6f}")

        # Normalize best weights
        best_weights = best_weights / best_weights.sum()

        logger.info(f"Evolution complete! Best weights: {best_weights}")
        return best_weights


def run_full_optimization(train_data: Tuple[np.ndarray, np.ndarray],
                         val_data: Tuple[np.ndarray, np.ndarray],
                         save_path: str = 'optimized_model.pth',
                         n_trials: int = 30) -> Dict:
    """
    Run complete hyperparameter optimization pipeline
    Returns optimized model and parameters
    """
    logger.info("🚀 Starting Full Hyperparameter Optimization Pipeline")
    print("=" * 60)

    # Step 1: Bayesian optimization
    print("📊 Phase 1: Bayesian Hyperparameter Optimization")
    optimizer = HyperparameterOptimizer(train_data, val_data)
    opt_results = optimizer.optimize(n_trials=n_trials)

    best_params = opt_results['best_params']
    print(".6f"    print(f"Best parameters: {best_params}")

    # Step 2: Train final model with best parameters
    print("\n🧠 Phase 2: Training Final Model")
    final_model = AdvancedEnsemble(
        input_size=train_data[0].shape[-1],
        hidden_size=best_params['hidden_size'],
        dropout=best_params['dropout']
    )

    # Override architecture parameters
    final_model.gru_model.num_layers = best_params['num_gru_layers']
    final_model.transformer_model.transformer_encoder.num_layers = best_params['num_transformer_layers']
    final_model.lstm_model.num_layers = best_params['num_lstm_layers']

    # Step 3: Evolutionary weight optimization
    print("🧬 Phase 3: Evolutionary Ensemble Weight Optimization")
    ev_optimizer = EvolutionaryOptimizer(final_model, population_size=30, generations=50)
    optimal_weights = ev_optimizer.evolve_weights(val_data)

    print(f"Optimal ensemble weights: {optimal_weights}")

    # Step 4: Full training with optimal parameters
    print("🎯 Phase 4: Full Training with Optimal Configuration")

    # Training setup
    device = optimizer.device
    final_model.to(device)

    criterion = nn.MSELoss()
    optimizer_adam = optim.AdamW(
        final_model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )

    # Full training
    train_dataset = TensorDataset(
        torch.FloatTensor(train_data[0]),
        torch.FloatTensor(train_data[1])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_data[0]),
        torch.FloatTensor(val_data[1])
    )

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)

    # Train for longer
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(50):  # Full training
        # Training
        final_model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer_adam.zero_grad()
            predictions, _ = final_model(batch_x)
            loss = criterion(predictions.squeeze(), batch_y)
            loss.backward()
            optimizer_adam.step()

            train_loss += loss.item()

        # Validation
        final_model.eval()
        val_loss = 0.0
        val_predictions = []
        val_actuals = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions, _ = final_model(batch_x)
                val_loss += criterion(predictions.squeeze(), batch_y).item()

                val_predictions.extend(predictions.squeeze().cpu().numpy())
                val_actuals.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        mape = mean_absolute_percentage_error(val_actuals, val_predictions)

        print(".6f"
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': final_model.state_dict(),
                'best_params': best_params,
                'optimal_weights': optimal_weights,
                'val_mape': mape,
                'epoch': epoch
            }, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("
✅ Optimization Complete!"    print(f"Model saved to: {save_path}")
    print(".6f"
    return {
        'model': final_model,
        'best_params': best_params,
        'optimal_weights': optimal_weights,
        'final_mape': mape,
        'model_path': save_path
    }


if __name__ == "__main__":
    # Test optimization pipeline
    print("🧪 Testing Hyperparameter Optimization")

    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    seq_len = 60
    n_features = 24

    # Generate synthetic time series data
    X = np.random.randn(n_samples, seq_len, n_features)
    y = np.random.randn(n_samples) * 0.01 + 0.001  # Small price changes

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run optimization
    results = run_full_optimization(
        (X_train, y_train),
        (X_val, y_val),
        save_path='optimized_ensemble_test.pth',
        n_trials=5  # Small number for testing
    )

    print("
🎯 Optimization Results:"    print(f"Final MAPE: {results['final_mape']:.6f}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Optimal weights: {results['optimal_weights']}")
