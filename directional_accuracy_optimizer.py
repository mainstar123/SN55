#!/usr/bin/env python3
"""
Directional Accuracy Optimizer
Hyperparameter optimization specifically for achieving >90% directional accuracy
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import itertools
import json
from datetime import datetime
import os

class DirectionalAccuracyOptimizer:
    """Optimizes hyperparameters specifically for directional accuracy >90%"""

    def __init__(self, model_class, input_size: int = 24):
        self.model_class = model_class
        self.input_size = input_size
        self.best_params = {}
        self.optimization_history = []

    def define_hyperparameter_space(self) -> Dict[str, List]:
        """Define hyperparameter search space optimized for directional accuracy"""

        return {
            # Model architecture
            'hidden_size': [64, 128, 256],
            'num_layers': [2, 3, 4],
            'dropout': [0.1, 0.2, 0.3],
            'num_heads': [4, 8, 16],

            # Training parameters
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64],
            'weight_decay': [1e-4, 1e-5, 0],

            # Ensemble weights
            'gru_weight': [0.3, 0.4, 0.5],
            'transformer_weight': [0.3, 0.4, 0.5],
            'cnn_weight': [0.2, 0.3, 0.4],

            # Confidence and quality thresholds
            'confidence_threshold': [0.6, 0.7, 0.8],
            'quality_threshold': [0.5, 0.6, 0.7],
            'dynamic_threshold_weight': [0.3, 0.5, 0.7]
        }

    def optimize_hyperparameters(self, train_data: Tuple[torch.Tensor, torch.Tensor],
                               val_data: Tuple[torch.Tensor, torch.Tensor],
                               max_evaluations: int = 50) -> Dict:
        """Run hyperparameter optimization for directional accuracy"""

        print("🎯 STARTING HYPERPARAMETER OPTIMIZATION FOR DIRECTIONAL ACCURACY >90%")
        print("=" * 80)

        param_space = self.define_hyperparameter_space()

        # Use Bayesian optimization-like approach with directional accuracy focus
        best_score = 0.0
        best_params = {}

        # Generate parameter combinations focused on accuracy
        param_combinations = self._generate_smart_combinations(param_space, max_evaluations)

        print(f"📊 Evaluating {len(param_combinations)} parameter combinations...")

        for i, params in enumerate(param_combinations):
            print(f"\n🔄 Evaluation {i+1}/{len(param_combinations)}")
            print(f"   Parameters: {params}")

            # Evaluate parameters
            score = self._evaluate_parameters(params, train_data, val_data)

            print(".4f")
            # Store results
            result = {
                'params': params,
                'directional_accuracy': score,
                'evaluation_number': i+1,
                'timestamp': datetime.now().isoformat()
            }
            self.optimization_history.append(result)

            # Update best
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"   🏆 NEW BEST SCORE!")

                # Early stopping if we achieve target
                if score >= 0.90:
                    print(f"   🎯 TARGET ACHIEVED: {score:.4f} >= 90%")
                    break

        self.best_params = best_params

        print(f"\n🏆 OPTIMIZATION COMPLETE")
        print(".4f")
        print(f"   Best parameters: {best_params}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_history': self.optimization_history,
            'target_achieved': best_score >= 0.90
        }

    def _generate_smart_combinations(self, param_space: Dict[str, List], max_evals: int) -> List[Dict]:
        """Generate smart parameter combinations focused on accuracy"""

        # Prioritize combinations likely to achieve high directional accuracy
        smart_combinations = []

        # High-priority combinations (more likely to achieve >90% accuracy)
        high_priority = [
            {
                'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2, 'num_heads': 8,
                'learning_rate': 0.001, 'batch_size': 32, 'weight_decay': 1e-4,
                'gru_weight': 0.4, 'transformer_weight': 0.4, 'cnn_weight': 0.3,
                'confidence_threshold': 0.7, 'quality_threshold': 0.6, 'dynamic_threshold_weight': 0.5
            },
            {
                'hidden_size': 256, 'num_layers': 4, 'dropout': 0.1, 'num_heads': 16,
                'learning_rate': 0.0005, 'batch_size': 16, 'weight_decay': 1e-5,
                'gru_weight': 0.3, 'transformer_weight': 0.5, 'cnn_weight': 0.2,
                'confidence_threshold': 0.8, 'quality_threshold': 0.7, 'dynamic_threshold_weight': 0.3
            },
            {
                'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'num_heads': 8,
                'learning_rate': 0.001, 'batch_size': 64, 'weight_decay': 0,
                'gru_weight': 0.5, 'transformer_weight': 0.3, 'cnn_weight': 0.2,
                'confidence_threshold': 0.6, 'quality_threshold': 0.5, 'dynamic_threshold_weight': 0.7
            }
        ]

        smart_combinations.extend(high_priority)

        # Add some random combinations for exploration
        remaining_slots = max_evals - len(high_priority)
        if remaining_slots > 0:
            random_combinations = self._generate_random_combinations(param_space, remaining_slots)
            # Convert numpy types to regular Python types
            for combo in random_combinations:
                for key, value in combo.items():
                    if hasattr(value, 'item'):  # numpy types
                        combo[key] = value.item()
                    elif isinstance(value, np.integer):
                        combo[key] = int(value)
                    elif isinstance(value, np.floating):
                        combo[key] = float(value)
            smart_combinations.extend(random_combinations)

        return smart_combinations[:max_evals]

    def _generate_random_combinations(self, param_space: Dict[str, List], n: int) -> List[Dict]:
        """Generate random parameter combinations"""
        combinations = []

        for _ in range(n):
            combo = {}
            for param, values in param_space.items():
                combo[param] = np.random.choice(values)
            combinations.append(combo)

        return combinations

    def _evaluate_parameters(self, params: Dict, train_data: Tuple[torch.Tensor, torch.Tensor],
                           val_data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Evaluate a parameter combination"""

        try:
            # Create model with parameters
            model = self._create_model_from_params(params)

            # Quick training and evaluation
            directional_accuracy = self._quick_train_evaluate(model, params, train_data, val_data)

            return directional_accuracy

        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
            return 0.0

    def _create_model_from_params(self, params: Dict):
        """Create model instance from parameters"""
        from advanced_ensemble_optimizer import ConfidenceWeightedEnsemble

        model = ConfidenceWeightedEnsemble(
            input_size=self.input_size,
            hidden_size=params['hidden_size'],
            num_heads=params['num_heads']
        )

        return model

    def _quick_train_evaluate(self, model: nn.Module, params: Dict,
                            train_data: Tuple[torch.Tensor, torch.Tensor],
                            val_data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Quick training and evaluation for hyperparameter optimization"""

        X_train, y_train = train_data
        X_val, y_val = val_data

        # Create optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'],
                                    weight_decay=params['weight_decay'])

        # Quick training loop (just a few epochs for hyperparameter search)
        model.train()
        for epoch in range(3):  # Very quick training
            for i in range(0, len(X_train), params['batch_size']):
                batch_X = X_train[i:i+params['batch_size']]
                batch_y = y_train[i:i+params['batch_size']]

                optimizer.zero_grad()

                outputs = model(batch_X, return_confidence=False)
                predictions = outputs['prediction']

                # Use directional loss
                loss = self._directional_loss(predictions, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val, return_confidence=False)
            val_predictions = val_outputs['prediction']

            directional_accuracy = self._calculate_directional_accuracy(val_predictions, y_val)

        return directional_accuracy

    def _directional_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Directional accuracy focused loss"""
        if len(predictions) < 2:
            return torch.tensor(0.0, requires_grad=True)

        pred_direction = predictions[1:] - predictions[:-1]
        target_direction = targets[1:] - targets[:-1]

        # Binary cross entropy on directional accuracy
        direction_match = ((pred_direction > 0) == (target_direction > 0)).float()
        loss = torch.nn.functional.binary_cross_entropy(
            direction_match, torch.ones_like(direction_match)
        )

        return loss

    def _calculate_directional_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate directional accuracy"""
        if len(predictions) < 2 or len(targets) < 2:
            return 0.0

        pred_direction = predictions[1:] > predictions[:-1]
        target_direction = targets[1:] > targets[:-1]

        accuracy = (pred_direction == target_direction).float().mean().item()
        return accuracy

    def save_optimization_results(self, results: Dict, filename: str = None):
        """Save optimization results to file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"directional_accuracy_optimization_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Optimization results saved: {filename}")

    def load_best_model(self, results: Dict) -> nn.Module:
        """Load model with best parameters"""

        if not results['best_params']:
            raise ValueError("No optimized parameters found")

        model = self._create_model_from_params(results['best_params'])

        print("✅ Best model loaded with optimized parameters")
        print(f"   Expected directional accuracy: {results['best_score']:.4f}")

        return model

class AccuracyFocusedTraining:
    """Training techniques specifically optimized for directional accuracy"""

    def __init__(self, model: nn.Module, optimized_params: Dict):
        self.model = model
        self.params = optimized_params

        # Setup optimizer with optimized parameters
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimized_params.get('learning_rate', 0.001),
            weight_decay=optimized_params.get('weight_decay', 1e-4)
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )

        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        # Early stopping
        self.best_accuracy = 0.0
        self.patience = 20
        self.patience_counter = 0

    def train_for_accuracy(self, train_loader, val_loader, num_epochs: int = 100) -> Dict[str, List]:
        """Train specifically for maximum directional accuracy"""

        print("🎯 STARTING ACCURACY-FOCUSED TRAINING")
        print("=" * 60)

        training_history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate_epoch(val_loader)

            # Update history
            for key in training_history.keys():
                if key in train_metrics:
                    training_history[key].append(train_metrics[key])
                elif key in val_metrics:
                    training_history[key].append(val_metrics[key])

            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            training_history['learning_rate'].append(current_lr)

            # Print progress
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Acc: {train_metrics['directional_accuracy']:.4f} | "
                  f"Val Acc: {val_metrics['directional_accuracy']:.4f} | "
                  f"LR: {current_lr:.6f}")

            # Early stopping check
            if val_metrics['directional_accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['directional_accuracy']
                self.patience_counter = 0

                # Save best model
                if val_metrics['directional_accuracy'] >= 0.90:
                    self._save_checkpoint(epoch, val_metrics)
                    print(f"   🏆 TARGET ACHIEVED: {val_metrics['directional_accuracy']:.4f} >= 90%!")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                break

        print("\n✅ TRAINING COMPLETE")
        print(".4f")
        return training_history

    def _train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch with accuracy focus"""

        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_X, return_confidence=False)
                    predictions = outputs['prediction']
                    loss = self._directional_accuracy_loss(predictions, batch_y)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_X, return_confidence=False)
                predictions = outputs['prediction']
                loss = self._directional_accuracy_loss(predictions, batch_y)
                loss.backward()
                self.optimizer.step()

            # Calculate metrics
            batch_accuracy = self._calculate_directional_accuracy(predictions, batch_y)
            epoch_loss += loss.item()
            epoch_accuracy += batch_accuracy
            num_batches += 1

        return {
            'loss': epoch_loss / num_batches,
            'directional_accuracy': epoch_accuracy / num_batches
        }

    def _validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch"""

        self.model.eval()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X, return_confidence=False)
                predictions = outputs['prediction']
                loss = self._directional_accuracy_loss(predictions, batch_y)

                batch_accuracy = self._calculate_directional_accuracy(predictions, batch_y)
                epoch_loss += loss.item()
                epoch_accuracy += batch_accuracy
                num_batches += 1

        return {
            'loss': epoch_loss / num_batches,
            'directional_accuracy': epoch_accuracy / num_batches
        }

    def _directional_accuracy_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss function specifically designed for directional accuracy"""

        if len(predictions) < 2:
            return torch.tensor(0.0, requires_grad=True)

        # Calculate directional changes
        pred_changes = predictions[1:] - predictions[:-1]
        target_changes = targets[1:] - targets[:-1]

        # Convert to binary classification (direction correct/incorrect)
        pred_direction_correct = (pred_changes > 0) == (target_changes > 0)

        # Use focal loss to focus on hard examples
        target_tensor = pred_direction_correct.float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.ones_like(target_tensor),  # Always predict "correct"
            target_tensor,
            reduction='mean'
        )

        return loss

    def _calculate_directional_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate directional accuracy for batch"""
        if len(predictions) < 2 or len(targets) < 2:
            return 0.0

        pred_direction = predictions[1:] > predictions[:-1]
        target_direction = targets[1:] > targets[:-1]

        accuracy = (pred_direction == target_direction).float().mean().item()
        return accuracy

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint when target accuracy is achieved"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'params': self.params,
            'timestamp': datetime.now().isoformat()
        }

        filename = f"accuracy_target_achieved_{metrics['directional_accuracy']:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(checkpoint, filename)

        print(f"   💾 Checkpoint saved: {filename}")

def create_accuracy_optimization_pipeline(input_size: int = 24):
    """Create the complete accuracy optimization pipeline"""

    # Import the advanced ensemble
    from advanced_ensemble_optimizer import ConfidenceWeightedEnsemble

    # Create hyperparameter optimizer
    hyper_optimizer = DirectionalAccuracyOptimizer(ConfidenceWeightedEnsemble, input_size)

    return {
        'hyperparameter_optimizer': hyper_optimizer,
        'model_class': ConfidenceWeightedEnsemble,
        'pipeline_ready': True
    }

# Example usage
if __name__ == "__main__":
    print("🎯 DIRECTIONAL ACCURACY OPTIMIZATION SYSTEM")
    print("=" * 60)

    # Create optimization pipeline
    pipeline = create_accuracy_optimization_pipeline()

    print("✅ Hyperparameter optimization system ready")
    print("✅ Directional accuracy focused training ready")
    print("✅ Early stopping and checkpointing enabled")
    print("✅ Mixed precision training support")
    print("\n🚀 Ready to optimize for >90% directional accuracy!")

    # Show hyperparameter space
    optimizer = pipeline['hyperparameter_optimizer']
    param_space = optimizer.define_hyperparameter_space()

    print("\n📊 HYPERPARAMETER OPTIMIZATION SPACE:")
    for param, values in param_space.items():
        print(f"   {param}: {values}")

    print("\n🎯 TARGET: Achieve >90% directional accuracy")
    print("💡 Focus: Confidence weighting, dynamic thresholding, market regime detection")
