#!/usr/bin/env python3
"""
Core Supremacy Training
Minimal but effective training to achieve >90% directional accuracy
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import os

class SupremacyModel(nn.Module):
    """Simplified but effective model for directional accuracy supremacy"""

    def __init__(self):
        super().__init__()

        # Simple but effective architecture
        self.layers = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class SupremacyTrainer:
    """Core trainer focused on directional accuracy"""

    def __init__(self, target_accuracy=0.90):
        self.target_accuracy = target_accuracy
        self.model = SupremacyModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.best_accuracy = 0.0

    def generate_data(self):
        """Generate synthetic training data"""
        np.random.seed(42)

        # Create realistic market patterns
        n_samples = 3000
        n_features = 24

        # Generate base features
        X = np.random.randn(n_samples, n_features).astype(np.float32)

        # Add temporal patterns
        t = np.arange(n_samples)
        trend = 0.001 * t
        seasonal = 0.05 * np.sin(2 * np.pi * t / 200)
        noise = 0.02 * np.random.randn(n_samples)

        # Create target with directional predictability
        y = trend + seasonal + noise
        y = (y - y.min()) / (y.max() - y.min())  # Normalize 0-1
        y = y.astype(np.float32)

        # Split data
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)

        X_train = torch.from_numpy(X[:train_size])
        y_train = torch.from_numpy(y[:train_size])
        X_val = torch.from_numpy(X[train_size:train_size+val_size])
        y_val = torch.from_numpy(y[train_size:train_size+val_size])
        X_test = torch.from_numpy(X[train_size+val_size:])
        y_test = torch.from_numpy(y[train_size+val_size:])

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def directional_loss(self, pred, target):
        """Loss optimized for directional accuracy"""
        if len(pred) < 2:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        # Calculate directional changes
        pred_changes = pred[1:] - pred[:-1]
        target_changes = target[1:] - target[:-1]

        # Direction prediction (up/down) - ensure gradients flow
        pred_direction = torch.sigmoid(pred_changes)  # Convert to 0-1 range
        target_direction = ((target_changes > 0).float() + 0.5) / 2.0  # 0.5 for no change, 0/1 for down/up

        # MSE loss on directional prediction
        loss = nn.functional.mse_loss(pred_direction, target_direction)
        return loss

    def calculate_directional_accuracy(self, predictions, targets):
        """Calculate directional accuracy"""
        if len(predictions) < 2 or len(targets) < 2:
            return 0.0

        pred_direction = predictions[1:] > predictions[:-1]
        target_direction = targets[1:] > targets[:-1]

        accuracy = (pred_direction == target_direction).float().mean().item()
        return accuracy

    def train_epoch(self, X, y, batch_size=32):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0

        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            self.optimizer.zero_grad()

            pred = self.model(batch_X).squeeze()

            # Simple MSE loss for now (will optimize for directional accuracy)
            loss = nn.functional.mse_loss(pred, batch_y)

            loss.backward()
            self.optimizer.step()

            # Calculate directional accuracy for monitoring
            pred_np = pred.detach().cpu().numpy()
            target_np = batch_y.cpu().numpy()
            accuracy = self.calculate_directional_accuracy(pred_np, target_np)

            total_loss += loss.item()
            total_accuracy += accuracy
            n_batches += 1

        return total_loss / n_batches, total_accuracy / n_batches

    def validate(self, X, y):
        """Validate model"""
        self.model.eval()

        with torch.no_grad():
            pred = self.model(X).squeeze()
            accuracy = self.calculate_directional_accuracy(pred.numpy(), y.numpy())

        return accuracy

    def train_for_supremacy(self):
        """Complete supremacy training"""
        print("🏆 CORE SUPREMACY TRAINING")
        print("=" * 50)
        print(".1f")
        print("💪 Goal: Secure #1 position")
        print("\n📊 Generating training data...")

        train_data, val_data, test_data = self.generate_data()
        X_train, y_train = train_data
        X_val, y_val = val_data

        print(f"   ✅ Data ready: {len(X_train)} train, {len(X_val)} val")

        print("\n🎯 STARTING TRAINING...")

        patience = 15
        patience_counter = 0

        for epoch in range(100):
            # Train
            train_loss, train_acc = self.train_epoch(X_train, y_train)

            # Validate
            val_acc = self.validate(X_val, y_val)

            print("2d")
            # Check for improvement
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                patience_counter = 0

                # Check if target achieved
                if val_acc >= self.target_accuracy:
                    print(f"\n🎯 TARGET ACHIEVED: {val_acc:.4f} >= {self.target_accuracy:.4f}")
                    self.save_model(val_acc)
                    return True
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break

        # Final evaluation
        print("\n🎯 FINAL EVALUATION")
        X_test, y_test = test_data
        test_acc = self.validate(X_test, y_test)

        print("\n🏆 RESULTS:")
        print(".4f")
        print(".4f")
        if test_acc >= self.target_accuracy:
            print("   🎯 SUPREMACY ACHIEVED!")
            self.save_model(test_acc)
            return True
        else:
            print(".4f")
            return False

    def save_model(self, accuracy):
        """Save the supremacy model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'target_achieved': accuracy >= self.target_accuracy,
            'timestamp': datetime.now().isoformat()
        }

        filename = f"supremacy_model_{accuracy:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(checkpoint, filename)

        # Save as latest
        torch.save(checkpoint, "latest_supremacy_model.pth")

        print(f"   💾 Model saved: {filename}")

def main():
    print("🏆 SUPREMACY TRAINING SYSTEM")
    print("=" * 50)
    print("🎯 Objective: Achieve >90% directional accuracy")
    print("💪 Goal: Secure #1 position on subnet 55")
    print("=" * 50)

    trainer = SupremacyTrainer(target_accuracy=0.90)

    try:
        success = trainer.train_for_supremacy()

        if success:
            print("\n🎯 SUPREMACY TRAINING COMPLETE!")
            print("   🏆 >90% directional accuracy achieved!")
            print("   💪 Ready to dominate subnet 55!")
        else:
            print("\n📈 Training complete - target not achieved")
            print("   🔄 Consider architecture improvements")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    main()
