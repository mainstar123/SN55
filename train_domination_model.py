#!/usr/bin/env python3
"""
TRAIN DOMINATION MODEL FOR #1 POSITION
Enhanced training pipeline with domination features for top miner performance
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our domination ensemble
from precog.miners.standalone_domination import WorkingEnsemble

class DominationDataset(Dataset):
    """Dataset optimized for domination training"""

    def __init__(self, data_path, sequence_length=60, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Load and preprocess data
        logger.info("Loading training data...")
        data = pd.read_csv(data_path)

        # Extract features (same as used in domination miner)
        features = self.extract_domination_features(data)
        targets = self.create_targets(data)

        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        # Create sequences
        self.X, self.y = self.create_sequences(features_scaled, targets)

        logger.info(f"Created dataset with {len(self.X)} samples")

    def extract_domination_features(self, data):
        """Extract the 24 features used in domination miner"""
        df = data.copy()

        # Basic price features
        prices = df['price'].values
        features = []

        for i in range(len(df)):
            row_features = []

            # Price returns at different lags
            row_features.append(prices[i] / prices[i-1] - 1 if i > 0 else 0)  # 1m return
            row_features.append(prices[i] / prices[i-5] - 1 if i >= 5 else 0)  # 5m return
            row_features.append(prices[i] / prices[i-15] - 1 if i >= 15 else 0)  # 15m return

            # Moving averages
            if i >= 5:
                row_features.append(np.mean(prices[i-5:i+1]) / prices[i] - 1)  # 5-period MA
            else:
                row_features.append(0)

            if i >= 10:
                row_features.append(np.mean(prices[i-10:i+1]) / prices[i] - 1)  # 10-period MA
            else:
                row_features.append(0)

            if i >= 20:
                row_features.append(np.mean(prices[i-20:i+1]) / prices[i] - 1)  # 20-period MA
            else:
                row_features.append(0)

            # Volatility measures
            if i >= 10:
                returns = np.diff(prices[i-10:i+1]) / prices[i-10:i]
                row_features.append(np.std(returns))  # Rolling volatility
                row_features.append(np.mean(np.abs(returns)))  # Mean absolute return
            else:
                row_features.append(0)
                row_features.append(0)

            # Momentum indicators
            if i >= 10:
                row_features.append((prices[i] - prices[i-10]) / prices[i-10])  # Momentum
                row_features.append(np.sum(np.diff(prices[i-10:i+1]) > 0))  # Up days
            else:
                row_features.append(0)
                row_features.append(0)

            # RSI-like indicator (simplified)
            if i >= 14:
                gains = np.diff(prices[i-14:i+1])
                avg_gain = np.mean(gains[gains > 0]) if np.any(gains > 0) else 0
                avg_loss = np.mean(-gains[gains < 0]) if np.any(gains < 0) else 0
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi = 100 - (100 / (1 + rs))
                row_features.append(rsi / 100)  # Normalized RSI
            else:
                row_features.append(0.5)  # Neutral

            # Bollinger Bands (simplified)
            if i >= 20:
                ma = np.mean(prices[i-20:i+1])
                std = np.std(prices[i-20:i+1])
                row_features.append((prices[i] - ma) / std)  # Position relative to bands
            else:
                row_features.append(0)

            # Volume-based features (if available)
            if 'volume' in df.columns:
                volume = df['volume'].iloc[i]
                if i >= 5:
                    avg_volume = np.mean(df['volume'].iloc[i-5:i+1])
                    row_features.append(volume / avg_volume - 1)  # Volume ratio
                else:
                    row_features.append(0)
            else:
                row_features.append(0)

            # Additional technical features to reach 24
            # MACD-like (simplified)
            if i >= 26:
                ema12 = np.mean(prices[i-12:i+1])
                ema26 = np.mean(prices[i-26:i+1])
                row_features.append((ema12 - ema26) / prices[i])  # MACD
            else:
                row_features.append(0)

            # Stochastic Oscillator (simplified)
            if i >= 14:
                high14 = np.max(prices[i-14:i+1])
                low14 = np.min(prices[i-14:i+1])
                stoch = (prices[i] - low14) / (high14 - low14) if high14 != low14 else 0.5
                row_features.append(stoch)
            else:
                row_features.append(0.5)

            # Fill remaining features with zeros to reach 24
            while len(row_features) < 24:
                row_features.append(0)

            features.append(row_features[:24])  # Ensure exactly 24 features

        return np.array(features)

    def create_targets(self, data):
        """Create prediction targets"""
        prices = data['price'].values
        targets = []

        for i in range(len(prices)):
            if i + self.prediction_horizon < len(prices):
                # Predict price change
                future_price = prices[i + self.prediction_horizon]
                current_price = prices[i]
                target = (future_price - current_price) / current_price
                targets.append(target)
            else:
                targets.append(0)  # Padding

        return np.array(targets)

    def create_sequences(self, features, targets):
        """Create input sequences for training"""
        X, y = [], []

        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(targets[i + self.sequence_length])

        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

class DominationTrainer:
    """Enhanced trainer for domination model"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)

        # Enhanced training setup
        self.criterion = nn.HuberLoss()  # More robust than MSE
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')

    def train_epoch(self, train_loader):
        """Train for one epoch with domination optimizations"""
        self.model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def validate(self, val_loader):
        """Validate model performance"""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=100, patience=10):
        """Complete training with early stopping and model saving"""
        logger.info(f"Starting domination training for {epochs} epochs")

        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step()

            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_domination_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(f"Training complete. Best validation loss: {self.best_loss:.6f}")
        return self.train_losses, self.val_losses

def create_enhanced_training_data():
    """Create or enhance training data for domination model"""

    # Check if we have existing training data
    if os.path.exists('data/training_data.csv'):
        logger.info("Using existing training data")
        return 'data/training_data.csv'

    # Generate synthetic high-quality training data
    logger.info("Generating enhanced training data for domination model")

    # Generate realistic BTC price data with various market conditions
    np.random.seed(42)
    n_samples = 50000  # Large dataset for domination training

    # Create time series with different market regimes
    time = np.arange(n_samples)

    # Base trend with noise
    trend = 88000 + 5000 * np.sin(2 * np.pi * time / 1000)  # Long-term cycles
    noise = np.random.normal(0, 200, n_samples)  # Price noise

    # Add volatility clusters (market regimes)
    volatility = np.ones(n_samples) * 0.01  # Base volatility

    # High volatility periods
    high_vol_periods = [(5000, 8000), (15000, 18000), (25000, 28000), (35000, 38000)]
    for start, end in high_vol_periods:
        volatility[start:end] = 0.03

    # Low volatility periods
    low_vol_periods = [(10000, 13000), (20000, 23000), (30000, 33000), (40000, 43000)]
    for start, end in low_vol_periods:
        volatility[start:end] = 0.005

    # Generate price series
    returns = np.random.normal(0, volatility)
    price = trend * (1 + np.cumsum(returns))

    # Add volume data
    base_volume = 1000000
    volume_noise = np.random.lognormal(0, 0.5, n_samples)
    volume = base_volume * volume_noise

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': time,
        'price': price,
        'volume': volume
    })

    # Save training data
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/training_data.csv', index=False)

    logger.info(f"Generated {n_samples} training samples")
    return 'data/training_data.csv'

def main():
    """Main training function for domination model"""

    logger.info("🚀 STARTING DOMINATION MODEL TRAINING")
    logger.info("=" * 60)
    logger.info("🎯 Target: Train model optimized for #1 miner performance")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create training data
    data_path = create_enhanced_training_data()

    # Create datasets
    logger.info("Creating training datasets...")
    train_dataset = DominationDataset(data_path, sequence_length=60)
    val_dataset = DominationDataset(data_path, sequence_length=60)

    # Split data (80% train, 20% validation)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Create model
    logger.info("Creating domination ensemble model...")
    model = WorkingEnsemble(input_size=24, hidden_size=128)

    # Create trainer
    trainer = DominationTrainer(model, device)

    # Train model
    logger.info("🎯 TRAINING FOR DOMINATION...")
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=200, patience=15)

    # Load best model
    logger.info("Loading best trained model...")
    model.load_state_dict(torch.load('models/best_domination_model.pth'))

    # Save final model and scaler
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/domination_model_final.pth')
    joblib.dump(train_dataset.dataset.scaler, 'models/feature_scaler_final.pkl')

    # Evaluate final performance
    final_val_loss = trainer.validate(val_loader)
    logger.info(f"Final validation loss: {final_val_loss:.6f}")

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Domination Model Training Progress')
    plt.legend()
    plt.savefig('models/training_progress.png')
    plt.close()

    logger.info("✅ DOMINATION MODEL TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info("🏆 Model Features:")
    logger.info("  • Ensemble: GRU + Transformer")
    logger.info("  • Features: 24 advanced indicators")
    logger.info("  • Training: Enhanced pipeline")
    logger.info("  • Optimization: AdamW + CosineAnnealing")
    logger.info("  • Validation Loss: {:.6f}".format(final_val_loss))
    logger.info("")
    logger.info("📁 Saved Files:")
    logger.info("  • models/domination_model_final.pth")
    logger.info("  • models/feature_scaler_final.pkl")
    logger.info("  • models/training_progress.png")
    logger.info("")
    logger.info("🚀 READY FOR MAINNET DOMINATION!")

if __name__ == "__main__":
    main()
