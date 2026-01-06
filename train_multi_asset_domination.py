#!/usr/bin/env python3
"""
MULTI-ASSET DOMINATION MODEL TRAINING
Train on BTC, ETH, and TAO data for superior subnet 55 performance
"""

# Set training mode BEFORE any imports to prevent model loading
import os
os.environ['TRAINING_MODE'] = 'true'

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import pickle
import time
import psutil
import GPUtil
from tqdm import tqdm
from typing import List, Dict, Tuple

# Add precog to path for imports
sys.path.append('.')

# Import required modules
from precog.utils.cm_data import CMData

# Import feature extraction function without loading models
import sys
sys.path.append('.')
import importlib.util
spec = importlib.util.spec_from_file_location("standalone_domination", "precog/miners/standalone_domination.py")
standalone_domination = importlib.util.module_from_spec(spec)
# Only load the functions we need, not the model loading code
spec.loader.exec_module(standalone_domination)
extract_comprehensive_features = standalone_domination.extract_comprehensive_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_asset_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import required modules
from precog.utils.cm_data import CMData

def simple_32_feature_extraction(data):
    """Simple 32-feature extraction for multi-asset training"""
    if len(data) < 10:
        return np.zeros(32), 0.5

    prices = data['price'].values

    # Validate prices
    if len(prices) == 0 or not np.all(np.isfinite(prices)):
        return np.zeros(32), 0.5

    current_price = prices[-1]

    # Ensure current price is valid
    if not np.isfinite(current_price) or current_price <= 0:
        return np.zeros(32), 0.5

    features = np.zeros(32)

    # Basic price features (0-7) - returns with safety checks
    if len(prices) >= 2:
        prev_price = prices[-2]
        if np.isfinite(prev_price) and prev_price > 0:
            features[0] = (current_price - prev_price) / prev_price  # 1-step return

    if len(prices) >= 6:
        prev_price = prices[-6]
        if np.isfinite(prev_price) and prev_price > 0:
            features[1] = (current_price - prev_price) / prev_price  # 5-step return

    if len(prices) >= 16:
        prev_price = prices[-16]
        if np.isfinite(prev_price) and prev_price > 0:
            features[2] = (current_price - prev_price) / prev_price  # 15-step return

    # Moving averages (4-5) - normalized with safety
    if len(prices) >= 5:
        ma5 = np.mean(prices[-5:])
        if np.isfinite(ma5):
            features[4] = ma5 / current_price - 1

    if len(prices) >= 10:
        ma10 = np.mean(prices[-10:])
        if np.isfinite(ma10):
            features[5] = ma10 / current_price - 1

    # Technical indicators (6-11) with safety checks
    if len(prices) >= 14:
        # Simple RSI approximation
        gains = []
        losses = []
        for i in range(1, min(15, len(prices))):
            if prices[-i-1] > 0 and np.isfinite(prices[-i]) and np.isfinite(prices[-i-1]):
                change_pct = (prices[-i] - prices[-i-1]) / prices[-i-1]
                if change_pct > 0:
                    gains.append(change_pct)
                else:
                    losses.append(-change_pct)

        if gains and losses:
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss > 0 and np.isfinite(avg_gain) and np.isfinite(avg_loss):
                rs = avg_gain / avg_loss
                features[6] = 100 - (100 / (1 + rs))  # RSI

    # Simple MACD-style
    if len(prices) >= 26:
        short_avg = np.mean(prices[-12:])
        long_avg = np.mean(prices[-26:])
        if np.isfinite(short_avg) and np.isfinite(long_avg):
            features[7] = (short_avg - long_avg) / current_price

    # Bollinger Bands
    if len(prices) >= 20:
        sma = np.mean(prices[-20:])
        std = np.std(prices[-20:])
        if np.isfinite(sma) and np.isfinite(std) and std > 0:
            features[9] = (current_price - sma) / (2 * std)

    # Statistical features (12-23)
    if len(prices) >= 10:
        # Calculate returns safely
        valid_prices = prices[np.isfinite(prices) & (prices > 0)]
        if len(valid_prices) >= 2:
            returns = np.diff(valid_prices) / valid_prices[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) > 0:
                features[12] = np.mean(returns) if np.isfinite(np.mean(returns)) else 0
                features[13] = np.std(returns) if np.isfinite(np.std(returns)) else 0
                features[14] = np.max(returns) if np.isfinite(np.max(returns)) else 0
                features[15] = np.min(returns) if np.isfinite(np.min(returns)) else 0

    # Price-based features (24-27) with safety
    features[24] = np.log(current_price) / 10 if current_price > 0 else 0  # Log price (scaled)

    if len(prices) >= 20:
        vol = np.std(prices[-20:])
        if np.isfinite(vol):
            features[25] = vol / current_price

    features[26] = len(prices) / 1000  # Data length indicator

    if len(prices) > 0:
        mean_price = np.mean(prices)
        if np.isfinite(mean_price):
            features[27] = mean_price / current_price - 1

    # Asset-specific features (28-31)
    features[28] = 1.0  # Asset type indicator
    features[29] = np.random.normal(0, 0.01)  # Small noise
    features[30] = np.random.normal(0, 0.01)  # Small noise
    features[31] = np.random.normal(0, 0.01)  # Small noise

    # Final safety check - replace any NaN or inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    confidence = 0.8  # Fixed confidence for training
    return features, confidence

class WorkingEnsemble(nn.Module):
    """Working ensemble for domination"""

    def __init__(self, input_size=32, hidden_size=128):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # GRU branch
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.gru_fc = nn.Linear(hidden_size, 1)

        # Transformer branch
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=4,
                dim_feedforward=hidden_size,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.transformer_fc = nn.Linear(input_size, 1)
        self.ensemble_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # GRU prediction
        gru_out, _ = self.gru(x)
        gru_pred = self.gru_fc(gru_out[:, -1, :])

        # Transformer prediction
        transformer_out = self.transformer_encoder(x)
        transformer_pred = self.transformer_fc(transformer_out[:, -1, :])

        # Ensemble
        weight = torch.sigmoid(self.ensemble_weight)
        prediction = weight * gru_pred + (1 - weight) * transformer_pred

        return prediction

class MultiAssetDominationDataset(Dataset):
    """Dataset for multi-asset training"""

    def __init__(self, assets: List[str] = ['btc', 'eth', 'tao_bittensor'], samples_per_asset: int = 5000):
        self.assets = assets
        self.samples_per_asset = samples_per_asset
        self.cm = CMData()
        self.feature_means = None
        self.feature_stds = None

        logger.info(f"📊 Loading data for assets: {assets}")
        self.data = self.load_multi_asset_data()

        # Fit scaler on combined data
        self.fit_scaler()

    def load_multi_asset_data(self) -> List[Dict]:
        """Load and combine data from multiple assets"""
        all_data = []

        for asset in self.assets:
            logger.info(f"🔄 Fetching data for {asset.upper()}...")
            try:
                # Fetch 6 hours of data (reasonable amount for training)
                asset_data = self.cm.get_recent_data(minutes=360, asset=asset)  # 6 hours

                if asset_data.empty:
                    logger.warning(f"No data for {asset}, using synthetic data")
                    asset_samples = self.generate_synthetic_asset_data(asset, self.samples_per_asset)
                else:
                    logger.info(f"✅ Got {len(asset_data)} data points for {asset}")
                    asset_samples = self.process_real_asset_data(asset_data, asset)

                    # If real data processing failed completely, fall back to synthetic
                    if len(asset_samples) == 0:
                        logger.warning(f"Real data processing failed for {asset}, using synthetic data")
                        asset_samples = self.generate_synthetic_asset_data(asset, self.samples_per_asset)

                all_data.extend(asset_samples)
                logger.info(f"📈 Generated {len(asset_samples)} training samples for {asset}")

            except Exception as e:
                logger.error(f"❌ Error loading {asset}: {e}")
                # Fallback to synthetic data
                asset_samples = self.generate_synthetic_asset_data(asset, self.samples_per_asset)
                all_data.extend(asset_samples)

        logger.info(f"🎯 Total training samples: {len(all_data)}")
        return all_data

    def process_real_asset_data(self, asset_df, asset: str) -> List[Dict]:
        """Process real asset data into training samples"""
        logger.info(f"🔄 Processing {len(asset_df)} rows of {asset} data")
        samples = []

        # Convert to numpy array
        prices = asset_df['price'].values
        volumes = asset_df.get('volume', np.ones(len(prices))).values

        logger.info(f"📊 {asset}: {len(prices)} price points, price range ${prices.min():.2f} - ${prices.max():.2f}")

        # Create sliding windows
        window_size = 60  # 60 minutes for features
        step_size = 5    # Every 5 minutes

        logger.info(f"🔄 Creating sliding windows: window_size={window_size}, step_size={step_size}")

        successful_samples = 0
        failed_samples = 0

        for i in range(window_size, len(prices) - 1, step_size):
            try:
                # Get price window
                price_window = prices[i-window_size:i]

                if len(price_window) != window_size:
                    failed_samples += 1
                    continue

                # Create mock dataframe for feature extraction
                mock_data = pd.DataFrame({
                    'price': price_window,
                    'volume': volumes[i-window_size:i] if len(volumes) > i else np.ones(window_size)
                })

                # Extract 32 features for multi-asset training
                features, _ = simple_32_feature_extraction(mock_data)

                if features is None or len(features) == 0:
                    failed_samples += 1
                    continue

                # Target: next price movement (normalized)
                current_price = prices[i]
                next_price = prices[i+1] if i+1 < len(prices) else prices[i]
                target = (next_price - current_price) / current_price

                # Clip extreme targets
                target = np.clip(target, -0.1, 0.1)

                samples.append({
                    'features': features.astype(np.float32),
                    'target': np.float32(target),
                    'asset': asset
                })

                successful_samples += 1

                # Log progress every 100 samples
                if successful_samples % 100 == 0:
                    logger.info(f"✅ {asset}: Generated {successful_samples} samples so far")

            except Exception as e:
                failed_samples += 1
                if failed_samples % 100 == 0:  # Only log every 100 failures to avoid spam
                    logger.warning(f"⚠️ {asset}: {failed_samples} failed samples, last error: {e}")
                continue

        logger.info(f"🎯 {asset}: Generated {len(samples)} training samples ({successful_samples} successful, {failed_samples} failed)")
        return samples

    def generate_synthetic_asset_data(self, asset: str, num_samples: int) -> List[Dict]:
        """Generate synthetic data for an asset when real data unavailable"""
        samples = []

        # Different base prices for different assets
        base_prices = {
            'btc': 95000,
            'eth': 3200,
            'tao_bittensor': 400
        }

        base_price = base_prices.get(asset, 1000)
        volatility = 0.02  # 2% volatility

        for _ in range(num_samples):
            # Generate synthetic price series
            prices = []
            price = base_price * (1 + np.random.normal(0, 0.1))  # Start with some variation

            for _ in range(60):  # 60-minute window
                # Random walk with mean reversion
                change = np.random.normal(0, volatility)
                price *= (1 + change)
                prices.append(price)

            # Create mock dataframe
            mock_data = pd.DataFrame({
                'price': prices,
                'volume': np.ones(60)
            })

            # Extract 32 features
            features, _ = simple_32_feature_extraction(mock_data)

            # Generate target
            target = np.random.normal(0, 0.01)  # Small price movement
            target = np.clip(target, -0.05, 0.05)

            samples.append({
                'features': features.astype(np.float32),
                'target': np.float32(target),
                'asset': asset
            })

        return samples

    def fit_scaler(self):
        """Fit feature scaler on training data"""
        logger.info("🔧 Fitting feature scaler...")

        if len(self.data) == 0:
            logger.error("❌ No training data available for scaler fitting!")
            raise ValueError("No training data available")

        all_features = np.array([sample['features'] for sample in self.data])

        logger.info(f"📊 Feature array shape: {all_features.shape}")

        # Calculate means and stds
        self.feature_means = np.mean(all_features, axis=0)
        self.feature_stds = np.std(all_features, axis=0)

        # Ensure they're numpy arrays
        self.feature_means = np.array(self.feature_means)
        self.feature_stds = np.array(self.feature_stds)

        # Avoid division by zero
        self.feature_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)

        logger.info(f"📏 Scaler fitted on {len(all_features)} samples with {len(self.feature_means)} features")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Apply scaling
        scaled_features = (sample['features'] - self.feature_means) / self.feature_stds

        return scaled_features, sample['target']

class DominationTrainer:
    """Trainer for domination model"""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.best_loss = float('inf')

        # Optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.MSELoss()

    def train(self, train_loader, val_loader, epochs=100):
        logger.info("🏋️ Starting multi-asset domination training...")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_features, batch_targets in progress_bar:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_features.unsqueeze(1))  # Add sequence dimension
                loss = self.criterion(outputs.squeeze(), batch_targets)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

            train_loss /= len(train_loader)

            # Validation
            val_loss = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            logger.info(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), 'models/multi_asset_domination_model.pth')
                logger.info(f"💾 Saved best multi-asset model with loss: {val_loss:.6f}")

        logger.info(f"Training complete. Best validation loss: {self.best_loss:.6f}")

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                outputs = self.model(batch_features.unsqueeze(1))
                loss = self.criterion(outputs.squeeze(), batch_targets)
                val_loss += loss.item()

        return val_loss / len(val_loader)

def get_system_stats():
    """Get system resource usage"""
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # GPU
        gpus = GPUtil.getGPUs()
        gpu_stats = []
        for gpu in gpus:
            gpu_stats.append({
                'name': gpu.name,
                'utilization': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
            })

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'gpu_stats': gpu_stats,
            'timestamp': time.time()
        }
    except Exception as e:
        logger.warning(f"Could not get system stats: {e}")
        return None

def main():
    logger.info("🚀 TRAINING MULTI-ASSET DOMINATION MODEL")
    logger.info("=" * 60)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cpu':
        logger.warning("⚠️ CUDA not available, training on CPU will be slow!")

    # Create multi-asset dataset
    logger.info("📊 Creating multi-asset training dataset...")
    dataset = MultiAssetDominationDataset(
        assets=['btc', 'eth', 'tao_bittensor'],
        samples_per_asset=5000
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Create model (32 features now for enhanced features)
    logger.info("🤖 Creating fresh multi-asset domination ensemble...")
    model = WorkingEnsemble(input_size=32, hidden_size=128)

    # Don't load existing model - start fresh for multi-asset training
    logger.info("🔄 Starting fresh training (not loading existing BTC-only model)")

    # Train model
    trainer = DominationTrainer(model, device)

    # Monitor training with system stats
    logger.info("📈 Starting training with system monitoring...")
    start_time = time.time()

    trainer.train(train_loader, val_loader, epochs=100)

    training_time = time.time() - start_time
    logger.info(f"⏱️ Total training time: {training_time:.2f} seconds")

    # Save scaler and metadata
    os.makedirs('models', exist_ok=True)
    scaler_data = {
        'means': dataset.feature_means,
        'stds': dataset.feature_stds,
        'assets': ['btc', 'eth', 'tao_bittensor'],
        'training_time': training_time,
        'timestamp': datetime.now().isoformat(),
        'best_loss': trainer.best_loss
    }
    with open('models/multi_asset_feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_data, f)

    # Save training metadata
    metadata = {
        'model_type': 'multi_asset_domination',
        'assets': ['btc', 'eth', 'tao_bittensor'],
        'best_loss': trainer.best_loss,
        'epoch': 100,
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
        'training_time_seconds': training_time
    }
    with open('models/multi_asset_training_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    logger.info("✅ MULTI-ASSET DOMINATION MODEL TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info("🏆 Model Capabilities:")
    logger.info("  • Multi-Asset Training: BTC, ETH, TAO")
    logger.info("  • Ensemble: GRU + Transformer")
    logger.info("  • Features: 32 advanced indicators per asset")
    logger.info("  • Best Loss: {:.6f}".format(trainer.best_loss))
    logger.info("  • Training Time: {:.2f} seconds".format(training_time))
    logger.info("=" * 60)
    logger.info("🎯 Ready for subnet 55 domination!")

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import pandas as pd
    except ImportError:
        logger.info("Installing pandas...")
        os.system("pip install pandas")

    main()
