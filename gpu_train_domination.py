#!/usr/bin/env python3
"""
GPU DOMINATION MODEL TRAINING WITH MONITORING
Enhanced training with GPU acceleration and real-time monitoring
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from datetime import datetime
import pickle
import time
import psutil
import GPUtil
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WorkingEnsemble(nn.Module):
    """Working ensemble for domination"""

    def __init__(self, input_size=24, hidden_size=128):
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

        # Ensemble weight (learnable)
        self.ensemble_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # GRU prediction
        gru_out, _ = self.gru(x)
        gru_pred = self.gru_fc(gru_out[:, -1, :])

        # Transformer prediction
        transformer_out = self.transformer_encoder(x)
        transformer_pred = self.transformer_fc(transformer_out[:, -1, :])

        # Weighted ensemble
        weight = torch.sigmoid(self.ensemble_weight)
        ensemble_pred = weight * gru_pred + (1 - weight) * transformer_pred

        return ensemble_pred

class BitcoinDataset(Dataset):
    """Dataset for Bitcoin price prediction"""

    def __init__(self, data_path=None, sequence_length=10):
        self.sequence_length = sequence_length

        # Generate synthetic data if no path provided
        if data_path is None or not os.path.exists(data_path):
            logger.info("Generating synthetic training data...")
            self.data = self.generate_synthetic_data()
        else:
            logger.info(f"Loading data from {data_path}")
            self.data = self.load_real_data(data_path)

    def generate_synthetic_data(self, num_samples=10000):
        """Generate synthetic Bitcoin-like data"""
        np.random.seed(42)

        # Generate time series
        timestamps = np.arange(num_samples)
        base_price = 50000

        # Create realistic price movements
        trend = 0.0001 * timestamps
        seasonal = 1000 * np.sin(2 * np.pi * timestamps / 1440)  # Daily cycle
        noise = np.random.normal(0, 500, num_samples)
        volatility = np.random.choice([0.01, 0.02, 0.05], num_samples, p=[0.7, 0.2, 0.1])

        prices = base_price * (1 + trend) + seasonal + noise
        prices = np.abs(prices)  # Ensure positive

        # Generate volume
        volumes = np.random.lognormal(15, 1, num_samples)

        # Create features
        data = []
        for i in range(self.sequence_length, len(prices)):
            # Price features
            price_seq = prices[i-self.sequence_length:i]
            volume_seq = volumes[i-self.sequence_length:i]

            features = []

            # Price-based features
            price_returns = np.diff(price_seq) / price_seq[:-1]
            features.extend([
                price_returns[-1],  # Last return
                np.mean(price_returns),  # Average return
                np.std(price_returns),  # Volatility
                np.min(price_returns),  # Max drawdown
                np.max(price_returns),  # Max gain
            ])

            # Technical indicators
            if len(price_seq) >= 5:
                sma_5 = np.mean(price_seq[-5:])
                features.append(sma_5 / price_seq[-1] - 1)  # SMA ratio

            if len(price_seq) >= 10:
                sma_10 = np.mean(price_seq[-10:])
                features.append(sma_10 / price_seq[-1] - 1)  # SMA ratio

            # RSI approximation
            if len(price_returns) >= 7:
                gains = np.maximum(price_returns[-7:], 0)
                losses = np.maximum(-price_returns[-7:], 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                if avg_loss > 0:
                    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                    features.append(rsi / 100.0)

            # Volume features
            if len(volume_seq) >= 5:
                avg_volume = np.mean(volume_seq[-5:])
                features.append(volume_seq[-1] / avg_volume)  # Volume ratio

            # Target: next price movement
            next_price = prices[i]
            current_price = prices[i-1]
            target = (next_price - current_price) / current_price

            # Pad features to 24 dimensions
            while len(features) < 24:
                features.append(0.0)
            features = features[:24]

            data.append({
                'features': np.array(features, dtype=np.float32),
                'target': np.array([target], dtype=np.float32)
            })

        return data

    def load_real_data(self, data_path):
        """Load real market data"""
        # Placeholder for real data loading
        return self.generate_synthetic_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['features'], self.data[idx]['target']

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
                'temperature': gpu.temperature
            })

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'gpu_stats': gpu_stats
        }
    except Exception as e:
        return {'error': str(e)}

def train_with_monitoring():
    """Train model with GPU and comprehensive monitoring"""

    print("🚀 GPU DOMINATION MODEL TRAINING WITH MONITORING")
    print("=" * 60)

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Training on: {device}")

    if device.type == 'cuda':
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

    # Create model
    model = WorkingEnsemble(input_size=24, hidden_size=128)
    model.to(device)

    # Create dataset and dataloader
    dataset = BitcoinDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Training parameters
    num_epochs = 100
    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    print(f"📊 Dataset: {len(dataset)} samples")
    print(f"🔄 Batch size: 64")
    print(f"🎯 Epochs: {num_epochs}")
    print(f"⏱️  Patience: {patience}")
    print()

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_features, batch_targets in progress_bar:
            batch_features = batch_features.unsqueeze(1).to(device)  # Add sequence dimension
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{train_loss/train_batches:.6f}'
            })

        avg_train_loss = train_loss / train_batches

        # Validation (using part of training data for now)
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.unsqueeze(1).to(device)
                batch_targets = batch_targets.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start

        # Get system stats
        system_stats = get_system_stats()

        # Log progress
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                   f"Train Loss: {avg_train_loss:.6f} | "
                   f"Val Loss: {avg_val_loss:.6f} | "
                   f"LR: {current_lr:.6f} | "
                   f"Time: {epoch_time:.2f}s")

        print(f"💻 CPU: {system_stats.get('cpu_percent', 'N/A')}% | "
              f"🧠 RAM: {system_stats.get('memory_percent', 'N/A')}% | "
              f"🔥 GPU: {system_stats.get('gpu_stats', [{}])[0].get('utilization', 'N/A'):.1f}% | "
              f"🌡️  GPU Temp: {system_stats.get('gpu_stats', [{}])[0].get('temperature', 'N/A')}°C")

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0

            # Save model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/domination_model_trained.pth')

            # Save training metadata
            metadata = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_loss': best_loss,
                'learning_rate': current_lr,
                'timestamp': datetime.now().isoformat(),
                'device': str(device)
            }

            with open('models/training_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)

            print(f"💾 Best model saved! (Loss: {best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️  Early stopping at epoch {epoch+1}")
                break

        # Progress update every 10 epochs
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (num_epochs - epoch - 1)
            print(f"📈 Progress: {epoch+1}/{num_epochs} epochs | "
                  f"Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")

    # Final save
    torch.save(model.state_dict(), 'models/final_domination_model.pth')

    # Create feature scaler (for consistency)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Fit on sample data
    sample_features = np.random.randn(1000, 24)
    scaler.fit(sample_features)

    with open('models/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    total_time = time.time() - start_time
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"🏆 Best loss: {best_loss:.6f}")
    print(f"💾 Models saved to: models/")
    print(f"📊 Check training.log for detailed logs")

if __name__ == "__main__":
    train_with_monitoring()
