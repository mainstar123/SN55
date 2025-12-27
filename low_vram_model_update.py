#!/usr/bin/env python3
"""
Low VRAM Model Update Script for RTX 4090
Limits memory usage to 7-8GB while updating models
"""

import torch
import torch.nn as nn
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LowVRAMModelUpdater:
    """Update models while limiting VRAM usage to 7-8GB on RTX 4090"""

    def __init__(self, max_vram_gb: float = 8.0):
        self.max_vram_gb = max_vram_gb
        self.device = self._setup_memory_limited_device()

        # Memory monitoring
        self.memory_stats = {
            'initial_vram': self._get_current_vram(),
            'peak_vram': 0,
            'memory_efficiency': True
        }

    def _setup_memory_limited_device(self):
        """Setup RTX 4090 with VRAM limitations"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - RTX 4090 required")

        device = torch.device('cuda:0')

        # Set memory fraction to limit usage (8GB / 24GB = ~0.33)
        memory_fraction = self.max_vram_gb / 24.0  # RTX 4090 has 24GB
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)

        # Additional memory optimizations
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False  # Disable for memory consistency
        torch.backends.cuda.matmul.allow_tf32 = False  # Disable for memory control

        logger.info(".1f")
        logger.info(f"Memory fraction set to: {memory_fraction:.2f} ({self.max_vram_gb}GB / 24GB)")
        logger.info("Additional memory optimizations enabled")

        return device

    def _get_current_vram(self) -> float:
        """Get current VRAM usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0

    def _monitor_memory(self) -> dict:
        """Monitor current memory usage"""
        current_vram = self._get_current_vram()
        self.memory_stats['peak_vram'] = max(self.memory_stats['peak_vram'], current_vram)

        return {
            'current_vram_gb': current_vram,
            'peak_vram_gb': self.memory_stats['peak_vram'],
            'remaining_vram_gb': self.max_vram_gb - current_vram,
            'memory_efficiency': current_vram <= self.max_vram_gb
        }

    def load_model_with_memory_checks(self, model_path: str, model_class):
        """Load model with memory monitoring"""
        logger.info(f"Loading model: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Create model with memory-efficient settings
        model = model_class(
            input_size=checkpoint.get('input_size', 24),
            hidden_size=min(checkpoint.get('hidden_size', 64), 64),  # Cap at 64
            dropout=checkpoint.get('dropout', 0.2)
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        # Memory check
        mem_stats = self._monitor_memory()
        logger.info(".1f")

        if not mem_stats['memory_efficiency']:
            logger.warning(".1f")
            torch.cuda.empty_cache()

        return model, checkpoint

    def update_model_incrementally(self, model, checkpoint, new_data_path: str,
                                 batch_size: int = 16, epochs: int = 5):
        """Update model with new data using limited VRAM"""

        logger.info("Starting incremental model update...")
        logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")

        # Load new training data
        if not os.path.exists(new_data_path):
            raise FileNotFoundError(f"Training data not found: {new_data_path}")

        # Load data efficiently (simplified - adjust based on your data format)
        import pandas as pd
        data = pd.read_csv(new_data_path)

        # Prepare data (simplified - adjust based on your preprocessing)
        features = data.drop(['target'], axis=1).values if 'target' in data.columns else data.values
        targets = data['target'].values if 'target' in data.columns else features[:, -1]

        # Create small batches to stay within memory limits
        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(
            torch.FloatTensor(features),
            torch.FloatTensor(targets)
        )

        # SMALL batch size for low VRAM
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer with memory-efficient settings
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        criterion = nn.MSELoss()

        # Fine-tuning loop
        model.train()
        best_loss = float('inf')

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs, _ = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                # Memory monitoring and cleanup
                if batch_count % 10 == 0:  # Every 10 batches
                    mem_stats = self._monitor_memory()
                    if not mem_stats['memory_efficiency']:
                        logger.warning(".1f")
                        torch.cuda.empty_cache()

                    logger.info(f"Epoch {epoch+1}, Batch {batch_count}: Loss={loss.item():.6f}, "
                              ".1f")

            avg_loss = epoch_loss / batch_count
            logger.info(f"Epoch {epoch+1}/{epochs}: Average Loss = {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss

        # Final memory check
        final_mem = self._monitor_memory()
        logger.info("
Model update completed!")
        logger.info(".1f")
        logger.info(f"Memory efficiency: {'✅ Maintained' if final_mem['memory_efficiency'] else '❌ Exceeded'}")

        # Create updated checkpoint
        updated_checkpoint = checkpoint.copy()
        updated_checkpoint.update({
            'model_state_dict': model.state_dict(),
            'update_timestamp': datetime.now().isoformat(),
            'fine_tune_loss': best_loss,
            'vram_limit_gb': self.max_vram_gb,
            'memory_stats': self.memory_stats
        })

        return model, updated_checkpoint

    def save_updated_model(self, model, checkpoint, output_path: str):
        """Save updated model with memory usage metadata"""
        logger.info(f"Saving updated model to: {output_path}")

        torch.save(checkpoint, output_path)

        # Save memory usage report
        memory_report = {
            'update_timestamp': datetime.now().isoformat(),
            'vram_limit_gb': self.max_vram_gb,
            'memory_stats': self.memory_stats,
            'device_info': {
                'gpu_name': torch.cuda.get_device_name(),
                'cuda_version': torch.version.cuda,
                'total_vram_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        }

        report_path = output_path.replace('.pth', '_memory_report.json')
        with open(report_path, 'w') as f:
            json.dump(memory_report, f, indent=2, default=str)

        logger.info(f"Memory usage report saved to: {report_path}")

def main():
    """Example usage for low VRAM model updates on RTX 4090"""

    print("🔄 LOW VRAM MODEL UPDATE FOR RTX 4090")
    print("=" * 50)
    print("Limiting VRAM usage to 7-8GB for model updates")
    print("=" * 50)

    # Configuration
    config = {
        'max_vram_gb': 8.0,
        'model_path': 'elite_domination_model.pth',  # Your current model
        'new_data_path': 'recent_market_data.csv',   # New training data
        'output_path': 'updated_model_low_vram.pth',
        'batch_size': 16,     # Small for low VRAM
        'epochs': 5,          # Few epochs for quick update
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    try:
        # Initialize updater
        updater = LowVRAMModelUpdater(max_vram_gb=config['max_vram_gb'])

        # Load model
        from advanced_ensemble_model import AdvancedEnsemble  # Adjust import as needed
        model, checkpoint = updater.load_model_with_memory_checks(
            config['model_path'],
            AdvancedEnsemble
        )

        # Update model
        updated_model, updated_checkpoint = updater.update_model_incrementally(
            model,
            checkpoint,
            config['new_data_path'],
            batch_size=config['batch_size'],
            epochs=config['epochs']
        )

        # Save updated model
        updater.save_updated_model(updated_model, updated_checkpoint, config['output_path'])

        print("\n✅ Model update completed successfully!")
        print(f"Updated model saved as: {config['output_path']}")
        print("VRAM usage was limited to 8GB throughout the process.")

    except Exception as e:
        print(f"❌ Error during model update: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
