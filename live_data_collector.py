#!/usr/bin/env python3
"""
Continuous Live Market Data Collector for Post-Deployment Model Improvement

This script runs continuously alongside your miner to collect fresh market data
for periodic model retraining and improvement.
"""

import sys
import os
import json
import csv
import time
import threading
import signal
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_data_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append('.')
from simple_data_fetch import SimpleCryptoDataFetcher


class LiveDataCollector:
    """Continuously collect live market data for model improvement"""

    def __init__(self, symbols=None, collection_interval=300, max_data_age_days=7):
        """
        Args:
            symbols: List of symbols to track
            collection_interval: Seconds between data collection (default 5 minutes)
            max_data_age_days: Maximum age of data to keep
        """
        self.symbols = symbols or ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'BNB', 'LINK', 'XRP']
        self.collection_interval = collection_interval
        self.max_data_age = timedelta(days=max_data_age_days)

        self.fetcher = SimpleCryptoDataFetcher()
        self.data_file = 'live_market_data.json'
        self.running = False
        self.thread = None

        # Load existing data
        self.market_data = self.load_data()
        logger.info(f"Loaded {len(self.market_data)} existing data points")

    def load_data(self):
        """Load existing market data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    # Convert timestamps back to datetime objects
                    for symbol_data in data.values():
                        for item in symbol_data:
                            item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")

        return defaultdict(list)

    def save_data(self):
        """Save market data to file"""
        try:
            # Convert datetime objects to ISO strings for JSON serialization
            data_to_save = {}
            for symbol, data_list in self.market_data.items():
                data_to_save[symbol] = []
                for item in data_list:
                    item_copy = item.copy()
                    item_copy['timestamp'] = item_copy['timestamp'].isoformat()
                    data_to_save[symbol].append(item_copy)

            with open(self.data_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)

            logger.info(f"Saved {sum(len(v) for v in data_to_save.values())} data points")

        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def clean_old_data(self):
        """Remove data older than max_data_age"""
        cutoff_time = datetime.now(timezone.utc) - self.max_data_age
        total_removed = 0

        for symbol in self.market_data:
            original_count = len(self.market_data[symbol])
            self.market_data[symbol] = [
                item for item in self.market_data[symbol]
                if item['timestamp'] > cutoff_time
            ]
            total_removed += original_count - len(self.market_data[symbol])

        if total_removed > 0:
            logger.info(f"Cleaned {total_removed} old data points")

    def collect_data(self):
        """Collect latest data for all symbols"""
        logger.info(f"Collecting data for {len(self.symbols)} symbols...")

        new_data_count = 0

        for symbol in self.symbols:
            try:
                # Get latest data
                temp_file = f'temp_{symbol}_data.csv'
                success = self.fetcher.create_training_csv([symbol], temp_file)

                if success:
                    # Read the data
                    with open(temp_file, 'r') as f:
                        reader = csv.DictReader(f)
                        symbol_data = []

                        for row in reader:
                            data_point = {
                                'timestamp': datetime.fromisoformat(row['timestamp']),
                                'open': float(row['open']),
                                'high': float(row['high']),
                                'low': float(row['low']),
                                'close': float(row['close']),
                                'volume': float(row['volume'])
                            }
                            symbol_data.append(data_point)

                    if symbol_data:
                        # Add to our collection (avoiding duplicates)
                        existing_timestamps = {
                            item['timestamp'] for item in self.market_data[symbol]
                        }

                        new_points = [
                            point for point in symbol_data
                            if point['timestamp'] not in existing_timestamps
                        ]

                        self.market_data[symbol].extend(new_points)
                        new_data_count += len(new_points)

                        if new_points:
                            logger.info(f"Added {len(new_points)} new points for {symbol}")

                    # Clean up temp file
                    try:
                        os.remove(temp_file)
                    except:
                        pass

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error collecting {symbol} data: {e}")

        logger.info(f"Data collection complete. Added {new_data_count} new data points")

    def get_training_data_csv(self, output_file='live_training_data.csv', hours_back=168):
        """Export recent data as CSV for model retraining"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        all_data = []
        for symbol, data_list in self.market_data.items():
            for item in data_list:
                if item['timestamp'] > cutoff_time:
                    row = item.copy()
                    row['symbol'] = symbol
                    all_data.append(row)

        if not all_data:
            logger.warning("No data available for CSV export")
            return False

        # Sort by timestamp
        all_data.sort(key=lambda x: x['timestamp'])

        # Write to CSV
        fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)

        logger.info(f"Exported {len(all_data)} data points to {output_file}")
        return True

    def get_stats(self):
        """Get collection statistics"""
        total_points = sum(len(data) for data in self.market_data.values())
        oldest_timestamp = None
        newest_timestamp = None

        for data_list in self.market_data.values():
            for item in data_list:
                if oldest_timestamp is None or item['timestamp'] < oldest_timestamp:
                    oldest_timestamp = item['timestamp']
                if newest_timestamp is None or item['timestamp'] > newest_timestamp:
                    newest_timestamp = item['timestamp']

        stats = {
            'total_data_points': total_points,
            'symbols_tracked': len(self.market_data),
            'oldest_data': oldest_timestamp.isoformat() if oldest_timestamp else None,
            'newest_data': newest_timestamp.isoformat() if newest_timestamp else None,
            'data_age_hours': (newest_timestamp - oldest_timestamp).total_seconds() / 3600 if oldest_timestamp and newest_timestamp else 0,
            'collection_interval_minutes': self.collection_interval / 60,
            'max_data_age_days': self.max_data_age.days
        }

        return stats

    def collection_loop(self):
        """Main collection loop"""
        logger.info("Starting live data collection...")
        logger.info(f"Collection interval: {self.collection_interval} seconds")
        logger.info(f"Symbols: {', '.join(self.symbols)}")

        while self.running:
            try:
                start_time = time.time()

                # Collect data
                self.collect_data()

                # Clean old data
                self.clean_old_data()

                # Save data
                self.save_data()

                # Log stats
                stats = self.get_stats()
                logger.info(f"Collection stats: {stats['total_data_points']} points, "
                          f"{stats['symbols_tracked']} symbols")

                # Sleep until next collection
                elapsed = time.time() - start_time
                sleep_time = max(0, self.collection_interval - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

    def start(self):
        """Start the data collection"""
        if self.running:
            logger.warning("Data collector already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self.collection_loop, daemon=True)
        self.thread.start()

        logger.info("Live data collector started")

    def stop(self):
        """Stop the data collection"""
        if not self.running:
            logger.warning("Data collector not running")
            return

        logger.info("Stopping live data collector...")
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)

        # Final save
        self.save_data()

        logger.info("Live data collector stopped")


def retrain_model(live_data_file='live_training_data.csv', model_name='live_improved_model.pth'):
    """Retrain model on collected live data"""
    try:
        logger.info("Starting model retraining on live data...")

        # Check if we have enough data
        if not os.path.exists(live_data_file):
            logger.error(f"Live data file {live_data_file} not found")
            return False

        # Import here to avoid circular imports
        import torch
        import torch.nn as nn
        import numpy as np
        from advanced_attention_mechanisms import create_enhanced_attention_ensemble

        # Load current best model
        model_files = ['live_fine_tuned_model.pth', 'robust_deployment_model.pth']
        current_model_file = None

        for model_file in model_files:
            if os.path.exists(model_file):
                current_model_file = model_file
                break

        if not current_model_file:
            logger.error("No existing model found for retraining")
            return False

        logger.info(f"Loading model from {current_model_file}")
        checkpoint = torch.load(current_model_file, map_location='cpu', weights_only=False)

        model = create_enhanced_attention_ensemble(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        # Prepare live data for training
        logger.info(f"Preparing live data from {live_data_file}")

        # Read CSV data
        live_data = []
        with open(live_data_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                live_data.append({
                    'symbol': row['symbol'],
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                })

        if len(live_data) < 1000:
            logger.warning("Insufficient live data for retraining")
            return False

        # Group by symbol and prepare features
        symbol_groups = defaultdict(list)
        for row in live_data:
            symbol_groups[row['symbol']].append(row)

        # Use same preprocessing as original training
        norm_params = checkpoint['normalization']
        close_mean = norm_params['close_mean']
        close_std = norm_params['close_std']
        vol_mean = norm_params['vol_mean']
        vol_std = norm_params['vol_std']

        all_features = []
        all_targets = []

        for symbol, data in symbol_groups.items():
            if len(data) < 30:
                continue

            closes = np.array([row['close'] for row in data])
            volumes = np.array([row['volume'] for row in data])

            # Same normalization
            closes_norm = (closes - close_mean) / (close_std + 1e-8)
            volumes_norm = (volumes - vol_mean) / (vol_std + 1e-8)

            # Simple features for live data
            features = [closes_norm, volumes_norm]
            feature_matrix = np.column_stack(features)

            # Create targets
            targets = []
            for i in range(1, len(closes_norm)):
                current = closes_norm[i-1]
                next_val = closes_norm[i]
                if current != 0:
                    pct_change = (next_val - current) / abs(current)
                    target = np.clip(pct_change * 50, -10, 10)
                else:
                    target = 0
                targets.append(target)
            targets.insert(0, 0.0)
            targets = np.array(targets, dtype=np.float32)

            # Create sequences
            seq_len = 25
            for i in range(seq_len, len(feature_matrix)):
                all_features.append(feature_matrix[i-seq_len:i])
                all_targets.append(targets[i])

        if not all_features:
            logger.error("No valid sequences created from live data")
            return False

        X_live = np.array(all_features, dtype=np.float32)
        y_live = np.array(all_targets, dtype=np.float32)

        logger.info(f"Created {len(X_live)} training sequences from live data")

        # Fine-tune on live data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=0.5)

        # Quick fine-tuning
        batch_size = 16
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_live), torch.from_numpy(y_live)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        logger.info("Fine-tuning model on live data...")

        for epoch in range(3):  # Quick fine-tuning
            model.train()
            epoch_loss = 0
            n_batches = 0

            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()

                outputs = model(batch_X)
                if isinstance(outputs, tuple):
                    predictions, _ = outputs
                else:
                    predictions = outputs

                if predictions.dim() > 1:
                    predictions = predictions.squeeze(-1)

                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

        # Quick validation
        model.eval()
        val_predictions = []
        val_actuals = []

        with torch.no_grad():
            val_size = min(50, len(X_live) // 4)
            for i in range(val_size):
                x = torch.from_numpy(X_live[-val_size + i:i - val_size + i + 1]).to(device)
                if len(x) == 0:
                    continue

                output = model(x)
                if isinstance(output, tuple):
                    pred, _ = output
                else:
                    pred = output
                pred_val = pred.cpu().numpy().flatten()[0]
                val_predictions.append(pred_val)
                val_actuals.append(y_live[-val_size + i])

        if val_predictions:
            directional_acc = np.mean(np.sign(np.array(val_predictions)) == np.sign(np.array(val_actuals)))
            logger.info(f"Live retraining validation - Directional Accuracy: {directional_acc:.1%}")

        # Save improved model
        improved_checkpoint = checkpoint.copy()
        improved_checkpoint['model_state_dict'] = model.state_dict()
        improved_checkpoint['retraining_timestamp'] = datetime.now(timezone.utc).isoformat()
        improved_checkpoint['live_data_points'] = len(X_live)

        torch.save(improved_checkpoint, model_name)

        logger.info(f"✅ Model retrained and saved as {model_name}")
        return True

    except Exception as e:
        logger.error(f"Error during model retraining: {e}")
        return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Live Market Data Collector')
    parser.add_argument('--action', choices=['start', 'stop', 'status', 'export', 'retrain'],
                       default='status', help='Action to perform')
    parser.add_argument('--symbols', nargs='+',
                       default=['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'BNB'],
                       help='Symbols to track')
    parser.add_argument('--interval', type=int, default=300,
                       help='Collection interval in seconds (default: 300 = 5 minutes)')
    parser.add_argument('--export-file', default='live_training_data.csv',
                       help='File to export training data to')
    parser.add_argument('--hours-back', type=int, default=168,
                       help='Hours of historical data to export (default: 168 = 1 week)')

    args = parser.parse_args()

    collector = LiveDataCollector(
        symbols=args.symbols,
        collection_interval=args.interval
    )

    if args.action == 'start':
        print("🚀 Starting live data collector...")
        collector.start()

        # Keep running
        try:
            while True:
                time.sleep(60)
                stats = collector.get_stats()
                print(f"📊 Status: {stats['total_data_points']} points, "
                      f"{stats['symbols_tracked']} symbols")

        except KeyboardInterrupt:
            print("\\n🛑 Stopping collector...")
            collector.stop()

    elif args.action == 'stop':
        collector.stop()
        print("✅ Collector stopped")

    elif args.action == 'status':
        stats = collector.get_stats()
        print("📊 LIVE DATA COLLECTOR STATUS")
        print("=" * 40)
        print(f"Total data points: {stats['total_data_points']}")
        print(f"Symbols tracked: {stats['symbols_tracked']}")
        print(f"Oldest data: {stats['oldest_data']}")
        print(f"Newest data: {stats['newest_data']}")
        print(f"Data age: {stats['data_age_hours']:.1f} hours")
        print(f"Collection interval: {stats['collection_interval_minutes']} minutes")
        print(f"Max data age: {stats['max_data_age_days']} days")

        if stats['total_data_points'] > 0:
            print("\\n✅ Collector is ACTIVE")
            print(f"💡 Run 'python3 live_data_collector.py --export' to get training data")
            print(f"💡 Run 'python3 live_data_collector.py --retrain' to improve your model")
        else:
            print("\\n⚠️ No data collected yet")
            print(f"💡 Run 'python3 live_data_collector.py --start' to begin collection")

    elif args.action == 'export':
        print(f"📤 Exporting training data to {args.export_file}...")
        success = collector.get_training_data_csv(args.export_file, args.hours_back)

        if success:
            print("✅ Data exported successfully"            print(f"🎯 Next: python3 simple_train.py --data {args.export_file}")
        else:
            print("❌ Export failed")

    elif args.action == 'retrain':
        print("🔄 Retraining model on live data...")

        # First export the data
        export_success = collector.get_training_data_csv(args.export_file, args.hours_back)

        if not export_success:
            print("❌ Could not export training data")
            return 1

        # Then retrain
        retrain_success = retrain_model(args.export_file)

        if retrain_success:
            print("✅ Model retrained successfully!")
            print("💡 Your miner will automatically use the improved model")
            print("🔄 Consider redeploying with the improved model:")
            print("python3 start_domination_miner.py --model live_improved_model.pth --deploy")
        else:
            print("❌ Model retraining failed")


if __name__ == "__main__":
    sys.exit(main())

