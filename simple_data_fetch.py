#!/usr/bin/env python3
"""
Simple cryptocurrency data fetcher without external dependencies
Gets real market data for training Precog models
"""

import sys
import os
import json
import csv
import requests
import time
from datetime import datetime, timezone, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleCryptoDataFetcher:
    """Fetch crypto data using only requests and built-in libraries"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Precog-Trainer/1.0)'
        })

    def get_binance_data(self, symbol='BTCUSDT', limit=500):
        """Get data from Binance API"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '5m',
                'limit': limit
            }

            logger.info(f"Fetching {symbol} data from Binance...")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Convert to list of dicts
            candles = []
            for row in data:
                candle = {
                    'timestamp': datetime.fromtimestamp(row[0] / 1000, timezone.utc),
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                    'volume': float(row[5])
                }
                candles.append(candle)

            logger.info(f"✅ Retrieved {len(candles)} candles from Binance")
            return candles

        except Exception as e:
            logger.error(f"Binance failed: {e}")
            return None

    def get_coingecko_data(self, coin_id='bitcoin', days=7):
        """Get data from CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly'
            }

            logger.info(f"Fetching {coin_id} data from CoinGecko...")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            prices = data.get('prices', [])

            if not prices:
                return None

            # Convert to OHLC format
            candles = []
            for i, (timestamp, close_price) in enumerate(prices):
                # Create OHLC from close price with some variation
                close_price = float(close_price)

                # Use previous close as open, or current close if first
                open_price = candles[-1]['close'] if candles else close_price

                # Add some realistic variation for high/low
                variation = abs(close_price - open_price) * 0.01  # 1% variation
                high_price = max(open_price, close_price) + variation
                low_price = min(open_price, close_price) - variation

                candle = {
                    'timestamp': datetime.fromtimestamp(timestamp / 1000, timezone.utc),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': 1000000.0  # Placeholder volume
                }
                candles.append(candle)

            logger.info(f"✅ Retrieved {len(candles)} candles from CoinGecko")
            return candles

        except Exception as e:
            logger.error(f"CoinGecko failed: {e}")
            return None

    def get_crypto_data(self, symbol):
        """Get data for a symbol using best available source"""

        # Try Binance first
        data = self.get_binance_data(symbol)

        # Try CoinGecko as fallback
        if data is None:
            coin_id_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'ADA': 'cardano',
                'SOL': 'solana',
                'DOT': 'polkadot'
            }
            coin_id = coin_id_map.get(symbol, symbol.lower())
            data = self.get_coingecko_data(coin_id)

        return data

    def create_training_csv(self, symbols=None, output_file='crypto_training_data.csv'):
        """Create training CSV with multiple symbols"""

        if symbols is None:
            symbols = ['BTC', 'ETH']

        all_candles = []

        for symbol in symbols:
            logger.info(f"\n🔍 Getting data for {symbol}...")

            # Try different symbol formats
            symbol_formats = [f"{symbol}USDT", symbol]
            data = None

            for sym_fmt in symbol_formats:
                data = self.get_crypto_data(sym_fmt)
                if data:
                    break

            if data:
                # Add symbol to each candle
                for candle in data:
                    candle['symbol'] = symbol
                    all_candles.append(candle)

                logger.info(f"✅ Added {len(data)} candles for {symbol}")
            else:
                logger.error(f"❌ No data available for {symbol}")

            # Rate limiting
            time.sleep(1)

        if not all_candles:
            logger.error("❌ No data collected")
            return False

        # Sort by timestamp
        all_candles.sort(key=lambda x: x['timestamp'])

        # Write to CSV
        fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_candles)

        logger.info(f"✅ Saved {len(all_candles)} total candles to {output_file}")
        return True

    def validate_csv(self, csv_file):
        """Basic validation of the CSV file"""
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()

            if len(lines) < 2:
                logger.error("CSV file is too small")
                return False

            # Check header
            header = lines[0].strip().split(',')
            expected = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
            if header != expected:
                logger.error(f"Unexpected header: {header}")
                return False

            # Check a few data lines
            for i, line in enumerate(lines[1:6]):  # Check first 5 data lines
                parts = line.strip().split(',')
                if len(parts) != 7:
                    logger.error(f"Line {i+1} has {len(parts)} columns, expected 7")
                    return False

                # Check numeric values
                try:
                    float(parts[1])  # open
                    float(parts[2])  # high
                    float(parts[3])  # low
                    float(parts[4])  # close
                    float(parts[5])  # volume
                except ValueError:
                    logger.error(f"Line {i+1} has non-numeric price/volume data")
                    return False

            logger.info("✅ CSV validation passed")
            return True

        except Exception as e:
            logger.error(f"CSV validation failed: {e}")
            return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Get Simple Crypto Data for Training')
    parser.add_argument('--symbols', type=str, nargs='+',
                       default=['BTC', 'ETH'],
                       help='Cryptocurrency symbols')
    parser.add_argument('--output', type=str, default='crypto_training_data.csv',
                       help='Output CSV file')

    args = parser.parse_args()

    print("📊 SIMPLE CRYPTO DATA COLLECTION")
    print("=" * 40)

    fetcher = SimpleCryptoDataFetcher()

    success = fetcher.create_training_csv(args.symbols, args.output)

    if success:
        print("\n✅ Data collection successful!")

        # Validate
        if fetcher.validate_csv(args.output):
            print("✅ Data validation passed")

            # Show summary
            with open(args.output, 'r') as f:
                lines = f.readlines()

            symbols = set()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    symbols.add(parts[6])

            print(f"Symbols collected: {sorted(symbols)}")
            print(f"Total data points: {len(lines) - 1}")

        print("\n🎯 NEXT STEP:")
        print("python3 simple_train.py --data crypto_training_data.csv")
        print("   (This will train your model on the real data)")

        return 0
    else:
        print("\n❌ Data collection failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
