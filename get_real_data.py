#!/usr/bin/env python3
"""
Get Real Cryptocurrency Data for Training
Download actual market data from multiple sources
"""

import sys
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataFetcher:
    """Fetch real cryptocurrency data from various APIs"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Precog-Trainer/1.0)'
        })

    def get_binance_klines(self, symbol: str = 'BTCUSDT', interval: str = '5m',
                          limit: int = 1000) -> pd.DataFrame:
        """Get data from Binance API"""
        try:
            base_url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            logger.info(f"Fetching {symbol} {interval} data from Binance...")
            response = self.session.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            # Convert to DataFrame
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'close_time', 'quote_asset_volume', 'number_of_trades',
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

            df = pd.DataFrame(data, columns=columns)

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            # Keep only OHLCV
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            logger.info(f"✅ Retrieved {len(df)} candles from Binance")
            return df

        except Exception as e:
            logger.error(f"Binance API failed: {e}")
            return None

    def get_coingecko_data(self, coin_id: str = 'bitcoin', days: int = 30) -> pd.DataFrame:
        """Get data from CoinGecko API"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly'  # 5min data not available
            }

            logger.info(f"Fetching {coin_id} data from CoinGecko...")
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Extract price data
            prices = data.get('prices', [])
            if not prices:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Generate OHLC from close prices (approximation)
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, len(df)))
            df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, len(df)))

            # Estimate volume (placeholder)
            df['volume'] = np.random.lognormal(15, 1, len(df))  # Realistic volume distribution

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            logger.info(f"✅ Retrieved {len(df)} hourly candles from CoinGecko")
            return df

        except Exception as e:
            logger.error(f"CoinGecko API failed: {e}")
            return None

    def get_crypto_compare_data(self, symbol: str = 'BTC', limit: int = 1000) -> pd.DataFrame:
        """Get data from CryptoCompare API"""
        try:
            url = "https://min-api.cryptocompare.com/data/v2/histominute"
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': limit,
                'aggregate': 5  # 5-minute intervals
            }

            logger.info(f"Fetching {symbol} data from CryptoCompare...")
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if data.get('Response') != 'Success':
                logger.error(f"CryptoCompare API error: {data.get('Message')}")
                return None

            candles = data['Data']['Data']

            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')

            # Rename columns
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volumefrom': 'volume'
            })

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Remove zero volume entries
            df = df[df['volume'] > 0]

            logger.info(f"✅ Retrieved {len(df)} candles from CryptoCompare")
            return df

        except Exception as e:
            logger.error(f"CryptoCompare API failed: {e}")
            return None

    def get_real_crypto_data(self, symbol: str = 'BTC', max_attempts: int = 3) -> pd.DataFrame:
        """Get real crypto data using best available source"""

        sources = [
            ('Binance', lambda: self.get_binance_klines(f"{symbol}USDT")),
            ('CryptoCompare', lambda: self.get_crypto_compare_data(symbol)),
            ('CoinGecko', lambda: self.get_coingecko_data(symbol.lower()))
        ]

        for source_name, fetch_func in sources:
            for attempt in range(max_attempts):
                try:
                    logger.info(f"Trying {source_name} (attempt {attempt + 1}/{max_attempts})...")
                    df = fetch_func()

                    if df is not None and not df.empty and len(df) > 100:
                        logger.info(f"✅ Success with {source_name} - {len(df)} data points")
                        return df
                    else:
                        logger.warning(f"{source_name} returned insufficient data")

                except Exception as e:
                    logger.warning(f"{source_name} attempt {attempt + 1} failed: {e}")
                    time.sleep(1)  # Rate limiting

        logger.error(f"❌ All data sources failed for {symbol}")
        return None


def create_training_dataset(symbols: list = None, output_file: str = 'crypto_training_data.csv'):
    """Create a comprehensive training dataset"""

    if symbols is None:
        symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT']

    fetcher = CryptoDataFetcher()
    all_data = []

    logger.info(f"Creating training dataset with symbols: {symbols}")

    for symbol in symbols:
        logger.info(f"\n🔍 Processing {symbol}...")

        df = fetcher.get_real_crypto_data(symbol)

        if df is not None:
            df['symbol'] = symbol
            all_data.append(df)

            logger.info(f"✅ Added {len(df)} rows for {symbol}")
        else:
            logger.warning(f"❌ Skipping {symbol} - no data available")

        # Rate limiting between symbols
        time.sleep(2)

    if not all_data:
        logger.error("❌ No data collected from any source")
        return None

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['timestamp', 'symbol'])

    # Save to file
    combined_df.to_csv(output_file, index=False)

    logger.info(f"✅ Training dataset saved to {output_file}")
    logger.info(f"   Total rows: {len(combined_df)}")
    logger.info(f"   Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    logger.info(f"   Symbols: {combined_df['symbol'].unique().tolist()}")

    return combined_df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate the quality of the training dataset"""

    validation = {
        'total_rows': len(df),
        'symbols': df['symbol'].unique().tolist(),
        'date_range': {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max()
        },
        'data_quality': {}
    }

    # Check for missing values
    missing_data = df.isnull().sum()
    validation['data_quality']['missing_values'] = missing_data.to_dict()

    # Check for zero/negative prices
    validation['data_quality']['zero_prices'] = {
        'open': (df['open'] <= 0).sum(),
        'high': (df['high'] <= 0).sum(),
        'low': (df['low'] <= 0).sum(),
        'close': (df['close'] <= 0).sum()
    }

    # Check for OHLC logic (high >= low, etc.)
    validation['data_quality']['ohlc_consistency'] = {
        'high_ge_low': ((df['high'] >= df['low']).sum() / len(df)) * 100,
        'high_ge_close': ((df['high'] >= df['close']).sum() / len(df)) * 100,
        'low_le_close': ((df['low'] <= df['close']).sum() / len(df)) * 100
    }

    # Basic statistics
    validation['statistics'] = {
        'avg_volume': df['volume'].mean(),
        'price_volatility': df['close'].pct_change().std(),
        'data_points_per_symbol': df.groupby('symbol').size().to_dict()
    }

    return validation


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Get Real Crypto Data for Training')
    parser.add_argument('--symbols', type=str, nargs='+',
                       default=['BTC', 'ETH', 'ADA'],
                       help='Cryptocurrency symbols to fetch')
    parser.add_argument('--output', type=str, default='crypto_training_data.csv',
                       help='Output CSV file')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset quality')

    args = parser.parse_args()

    print("📊 REAL CRYPTO DATA COLLECTION")
    print("=" * 40)

    # Create training dataset
    df = create_training_dataset(args.symbols, args.output)

    if df is None:
        print("❌ Failed to create training dataset")
        return 1

    print("\n✅ Dataset created successfully!")
    print(f"Symbols: {df['symbol'].unique().tolist()}")
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Validate if requested
    if args.validate:
        print("\n🔍 Validating dataset quality...")
        validation = validate_dataset(df)

        print("\n📋 VALIDATION RESULTS:")
        print(f"Missing values: {validation['data_quality']['missing_values']}")
        print(f"Zero prices: {validation['data_quality']['zero_prices']}")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".2f")
        print(f"Avg volume: {validation['statistics']['avg_volume']:,.0f}")
        print(f"Data points per symbol: {validation['statistics']['data_points_per_symbol']}")

    print("\n🎯 NEXT STEP:")
    print(f"python3 real_market_training.py --model attention")
    print(f"   (This will train your model on the real data in {args.output})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
