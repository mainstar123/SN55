#!/usr/bin/env python3
"""
Simplified local backtesting for Precog model performance validation

Bypasses Bittensor imports to focus on core ML performance testing.
Tests if your model can achieve "first place" level accuracy locally.
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define model classes directly to avoid Bittensor dependencies

# Simple ensemble implementation (matching train_models.py)
class EnsembleForecaster:
    """Simple ensemble combining GRU predictions with statistical methods"""

    def __init__(self, gru_model=None, weights=None):
        self.gru_model = gru_model
        self.weights = weights or {'gru': 0.7, 'ma': 0.2, 'naive': 0.1}

    def predict_ensemble(self, features, recent_prices):
        """Generate ensemble prediction"""
        predictions = {}

        # GRU prediction
        if self.gru_model is not None:
            with torch.no_grad():
                predictions['gru'] = self.gru_model(features).item()

        # Moving average prediction
        if len(recent_prices) >= 60:
            ma_pred = recent_prices[-60:].mean()
            # Convert to return (relative to current price)
            current_price = recent_prices[-1]
            predictions['ma'] = (ma_pred - current_price) / current_price
        else:
            predictions['ma'] = 0.0

        # Naive prediction (momentum)
        if len(recent_prices) >= 2:
            recent_return = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]
            predictions['naive'] = recent_return * 0.5  # Dampened momentum
        else:
            predictions['naive'] = 0.0

        # Weighted ensemble
        ensemble_pred = sum(predictions[key] * self.weights.get(key, 0)
                          for key in predictions.keys())

        return ensemble_pred

class GRUPriceForecaster(nn.Module):
    """GRU model for 1-hour BTC price point forecast"""

    def __init__(self, input_size=10, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len=60, features=24)
        out, _ = self.gru(x)
        # Take last output and apply dropout
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class QuantileIntervalForecaster(nn.Module):
    """Quantile regression for 90% confidence intervals (5th-95th percentile)"""

    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.q05 = nn.Linear(hidden_size, 1)  # Lower bound (5th percentile)
        self.q95 = nn.Linear(hidden_size, 1)  # Upper bound (95th percentile)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        lower = self.q05(out)
        upper = self.q95(out)
        return lower, upper


class EnhancedGRUPriceForecaster(nn.Module):
    """Enhanced GRU with attention mechanism for better temporal feature extraction"""

    def __init__(self, input_size=24, hidden_size=128, num_layers=3, dropout=0.3, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Multi-layer GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Multi-head attention for temporal feature weighting
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        # Enhanced prediction head
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 4)

    def forward(self, x):
        # x: (batch, seq_len=60, features=24)

        # GRU encoding
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size)

        # Apply attention mechanism
        # Use the last timestep as query, all timesteps as key/value
        query = gru_out[:, -1:, :]  # (batch, 1, hidden_size)
        key_value = gru_out  # (batch, seq_len, hidden_size)

        attn_out, attn_weights = self.attention(query, key_value, key_value)
        # attn_out: (batch, 1, hidden_size)

        # Layer normalization
        out = self.layer_norm(attn_out.squeeze(1))  # (batch, hidden_size)
        out = self.dropout(out)

        # Enhanced prediction head with residual connections
        residual = out

        out = F.relu(self.fc1(out))
        out = self.batch_norm1(out)
        out = self.dropout(out)

        out = F.relu(self.fc2(out))
        out = self.batch_norm2(out)
        out = self.dropout(out)

        # Residual connection
        if out.shape[-1] == residual.shape[-1]:
            out = out + residual

        out = self.fc3(out)
        return out


def calculate_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    if len(prices) < period + 1:
        return 50.0

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_parkinson_volatility(prices: np.ndarray, window: int = 60) -> np.ndarray:
    """Calculate Parkinson volatility using high-low range"""
    if len(prices) < window:
        return np.full(len(prices), np.nan)

    volatility = np.full(len(prices), np.nan)

    for i in range(window, len(prices)):
        window_prices = prices[i-window:i]
        # Simplified Parkinson (normally needs OHLC data)
        returns = np.diff(window_prices) / window_prices[:-1]
        volatility[i] = np.sqrt(np.sum(returns**2) / len(returns)) * np.sqrt(252)  # Annualized

    return volatility


def calculate_garman_klass_volatility(prices: np.ndarray, window: int = 60) -> np.ndarray:
    """Calculate Garman-Klass volatility (simplified version)"""
    if len(prices) < window:
        return np.full(len(prices), np.nan)

    volatility = np.full(len(prices), np.nan)

    for i in range(window, len(prices)):
        window_prices = prices[i-window:i]
        returns = np.diff(window_prices) / window_prices[:-1]
        volatility[i] = np.std(returns) * np.sqrt(252)

    return volatility


def calculate_true_strength_index(prices: np.ndarray, r: int = 25, s: int = 13) -> np.ndarray:
    """Calculate True Strength Index (TSI)"""
    if len(prices) < r + s:
        return np.full(len(prices), np.nan)

    tsi = np.full(len(prices), np.nan)

    # Price change
    pc = np.diff(prices)
    pc = np.concatenate([[0], pc])  # Pad with zero

    # Double smoothed PC
    pc_smooth1 = pd.Series(pc).ewm(span=r).mean().values
    pc_smooth2 = pd.Series(pc_smooth1).ewm(span=s).mean().values

    # Absolute PC
    apc = np.abs(pc)
    apc_smooth1 = pd.Series(apc).ewm(span=r).mean().values
    apc_smooth2 = pd.Series(apc_smooth1).ewm(span=s).mean().values

    # TSI
    valid_idx = ~np.isnan(pc_smooth2) & ~np.isnan(apc_smooth2) & (apc_smooth2 != 0)
    tsi[valid_idx] = 100 * (pc_smooth2[valid_idx] / apc_smooth2[valid_idx])

    return tsi


def calculate_kaufman_adaptive_ma(prices: np.ndarray, period: int = 30) -> np.ndarray:
    """Calculate Kaufman Adaptive Moving Average (KAMA)"""
    if len(prices) < period:
        return np.full(len(prices), np.nan)

    kama = np.full(len(prices), np.nan)
    kama[period-1] = prices[period-1]  # Initial value

    for i in range(period, len(prices)):
        # Efficiency Ratio
        change = abs(prices[i] - prices[i-period])
        volatility = sum(abs(prices[j] - prices[j-1]) for j in range(i-period+1, i+1))
        er = change / volatility if volatility != 0 else 0

        # Smoothing constant
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2

        # KAMA
        kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])

    return kama


def calculate_trix(prices: np.ndarray, period: int = 15) -> np.ndarray:
    """Calculate TRIX (Triple Exponential Average)"""
    if len(prices) < period * 3:
        return np.full(len(prices), np.nan)

    # Triple EMA
    ema1 = pd.Series(prices).ewm(span=period).mean()
    ema2 = ema1.ewm(span=period).mean()
    ema3 = ema2.ewm(span=period).mean()

    # TRIX
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    return trix.values


def calculate_price_entropy(prices: np.ndarray, window: int = 60) -> np.ndarray:
    """Calculate price entropy as a complexity measure"""
    if len(prices) < window:
        return np.full(len(prices), np.nan)

    entropy = np.full(len(prices), np.nan)

    for i in range(window, len(prices)):
        window_prices = prices[i-window:i]
        returns = np.diff(window_prices) / window_prices[:-1]

        # Discretize returns into bins
        bins = np.linspace(-0.02, 0.02, 21)  # 2% return bins
        hist, _ = np.histogram(returns, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities

        if len(hist) > 0:
            entropy[i] = -np.sum(hist * np.log2(hist))

    return entropy


def calculate_schaff_trend_cycle(prices: np.ndarray, cycle: int = 10, smooth: int = 3) -> np.ndarray:
    """Calculate Schaff Trend Cycle"""
    if len(prices) < cycle * 2:
        return np.full(len(prices), np.nan)

    stc = np.full(len(prices), np.nan)

    # MACD-like calculation
    ema1 = pd.Series(prices).ewm(span=cycle).mean()
    ema2 = pd.Series(prices).ewm(span=cycle*2).mean()
    macd = ema1 - ema2

    # Cycle values
    cycle_max = macd.rolling(cycle).max()
    cycle_min = macd.rolling(cycle).min()

    # Stochastic of MACD
    k = 100 * (macd - cycle_min) / (cycle_max - cycle_min)

    # Smoothed K
    d = k.rolling(smooth).mean()

    # Schaff Trend Cycle
    stc_max = d.rolling(cycle).max()
    stc_min = d.rolling(cycle).min()
    valid = ~np.isnan(stc_max) & ~np.isnan(stc_min) & (stc_max != stc_min)
    stc[valid] = 100 * (d[valid] - stc_min[valid]) / (stc_max[valid] - stc_min[valid])

    return stc


def calculate_coppock_curve(prices: np.ndarray, wma_period: int = 14, roc1: int = 11, roc2: int = 14) -> np.ndarray:
    """Calculate Coppock Curve"""
    if len(prices) < max(wma_period, roc1, roc2):
        return np.full(len(prices), np.nan)

    # Rate of change
    roc1_values = 100 * (prices - np.roll(prices, roc1)) / np.roll(prices, roc1)
    roc2_values = 100 * (prices - np.roll(prices, roc2)) / np.roll(prices, roc2)

    # Sum of ROCs
    roc_sum = roc1_values + roc2_values

    # Weighted moving average
    weights = np.arange(1, wma_period + 1)
    wma = np.full(len(prices), np.nan)

    for i in range(wma_period - 1, len(prices)):
        window = roc_sum[i-wma_period+1:i+1]
        if not np.any(np.isnan(window)):
            wma[i] = np.sum(window * weights) / np.sum(weights)

    return wma


def calculate_volume_price_trend(prices: np.ndarray) -> np.ndarray:
    """Calculate Volume Price Trend (VPT) - synthetic volume"""
    # Create synthetic volume based on price movements
    returns = np.diff(prices) / prices[:-1]
    volume = np.abs(returns) * 1000 + 100  # Base volume + volatility component

    vpt = np.zeros(len(prices))
    vpt[0] = volume[0]

    for i in range(1, len(prices)):
        vpt[i] = vpt[i-1] + volume[i-1] * returns[i-1]

    return vpt


def calculate_ease_of_movement(prices: np.ndarray, box_ratio: float = 0.1) -> np.ndarray:
    """Calculate Ease of Movement (EMV)"""
    if len(prices) < 2:
        return np.full(len(prices), np.nan)

    # Price change
    price_change = np.diff(prices)

    # Box ratio (simplified)
    box = prices * box_ratio

    # Midpoint move
    midpoint_move = (prices[1:] + prices[:-1]) / 2 - (prices[:-1] + np.roll(prices, -1)[:-1]) / 2

    # Distance moved
    distance = np.abs(midpoint_move)

    # EMV
    emv = np.full(len(prices), np.nan)
    valid = box[:-1] != 0
    emv[1:][valid] = (price_change[valid] / box[:-1][valid]) * (distance[valid] / box[:-1][valid])

    return emv


def calculate_fractal_dimension(prices: np.ndarray, window: int = 60) -> np.ndarray:
    """Calculate fractal dimension using Higuchi's method (simplified)"""
    if len(prices) < window:
        return np.full(len(prices), np.nan)

    fd = np.full(len(prices), np.nan)

    for i in range(window, len(prices)):
        window_prices = prices[i-window:i]

        # Simplified fractal dimension calculation
        # Count how many times the price changes direction
        returns = np.diff(window_prices)
        direction_changes = np.sum(np.diff(np.sign(returns)) != 0)

        if direction_changes > 0:
            # Rough estimate of fractal dimension
            fd[i] = 1 + np.log(direction_changes) / np.log(window)

    return fd


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators and features"""
    prices = df['price'].values

    # Basic price features
    returns = np.zeros_like(prices)
    returns[1:] = np.diff(prices) / prices[:-1]

    # Add basic technical indicators
    for i in range(len(df)):
        # Price returns at different lags
        df.loc[df.index[i], 'ret_1m'] = returns[i] if i > 0 else 0
        df.loc[df.index[i], 'ret_5m'] = (prices[i] - prices[i-5]) / prices[i-5] if i >= 5 else 0
        df.loc[df.index[i], 'ret_15m'] = (prices[i] - prices[i-15]) / prices[i-15] if i >= 15 else 0
        df.loc[df.index[i], 'ret_60m'] = (prices[i] - prices[i-60]) / prices[i-60] if i >= 60 else 0

        # Volatility measures
        df.loc[df.index[i], 'vol_15m'] = np.std(returns[max(0, i-15):i+1]) if i > 0 else 0
        df.loc[df.index[i], 'vol_60m'] = np.std(returns[max(0, i-60):i+1]) if i > 0 else 0

        # RSI
        recent_prices = prices[max(0, i-14):i+1]
        if len(recent_prices) >= 2:
            df.loc[df.index[i], 'rsi'] = calculate_rsi(recent_prices)
        else:
            df.loc[df.index[i], 'rsi'] = 50.0

        # Temporal features
        hour = df.index[i].hour
        df.loc[df.index[i], 'hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df.loc[df.index[i], 'hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # Order flow proxy (simplified)
        df.loc[df.index[i], 'order_flow'] = 0.0

    # Enhanced volatility measures
    df['realized_volatility'] = df['price'].pct_change().rolling(60).std()
    df['parkinson_volatility'] = calculate_parkinson_volatility(prices)
    df['garman_klass_volatility'] = calculate_garman_klass_volatility(prices)

    # Advanced momentum indicators
    df['tsi'] = calculate_true_strength_index(prices)
    df['kama'] = calculate_kaufman_adaptive_ma(prices)
    df['trix'] = calculate_trix(prices)

    # Statistical features
    df['price_skewness'] = df['price'].rolling(60).skew()
    df['price_kurtosis'] = df['price'].rolling(60).kurt()
    df['price_entropy'] = calculate_price_entropy(prices)

    # Cycle indicators
    df['schaff_trend_cycle'] = calculate_schaff_trend_cycle(prices)
    df['coppock_curve'] = calculate_coppock_curve(prices)

    # Volume-based features (synthetic for now)
    df['volume_price_trend'] = calculate_volume_price_trend(prices)
    df['ease_of_movement'] = calculate_ease_of_movement(prices)

    # Fractal dimension (complexity measure)
    df['fractal_dimension'] = calculate_fractal_dimension(prices)

    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

    return df


def extract_features(prices, lookback=60):
    """
    Extract 24 enhanced features from price data

    Features:
    0-9: Basic features (returns, volatility, RSI, temporal)
    10-12: Advanced volatility measures
    13-15: Momentum indicators
    16-18: Statistical features
    19-20: Cycle indicators
    21-23: Volume-based and complexity features
    """
    if len(prices) < lookback:
        # Pad with last price if insufficient data
        last_price = prices[-1] if len(prices) > 0 else 50000.0
        prices = np.full(lookback, last_price)
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=lookback, freq='1min')
    else:
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=len(prices), freq='1min')

    # Create DataFrame for enhanced feature extraction
    df = pd.DataFrame({'price': prices}, index=timestamps[:len(prices)])

    # Add advanced features
    df = add_advanced_features(df)

    # Extract feature columns (24 features)
    feature_columns = [
        'ret_1m', 'ret_5m', 'ret_15m', 'ret_60m',
        'vol_15m', 'vol_60m', 'rsi', 'order_flow', 'hour_sin', 'hour_cos',
        'realized_volatility', 'parkinson_volatility', 'garman_klass_volatility',
        'tsi', 'kama', 'trix', 'price_skewness', 'price_kurtosis', 'price_entropy',
        'schaff_trend_cycle', 'coppock_curve', 'volume_price_trend', 'ease_of_movement', 'fractal_dimension'
    ]

    features = df[feature_columns].values[-lookback:]
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def load_models():
    """Load trained models and scalers"""
    point_model = None
    interval_model = None
    scaler = None

    try:
        # Load point model (Enhanced GRU with attention)
        if os.path.exists('models/gru_point.pth'):
            point_model = EnhancedGRUPriceForecaster()
            checkpoint = torch.load('models/gru_point.pth', map_location='cpu')
            point_model.load_state_dict(checkpoint)
            point_model.eval()
            logger.info("Loaded Enhanced GRU point model with attention")

            # Try to load ensemble configuration
            ensemble = None
            if os.path.exists('models/ensemble_config.pkl'):
                try:
                    import pickle
                    with open('models/ensemble_config.pkl', 'rb') as f:
                        ensemble_config = pickle.load(f)
                    ensemble = EnsembleForecaster(gru_model=point_model, weights=ensemble_config['weights'])
                    logger.info("Loaded ensemble model with weights: {}".format(ensemble.weights))
                except Exception as e:
                    logger.warning(f"Could not load ensemble config: {e}")
                    ensemble = EnsembleForecaster(gru_model=point_model)
        else:
            logger.warning("No trained point model found")
            return None, None, None

        # Try to load interval model (may fail due to architecture issues)
        try:
            if os.path.exists('models/quantile_interval.pth'):
                interval_model = QuantileIntervalForecaster()
                checkpoint = torch.load('models/quantile_interval.pth', map_location='cpu')
                interval_model.load_state_dict(checkpoint)
                interval_model.eval()
                logger.info("Loaded quantile interval model")
            else:
                logger.warning("No trained interval model found - using dummy intervals")
        except Exception as e:
            logger.warning(f"Failed to load interval model ({e}) - using dummy intervals")
            interval_model = None

        # Load feature scaler (critical for proper inference)
        if os.path.exists('models/feature_scaler.pkl'):
            import joblib
            scaler = joblib.load('models/feature_scaler.pkl')
            logger.info("Loaded feature scaler")
        else:
            logger.warning("No feature scaler found - features will not be scaled")

        return point_model, interval_model, scaler

    except Exception as e:
        logger.error(f"Error loading point model: {e}")
        return None, None, None


def create_mock_data(days=30):
    """Create mock BTC price data for testing"""
    logger.info(f"Creating mock {days}-day BTC price dataset...")

    # Generate realistic BTC-like price series
    timestamps = pd.date_range(start=datetime.now() - timedelta(days=days),
                              end=datetime.now(), freq='1min')

    # Base price with trend and volatility
    base_price = 50000
    trend = np.linspace(0, 2000, len(timestamps))  # Slight upward trend
    volatility = np.random.choice([0.001, 0.002, 0.005], len(timestamps), p=[0.6, 0.3, 0.1])
    noise = np.random.normal(0, 1, len(timestamps))

    prices = base_price + trend + (noise * volatility * base_price)

    # Ensure positive prices
    prices = np.maximum(prices, 10000)

    df = pd.DataFrame({'price': prices}, index=timestamps)
    logger.info(f"Created {len(df)} price points")
    return df


def run_backtest(test_days=7, use_mock_data=True):
    """Run comprehensive backtest"""
    logger.info("🚀 Starting Precog Model Backtest")
    logger.info("=" * 50)

    # Load models and scaler
    point_model, interval_model, scaler = load_models()
    if point_model is None:
        logger.error("❌ Failed to load point model")
        return None
    if interval_model is None:
        logger.warning("⚠️ Failed to load interval model - using dummy intervals")
        interval_model = None

    # Get test data
    if use_mock_data:
        logger.info("Using mock data for testing...")
        data = create_mock_data(days=30)
    else:
        # Try to load real data
        data_file = 'data/btc_1m_train.csv'
        if os.path.exists(data_file):
            data = pd.read_csv(data_file, index_col=0, parse_dates=True)
            data.index = pd.to_datetime(data.index)
            logger.info(f"Loaded real data: {len(data)} points")
        else:
            logger.warning("No real data found, using mock data")
            data = create_mock_data(days=30)

    # Prepare test period
    test_end = data.index[-1]
    test_start = test_end - timedelta(days=test_days)

    test_data = data[(data.index >= test_start) & (data.index <= test_end)]
    logger.info(f"Testing on {len(test_data)} minutes ({test_days} days)")

    if len(test_data) < 120:
        logger.error("Insufficient test data")
        return None

    # Storage for results
    predictions = []
    actuals = []
    intervals = []
    response_times = []

    # Test every 60 minutes (simulate prediction requests)
    test_indices = range(60, len(test_data) - 60, 60)

    logger.info("Running predictions...")
    for i in test_indices:
        start_time = time.perf_counter()

        try:
            # Get price window for feature extraction
            price_window = test_data.iloc[i-60:i]['price'].values

            # Extract features
            features = extract_features(price_window)

            # Apply feature scaling (critical for proper inference)
            if scaler is not None:
                # Reshape for scaler (batch_size=1, seq_len=60, features=24)
                features_np = features.numpy()
                batch_size, seq_len, n_features = features_np.shape
                features_2d = features_np.reshape(batch_size * seq_len, n_features)
                scaled_features_2d = scaler.transform(features_2d)
                scaled_features = scaled_features_2d.reshape(batch_size, seq_len, n_features)
                features = torch.tensor(scaled_features, dtype=torch.float32)
            else:
                logger.warning("No scaler available - using unscaled features")

            # Make predictions (on CPU for backtesting)
            device = torch.device('cpu')
            features = features.to(device)

            with torch.no_grad():
                # Models predict RETURNS, not absolute prices
                if ensemble is not None:
                    # Use ensemble prediction
                    recent_prices = price_window[-60:] if len(price_window) >= 60 else price_window
                    point_return_pred = ensemble.predict_ensemble(features, recent_prices)
                    logger.debug(f"Ensemble prediction: {point_return_pred}")
                else:
                    # Use single model prediction
                    point_return_pred = point_model(features).item()

                if interval_model is not None:
                    lower_return, upper_return = interval_model(features)
                    lower_bound_return = lower_return.item()
                    upper_bound_return = upper_return.item()
                else:
                    # Dummy interval predictions (simple confidence bounds)
                    lower_bound_return = point_return_pred - 0.01  # -1%
                    upper_bound_return = point_return_pred + 0.01  # +1%

            response_time = time.perf_counter() - start_time

            # Get current price and actual price 1 hour later
            current_price = test_data.iloc[i]['price']
            actual_price = test_data.iloc[i + 60]['price']

            # Convert return predictions back to price predictions
            point_pred = current_price * (1 + point_return_pred)
            lower_bound = current_price * (1 + lower_bound_return)
            upper_bound = current_price * (1 + upper_bound_return)

            # Store results
            predictions.append(point_pred)
            actuals.append(actual_price)
            intervals.append([lower_bound, upper_bound])
            response_times.append(response_time)

        except Exception as e:
            logger.error(f"Error in prediction {i}: {e}")
            continue

    if not predictions:
        logger.error("No predictions generated")
        return None

    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    intervals = np.array(intervals)
    response_times = np.array(response_times)

    # Point forecast metrics
    mape = mean_absolute_percentage_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = np.mean(np.abs(predictions - actuals))

    # Interval coverage (90% confidence interval)
    lower_bounds = intervals[:, 0]
    upper_bounds = intervals[:, 1]
    coverages = (lower_bounds <= actuals) & (actuals <= upper_bounds)
    coverage_rate = np.mean(coverages)

    # Interval width
    avg_width = np.mean((upper_bounds - lower_bounds) / actuals)

    # Response time
    avg_response_time = np.mean(response_times)
    max_response_time = np.max(response_times)

    # Performance assessment
    results = {
        'sample_size': len(predictions),
        'test_period_days': test_days,
        'use_mock_data': use_mock_data,

        # Point forecast metrics
        'mape': mape,
        'mape_percent': mape * 100,
        'rmse': rmse,
        'mae': mae,

        # Interval metrics
        'coverage_rate': coverage_rate,
        'coverage_percent': coverage_rate * 100,
        'avg_interval_width_percent': avg_width * 100,

        # Performance metrics
        'avg_response_time': avg_response_time,
        'max_response_time': max_response_time,

        # First place assessment
        'first_place_ready': (
            mape < 0.0008 and  # <0.08%
            rmse < 70 and      # <70 RMSE
            coverage_rate > 0.92  # >92% coverage
        ),

        # Target achievement
        'mape_target_met': mape < 0.0008,
        'rmse_target_met': rmse < 70,
        'coverage_target_met': coverage_rate > 0.92,
        'response_target_met': max_response_time < 3.0
    }

    return results


def print_results(results):
    """Print formatted results"""
    print("\n" + "="*70)
    print("🎯 PRECOG MODEL PERFORMANCE ASSESSMENT")
    print("="*70)

    print(f"📊 Sample Size: {results['sample_size']} predictions")
    print(f"📅 Test Period: {results['test_period_days']} days")
    print(f"🎭 Data Type: {'Mock' if results['use_mock_data'] else 'Real'}")
    print()

    print("📈 POINT FORECAST PERFORMANCE:")
    print(f"  • MAPE: {results['mape_percent']:.4f}% (Target: <0.08%) {'✅' if results['mape_target_met'] else '❌'}")
    print(f"  • RMSE: ${results['rmse']:.2f} (Target: <70) {'✅' if results['rmse_target_met'] else '❌'}")
    print(f"  • MAE:  ${results['mae']:.2f}")
    print()

    print("🎯 INTERVAL FORECAST PERFORMANCE:")
    print(f"  • Coverage: {results['coverage_percent']:.1f}% (Target: >92%) {'✅' if results['coverage_target_met'] else '❌'}")
    print(f"  • Avg Width: {results['avg_interval_width_percent']:.1f}% of price")
    print()

    print("⚡ PERFORMANCE METRICS:")
    print(f"  • Avg Response Time: {results['avg_response_time']:.3f}s")
    print(f"  • Max Response Time: {results['max_response_time']:.3f}s (Target: <3.0s) {'✅' if results['response_target_met'] else '❌'}")
    print()

    print("🏆 FIRST PLACE READINESS:")
    if results['first_place_ready']:
        print("🎉 CONGRATULATIONS! Your model is FIRST PLACE READY!")
        print("   ✓ MAPE <0.08% (top-1 level)")
        print("   ✓ RMSE <70 (top-1 level)")
        print("   ✓ Coverage >92% (top-1 level)")
        print()
        print("🚀 You can proceed to testnet with confidence!")
        print("   Your model should dominate the competition.")
    else:
        print("⚠️  MODEL NEEDS IMPROVEMENT BEFORE TESTNET:")
        issues = []
        if not results['mape_target_met']:
            issues.append("- Point forecast accuracy too low (train longer or adjust architecture)")
        if not results['rmse_target_met']:
            issues.append("- RMSE too high (reduce overfitting or add regularization)")
        if not results['coverage_target_met']:
            issues.append("- Interval coverage insufficient (adjust quantile loss weights)")
        if not results['response_target_met']:
            issues.append("- Response time too slow (optimize model size)")

        for issue in issues:
            print(f"   {issue}")

        print()
        print("💡 Recommendations:")
        print("   1. Train models longer with more data")
        print("   2. Adjust hyperparameters (learning rate, hidden size)")
        print("   3. Consider ensemble approaches")
        print("   4. Add more sophisticated features")

    print("\n" + "="*70)


def main():
    """Main backtest function"""
    import argparse

    parser = argparse.ArgumentParser(description='Test Precog model for first place readiness')
    parser.add_argument('--test-days', type=int, default=7, help='Days of data to test')
    parser.add_argument('--real-data', action='store_true', help='Use real data instead of mock')

    args = parser.parse_args()

    # Run backtest
    results = run_backtest(test_days=args.test_days, use_mock_data=not args.real_data)

    if results:
        print_results(results)

        # Save results
        results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    json_results[k] = v.tolist()
                elif isinstance(v, (np.floating, np.float32, np.float64)):
                    json_results[k] = float(v)
                elif isinstance(v, (np.integer, np.int32, np.int64)):
                    json_results[k] = int(v)
                elif isinstance(v, (bool, np.bool_)):
                    json_results[k] = bool(v)
                else:
                    json_results[k] = v
            json.dump(json_results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        # Exit with appropriate code
        exit(0 if results['first_place_ready'] else 1)

    else:
        logger.error("Backtest failed")
        exit(1)


if __name__ == "__main__":
    main()
