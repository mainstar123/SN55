import time
from typing import Tuple, Dict, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import bittensor as bt

from precog.protocol import Challenge
from precog.utils.cm_data import CMData

# Configure logging
logger = logging.getLogger(__name__)

# Global models (loaded once)
point_model = None
interval_model = None
xgb_model = None
scaler = None

# Feature scalers for different feature groups
price_scaler = None
volatility_scaler = None
momentum_scaler = None
temporal_scaler = None

# Model loading state
models_loaded = False

class GRUPriceForecaster(nn.Module):
    """GRU model for 1-hour BTC price point forecast"""

    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len=60, features=10)
        out, _ = self.gru(x)
        # Take last output and apply dropout
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class EnhancedGRUPriceForecaster(nn.Module):
    """Enhanced GRU with attention mechanism for better temporal feature extraction"""

    def __init__(self, input_size=10, hidden_size=128, num_layers=3, dropout=0.3, num_heads=8):
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
        # x: (batch, seq_len=60, features=10)

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


class QuantileIntervalForecaster(nn.Module):
    """Quantile regression for 90% confidence intervals (5th-95th percentile)"""

    def __init__(self, input_size=10, hidden_size=32, num_layers=1, dropout=0.1):
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


def add_advanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators and features"""
    df = data.copy()
    prices = df['price'].values
    timestamps = df.index

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
        hour = timestamps[i].hour
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


def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index)"""
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI

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


def extract_features(cm_data: CMData, lookback_minutes: int = 60) -> torch.Tensor:
    """
    Extract 10 features from CoinMetrics historical data

    Features:
    0-3: Price returns (1m, 5m, 15m, 60m)
    4-5: Volatility (rolling std 15m, 60m)
    6: RSI (14-period)
    7: Order flow proxy (simplified)
    8-9: Temporal features (hour sin/cos)
    """
    global price_scaler, volatility_scaler, momentum_scaler, temporal_scaler

    try:
        # Fetch last 60 minutes of 1-minute BTC prices
        end_time = pd.Timestamp.now()
        start_time = end_time - pd.Timedelta(minutes=lookback_minutes)

        # Get price data from CoinMetrics
        price_data = cm_data.get_CM_ReferenceRate(
            assets=['btc'],
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            frequency="1m"
        )

        if price_data.empty or 'ReferenceRateUSD' not in price_data.columns:
            bt.logging.warning("No price data available from CoinMetrics, using fallback")
            return torch.randn(1, 60, 10)  # Fallback random features

        prices = price_data['ReferenceRateUSD'].values
        timestamps = pd.to_datetime(price_data['time'])

        if len(prices) < 60:
            bt.logging.warning(f"Insufficient data points: {len(prices)}, padding with last value")
            # Pad with last price if insufficient data
            last_price = prices[-1] if len(prices) > 0 else 50000.0
            prices = np.full(60, last_price)
            timestamps = pd.date_range(end=end_time, periods=60, freq='1min')

        # Extract features for each minute
        features = []
        for i in range(len(prices)):
            if i < 60:  # Ensure we have at least 60 minutes of data
                # Price returns
                ret_1m = (prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0
                ret_5m = (prices[i] - prices[i-5]) / prices[i-5] if i >= 5 else 0
                ret_15m = (prices[i] - prices[i-15]) / prices[i-15] if i >= 15 else 0
                ret_60m = (prices[i] - prices[i-60]) / prices[i-60] if i >= 60 else 0

                # Volatility
                vol_15m = np.std(prices[max(0, i-15):i+1]) if i > 0 else 0
                vol_60m = np.std(prices[max(0, i-60):i+1]) if i > 0 else 0

                # RSI
                rsi = calculate_rsi(prices[:i+1])

                # Order flow proxy (simplified - could be enhanced with real order book data)
                order_flow = 0.0  # Placeholder

                # Temporal features (cyclical encoding)
                hour = timestamps[i].hour
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)

                feature_vector = [ret_1m, ret_5m, ret_15m, ret_60m,
                                vol_15m, vol_60m, rsi, order_flow, hour_sin, hour_cos]
                features.append(feature_vector)

        # Ensure we have exactly 60 time steps
        if len(features) > 60:
            features = features[-60:]
        elif len(features) < 60:
            # Pad with last feature if insufficient
            last_feature = features[-1] if features else [0] * 10
            features.extend([last_feature] * (60 - len(features)))

        features_array = np.array(features)

        # Scale features if scalers are available
        if price_scaler is not None:
            # Apply feature group scaling
            scaled_features = features_array.copy()

            # Price returns (indices 0-3)
            scaled_features[:, 0:4] = price_scaler.transform(features_array[:, 0:4])

            # Volatility (indices 4-5)
            scaled_features[:, 4:6] = volatility_scaler.transform(features_array[:, 4:6])

            # Momentum (RSI, index 6)
            scaled_features[:, 6:7] = momentum_scaler.transform(features_array[:, 6:7])

            # Temporal (indices 8-9)
            scaled_features[:, 8:10] = temporal_scaler.transform(features_array[:, 8:10])

            features_array = scaled_features

        return torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)  # (1, 60, 10)

    except Exception as e:
        bt.logging.error(f"Error extracting features: {e}")
        return torch.randn(1, 60, 10)  # Fallback


def load_models():
    """Load trained models and scalers"""
    global point_model, interval_model, xgb_model, scaler
    global price_scaler, volatility_scaler, momentum_scaler, temporal_scaler
    global models_loaded

    if models_loaded:
        return

    try:
        # Load GRU point forecast model
        if point_model is None:
            point_model = GRUPriceForecaster()
            model_path = 'models/gru_point.pth'
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                point_model.load_state_dict(checkpoint)
                point_model.eval()
                bt.logging.info(f"Loaded GRU point model from {model_path}")
            except FileNotFoundError:
                bt.logging.warning(f"GRU point model not found at {model_path}, using untrained model")
            except Exception as e:
                bt.logging.error(f"Error loading GRU point model: {e}")

        # Load LSTM interval model
        if interval_model is None:
            interval_model = QuantileIntervalForecaster()
            model_path = 'models/quantile_interval.pth'
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                interval_model.load_state_dict(checkpoint)
                interval_model.eval()
                bt.logging.info(f"Loaded quantile interval model from {model_path}")
            except FileNotFoundError:
                bt.logging.warning(f"Interval model not found at {model_path}, using untrained model")
            except Exception as e:
                bt.logging.error(f"Error loading interval model: {e}")

        # Load scalers
        try:
            import joblib
            price_scaler = joblib.load('models/price_scaler.pkl')
            volatility_scaler = joblib.load('models/volatility_scaler.pkl')
            momentum_scaler = joblib.load('models/momentum_scaler.pkl')
            temporal_scaler = joblib.load('models/temporal_scaler.pkl')
            bt.logging.info("Loaded feature scalers")
        except FileNotFoundError:
            bt.logging.warning("Feature scalers not found, using unscaled features")
        except Exception as e:
            bt.logging.error(f"Error loading scalers: {e}")

        models_loaded = True
        bt.logging.success("Model loading completed")

    except Exception as e:
        bt.logging.error(f"Error in load_models: {e}")


def detect_market_regime(features: torch.Tensor) -> str:
    """Detect current market regime for model weighting"""
    try:
        # Simple regime detection based on recent volatility and trend
        recent_returns = features[0, -10:, 3].mean().item()  # 60-min returns average
        recent_volatility = features[0, -10:, 5].mean().item()  # 60-min volatility average

        if recent_volatility > 0.02:  # High volatility threshold
            return 'high_volatility'
        elif recent_returns > 0.005:  # Bullish trend
            return 'bull_stable'
        elif recent_returns < -0.005:  # Bearish trend
            return 'bear_stable'
        else:
            return 'sideways'

    except Exception as e:
        bt.logging.error(f"Error detecting market regime: {e}")
        return 'sideways'


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """
    Custom forward function for Precog miner using GRU/LSTM ensemble

    Target: MAPE <0.09%, Coverage >85%, RMSE <77
    """
    start_time = time.perf_counter()

    # Load models if not already loaded
    load_models()

    try:
        # Get assets to predict
        raw_assets = synapse.assets if hasattr(synapse, "assets") else ["btc"]
        assets = [asset.lower() for asset in raw_assets]

        predictions = {}
        intervals = {}

        for asset in assets:
            if asset != "btc":
                bt.logging.warning(f"Asset {asset} not supported, skipping")
                continue

            # Extract features
            features = extract_features(cm, lookback_minutes=60)
            bt.logging.debug(f"Extracted features shape: {features.shape}")

            # Detect market regime
            regime = detect_market_regime(features)
            bt.logging.debug(f"Detected regime: {regime}")

            # Generate predictions
            with torch.no_grad():
                # Point forecast (ensemble of GRU)
                if point_model is not None:
                    point_pred = point_model(features).item()
                else:
                    # Fallback: use last known price + small adjustment
                    point_pred = 50000.0  # Fallback price
                    bt.logging.warning("Using fallback point prediction")

                # Interval forecast (LSTM quantile regression)
                if interval_model is not None:
                    lower_raw, upper_raw = interval_model(features)
                    lower_bound = lower_raw.item()
                    upper_bound = upper_raw.item()

                    # Ensure interval covers point prediction
                    current_price = point_pred
                    lower_bound = min(lower_bound, current_price * 0.99)  # At least 1% below
                    upper_bound = max(upper_bound, current_price * 1.01)  # At least 1% above
                else:
                    # Fallback interval: ±3%
                    margin = point_pred * 0.03
                    lower_bound = point_pred - margin
                    upper_bound = point_pred + margin
                    bt.logging.warning("Using fallback interval prediction")

            # Apply regime-specific adjustments
            if regime == 'high_volatility':
                # Widen intervals for volatile markets
                interval_width = upper_bound - lower_bound
                upper_bound += interval_width * 0.2
                lower_bound -= interval_width * 0.2
            elif regime == 'bull_stable':
                # Slightly optimistic bias for bull markets
                point_pred *= 1.002

            predictions[asset] = point_pred
            intervals[asset] = [lower_bound, upper_bound]

            # Log prediction
            bt.logging.info(
                f"{asset} | Point: ${point_pred:.2f} | Interval: [${lower_bound:.2f}, ${upper_bound:.2f}] | Regime: {regime}"
            )

            # Log to file for monitoring
            try:
                with open('logs/predictions.log', 'a') as f:
                    timestamp = pd.Timestamp.now().isoformat()
                    f.write(f"{timestamp},{asset},{point_pred},{lower_bound},{upper_bound},{regime}\n")
            except Exception as e:
                bt.logging.error(f"Error logging prediction: {e}")

        synapse.predictions = predictions
        synapse.intervals = intervals

        # Performance monitoring
        inference_time = time.perf_counter() - start_time
        bt.logging.debug(f"Inference time: {inference_time:.3f}s")

        if predictions:
            bt.logging.success(f"Custom model predictions complete for {list(predictions.keys())}")
        else:
            bt.logging.warning("No predictions generated")

        return synapse

    except Exception as e:
        bt.logging.error(f"Error in custom forward function: {e}")
        # Fallback to basic prediction
        fallback_price = 50000.0
        synapse.predictions = {"btc": fallback_price}
        synapse.intervals = {"btc": [fallback_price * 0.97, fallback_price * 1.03]}
        return synapse
