
"""
DOMINATION FORWARD FUNCTION
Enhanced forward function with peak hour optimization and ensemble methods
"""

import time
import torch
import numpy as np
import logging
from datetime import datetime, timezone
import statistics

logger = logging.getLogger(__name__)

# Global performance tracking
performance_history = []
reward_history = []
prediction_count = 0
total_reward = 0.0
response_times = []

# Peak hour configuration
PEAK_HOURS = [9, 10, 13, 14]  # UTC hours
PEAK_FREQUENCY = 15  # minutes
NORMAL_FREQUENCY = 60  # minutes
PEAK_CONFIDENCE_THRESHOLD = 0.75
NORMAL_CONFIDENCE_THRESHOLD = 0.85

# Market regime configuration
MARKET_REGIMES = {
    'bull': {'freq': 30, 'threshold': 0.75, 'description': 'High volatility - frequent predictions'},
    'bear': {'freq': 45, 'threshold': 0.82, 'description': 'Medium volatility - balanced'},
    'volatile': {'freq': 15, 'threshold': 0.70, 'description': 'High volatility - aggressive'},
    'ranging': {'freq': 60, 'threshold': 0.85, 'description': 'Low volatility - conservative'}
}

# Volatility thresholds
VOLATILITY_THRESHOLDS = {
    'high': 0.05,    # 5% price movement
    'medium': 0.02,  # 2% price movement
    'low': 0.01      # 1% price movement
}

def get_current_hour_utc():
    """Get current hour in UTC"""
    return datetime.now(timezone.utc).hour

def is_peak_hour():
    """Check if current time is peak hour"""
    current_hour = get_current_hour_utc()
    return current_hour in PEAK_HOURS

def detect_market_regime(price_data):
    """Detect current market regime"""
    if len(price_data) < 5:
        return 'ranging'

    try:
        prices = price_data[-60:] if len(price_data) >= 60 else price_data
        if len(prices) < 5:
            return 'ranging'

        # Calculate volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0

        # Calculate trend
        if len(prices) >= 10:
            short_trend = (prices[-1] - prices[-10]) / prices[-10]
            if short_trend > 0.02:
                trend = 'bull'
            elif short_trend < -0.02:
                trend = 'bear'
            else:
                trend = 'sideways'
        else:
            trend = 'sideways'

        # Determine regime
        if volatility > VOLATILITY_THRESHOLDS['high']:
            return 'volatile'
        elif trend == 'bull' and volatility < VOLATILITY_THRESHOLDS['medium']:
            return 'bull'
        elif trend == 'bear':
            return 'bear'
        else:
            return 'ranging'

    except Exception as e:
        logger.warning(f"Regime detection error: {e}")
        return 'ranging'

def get_adaptive_parameters(market_regime, is_peak_hour):
    """Get adaptive prediction parameters"""
    regime_params = MARKET_REGIMES[market_regime]

    if is_peak_hour:
        params = regime_params.copy()
        params['freq'] = min(params['freq'], PEAK_FREQUENCY)
        params['threshold'] = min(params['threshold'], PEAK_CONFIDENCE_THRESHOLD)
        params['description'] += " (PEAK HOUR BONUS)"
    else:
        params = regime_params.copy()
        params['freq'] = min(params['freq'], NORMAL_FREQUENCY)
        params['threshold'] = min(params['threshold'], NORMAL_CONFIDENCE_THRESHOLD)

    # Additional domination multiplier for peak hours
    if is_peak_hour and market_regime in ['volatile', 'bull']:
        params['freq'] = max(10, params['freq'] // 2)  # At least every 10 minutes
        params['threshold'] = max(0.65, params['threshold'] - 0.1)  # Lower threshold

    return params

def should_make_prediction(confidence_score, market_regime, is_peak_hour):
    """Determine if prediction should be made"""
    params = get_adaptive_parameters(market_regime, is_peak_hour)

    # Peak hour bonus
    if is_peak_hour:
        confidence_score *= 1.2

    return confidence_score >= params['threshold']

def track_prediction(prediction_value, actual_value, confidence, response_time, reward=0.0):
    """Track prediction performance"""
    global prediction_count, total_reward, response_times

    prediction_count += 1
    total_reward += reward
    response_times.append(response_time)

    # Keep only last 100 response times
    if len(response_times) > 100:
        response_times = response_times[-100:]

    # Log performance every 10 predictions
    if prediction_count % 10 == 0:
        avg_reward = total_reward / prediction_count
        avg_response_time = statistics.mean(response_times) if response_times else 0

        logger.info(f"📊 Performance Update: {prediction_count} predictions | "
                   f"Avg Reward: {avg_reward:.6f} TAO | "
                   f"Avg Response: {avg_response_time:.3f}s")

        # Check domination targets
        if avg_reward >= 0.08:
            logger.info("🎉 TARGET ACHIEVED: Surpassing UID 31 level!")
        elif avg_reward >= 0.05:
            logger.info("✅ Good progress - on track for domination")
        else:
            logger.warning("⚠️ Reward performance needs improvement")

def load_domination_models():
    """Load domination models"""
    global point_model, interval_model, scaler, models_loaded

    if 'models_loaded' in globals() and models_loaded:
        return

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading domination models on device: {device}")

        # Load domination ensemble
        from direct_domination_upgrade import BasicEnsemble
        point_model = BasicEnsemble(input_size=24, hidden_size=128)
        point_model.load_state_dict(torch.load('models/domination_ensemble.pth', map_location=device))
        point_model.to(device)
        point_model.eval()

        # Try to load scaler
        try:
            import joblib
            scaler = joblib.load('models/feature_scaler.pkl')
        except:
            logger.warning("Feature scaler not found, using identity scaling")
            scaler = None

        models_loaded = True
        logger.info("✅ Domination models loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise

async def forward(synapse, cm):
    """Domination forward function with peak hour optimization"""

    start_time = time.time()

    try:
        # Get market data
        data = cm.get_recent_data(minutes=60)
        if data.empty:
            logger.warning("No market data available")
            synapse.predictions = None
            synapse.intervals = None
            return synapse

        # Detect market conditions
        market_regime = detect_market_regime(data['price'].values)
        is_peak = is_peak_hour()

        params = get_adaptive_parameters(market_regime, is_peak)

        logger.info(f"🎯 Market Regime: {market_regime.upper()} | Peak Hour: {is_peak} | "
                   f"Strategy: {params['description']}")

        # Extract features (simplified version)
        current_price = data['price'].iloc[-1]

        # Create basic feature vector (24 features as expected by model)
        features = np.zeros(24)

        # Add basic price features
        if len(data) >= 2:
            features[0] = (current_price - data['price'].iloc[-2]) / data['price'].iloc[-2]  # 1m return

        if len(data) >= 6:
            features[1] = (current_price - data['price'].iloc[-6]) / data['price'].iloc[-6]  # 5m return

        if len(data) >= 16:
            features[2] = (current_price - data['price'].iloc[-16]) / data['price'].iloc[-16]  # 15m return

        # Add some basic technical indicators (simplified)
        prices = data['price'].values[-60:] if len(data) >= 60 else data['price'].values
        if len(prices) >= 20:
            # Simple moving averages
            features[3] = np.mean(prices[-5:]) / current_price - 1  # 5-period MA
            features[4] = np.mean(prices[-10:]) / current_price - 1  # 10-period MA
            features[5] = np.mean(prices[-20:]) / current_price - 1  # 20-period MA

        # Calculate confidence based on market stability
        market_volatility = np.std(np.diff(prices[-20:]) / prices[-21:-1]) if len(prices) >= 21 else 0.01
        confidence_score = min(1.0, 1.0 / (1.0 + market_volatility * 10))

        # Decide whether to predict
        should_predict = should_make_prediction(confidence_score, market_regime, is_peak)

        if should_predict:
            # Make prediction using ensemble
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)

            # Apply scaling if available
            if scaler:
                features_scaled = scaler.transform(features.reshape(1, -1))
                features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)

            with torch.no_grad():
                if 'point_model' in globals() and point_model:
                    point_prediction = point_model(features_tensor).item()
                else:
                    # Fallback prediction
                    point_prediction = current_price * (1 + np.random.normal(0, 0.005))

            # Convert to TAO prediction
            tao_prediction = point_prediction * 1000  # Rough conversion

            # Create intervals
            interval_width = abs(point_prediction * 0.05)
            lower_bound = point_prediction - interval_width
            upper_bound = point_prediction + interval_width

            synapse.predictions = [tao_prediction]
            synapse.intervals = [[lower_bound * 1000, upper_bound * 1000]]

            # Track performance
            response_time = time.time() - start_time
            track_prediction(
                prediction_value=point_prediction,
                actual_value=current_price,
                confidence=confidence_score,
                response_time=response_time,
                reward=0.0  # Will be updated when reward is received
            )

            logger.info(f"🎯 Prediction made: {tao_prediction:.2f} TAO | "
                       f"Confidence: {confidence_score:.2f} | "
                       f"Regime: {market_regime} | "
                       f"Peak: {is_peak}")

        else:
            synapse.predictions = None
            synapse.intervals = None
            logger.info(f"⏸️ Skipping prediction (confidence: {confidence_score:.2f} < threshold)")

    except Exception as e:
        logger.error(f"❌ Domination forward error: {e}")
        synapse.predictions = None
        synapse.intervals = None

    return synapse

# Initialize models on import
try:
    load_domination_models()
except Exception as e:
    logger.warning(f"Could not load domination models: {e}")
    logger.info("Running with basic prediction capabilities")
