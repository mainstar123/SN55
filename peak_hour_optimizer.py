"""
Ultra-Precise Peak Hour Detection and Optimization for Precog #1 Miner
15-minute granularity analysis for maximum reward capture
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timezone, timedelta
import logging
import json
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class UltraPrecisePeakHourOptimizer:
    """
    15-minute granularity peak hour detection for maximum TAO earnings
    Analyzes reward patterns at minute-level precision
    """

    def __init__(self, timezone_offset: int = 0, analysis_window_days: int = 7):
        self.timezone_offset = timezone_offset
        self.analysis_window_days = analysis_window_days

        # Ultra-precise time tracking (15-minute intervals)
        self.time_intervals = {}  # (hour, 15min_slot) -> performance data
        self.reward_history = []
        self.peak_intervals = set()

        # Known Bittensor peak periods (UTC)
        self.known_peak_periods = {
            (9, 0): 1.0, (9, 15): 1.2, (9, 30): 1.4, (9, 45): 1.2,  # 9:00-10:00
            (10, 0): 1.4, (10, 15): 1.6, (10, 30): 1.4, (10, 45): 1.2,  # 10:00-11:00
            (11, 0): 1.0,  # End of morning peak
            (13, 0): 1.0, (13, 15): 1.2, (13, 30): 1.4, (13, 45): 1.2,  # 13:00-14:00
            (14, 0): 1.4, (14, 15): 1.6, (14, 30): 1.4, (14, 45): 1.2,  # 14:00-15:00
            (15, 0): 1.0,  # End of afternoon peak
        }

        # Initialize all 15-minute intervals
        self._initialize_intervals()

    def _initialize_intervals(self):
        """Initialize all 15-minute intervals for 24 hours"""
        for hour in range(24):
            for quarter in [0, 15, 30, 45]:
                key = (hour, quarter)
                self.time_intervals[key] = {
                    'rewards': [],
                    'predictions': [],
                    'success_rate': [],
                    'avg_reward': 0.0,
                    'total_volume': 0,
                    'peak_score': self.known_peak_periods.get(key, 0.1)
                }

    def update_reward_data(self, reward: float, timestamp: Optional[datetime] = None,
                          prediction_success: bool = None):
        """Update reward data with 15-minute precision"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Adjust for timezone
        local_time = timestamp + timedelta(hours=self.timezone_offset)
        hour = local_time.hour
        minute = local_time.minute

        # Get 15-minute interval (0, 15, 30, 45)
        quarter = (minute // 15) * 15
        interval_key = (hour, quarter)

        # Store data
        self.reward_history.append({
            'reward': reward,
            'timestamp': timestamp,
            'local_hour': hour,
            'local_minute': minute,
            'interval': interval_key,
            'success': prediction_success
        })

        # Update interval statistics
        if interval_key in self.time_intervals:
            interval_data = self.time_intervals[interval_key]
            interval_data['rewards'].append(reward)
            interval_data['total_volume'] += 1

            if prediction_success is not None:
                interval_data['predictions'].append(prediction_success)
                interval_data['success_rate'].append(prediction_success)

        # Keep only recent data
        cutoff_time = timestamp - timedelta(days=self.analysis_window_days)
        self.reward_history = [r for r in self.reward_history if r['timestamp'] > cutoff_time]

    def analyze_peak_intervals(self) -> Dict[str, Union[List[Tuple[int, int]], Dict, float]]:
        """Analyze 15-minute intervals to identify optimal prediction times"""
        # Update statistics for all intervals
        for interval_key, data in self.time_intervals.items():
            rewards = data['rewards']
            if rewards:
                data['avg_reward'] = np.mean(rewards)

                # Calculate success rate if available
                if data['predictions']:
                    data['success_rate'] = np.mean(data['predictions'])
                else:
                    # Estimate success rate from reward patterns
                    data['success_rate'] = min(1.0, max(0.1, data['avg_reward'] / 0.001))  # Rough estimation

        # Identify top performing intervals
        interval_performance = []
        for interval_key, data in self.time_intervals.items():
            if data['total_volume'] >= 5:  # Minimum data requirement
                # Combined score: reward + volume + success rate
                reward_score = data['avg_reward'] / max([d['avg_reward'] for d in self.time_intervals.values() if d['rewards']], default=1e-6)
                volume_score = min(1.0, data['total_volume'] / 50)  # Normalize volume
                success_score = data['success_rate']

                combined_score = (reward_score * 0.5 + volume_score * 0.3 + success_score * 0.2)
                peak_multiplier = combined_score * 2.0  # Convert to multiplier

                interval_performance.append({
                    'interval': interval_key,
                    'score': combined_score,
                    'peak_multiplier': peak_multiplier,
                    'avg_reward': data['avg_reward'],
                    'volume': data['total_volume'],
                    'success_rate': data['success_rate']
                })

        # Sort by performance
        interval_performance.sort(key=lambda x: x['score'], reverse=True)

        # Select top 12 intervals (3 hours of peak time)
        top_intervals = interval_performance[:12]
        peak_intervals = [item['interval'] for item in top_intervals]

        # Calculate overall peak multiplier
        avg_peak_reward = np.mean([item['avg_reward'] for item in top_intervals])
        avg_off_peak_reward = np.mean([data['avg_reward']
                                      for data in self.time_intervals.values()
                                      if data['rewards'] and (data['total_volume'] >= 5)])

        if avg_off_peak_reward > 0:
            overall_peak_multiplier = avg_peak_reward / avg_off_peak_reward
        else:
            overall_peak_multiplier = 2.0  # Default

        self.peak_intervals = set(peak_intervals)

        return {
            'peak_intervals': peak_intervals,
            'top_intervals': top_intervals,
            'overall_peak_multiplier': overall_peak_multiplier,
            'interval_performance': interval_performance,
            'total_analyzed_intervals': len([d for d in self.time_intervals.values() if d['total_volume'] >= 5])
        }

    def should_predict_now(self, current_time: Optional[datetime] = None) -> Tuple[bool, float, Dict]:
        """
        Determine if current 15-minute interval is optimal for predictions
        Returns: (should_predict, confidence_multiplier, interval_info)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Get current 15-minute interval
        local_time = current_time + timedelta(hours=self.timezone_offset)
        hour = local_time.hour
        minute = local_time.minute
        quarter = (minute // 15) * 15
        current_interval = (hour, quarter)

        # Check if it's a peak interval
        is_peak = current_interval in self.peak_intervals

        # Get interval performance data
        interval_data = self.time_intervals.get(current_interval, {})
        avg_reward = interval_data.get('avg_reward', 0.0005)  # Default

        # Calculate confidence multiplier
        if is_peak:
            # High confidence during peak intervals
            base_multiplier = 1.5
            reward_bonus = min(2.0, avg_reward / 0.0005)  # Reward-based bonus
            confidence_multiplier = base_multiplier * reward_bonus
        else:
            # Lower confidence off-peak
            confidence_multiplier = 0.3

        interval_info = {
            'interval': current_interval,
            'is_peak': is_peak,
            'avg_reward': avg_reward,
            'volume': interval_data.get('total_volume', 0),
            'success_rate': interval_data.get('success_rate', 0.5),
            'time_remaining_minutes': 15 - (minute % 15)
        }

        return is_peak, confidence_multiplier, interval_info

    def get_prediction_schedule(self) -> Dict[str, Union[List, Dict, int]]:
        """Get detailed prediction schedule for all 15-minute intervals"""
        schedule = {}
        total_daily_predictions = 0

        for hour in range(24):
            for quarter in [0, 15, 30, 45]:
                interval_key = (hour, quarter)
                is_peak = interval_key in self.peak_intervals
                interval_data = self.time_intervals[interval_key]

                if is_peak:
                    predictions_per_interval = max(3, min(8, interval_data['total_volume'] // 10))  # 3-8 predictions
                else:
                    predictions_per_interval = max(0, min(2, interval_data['total_volume'] // 50))  # 0-2 predictions

                schedule[interval_key] = {
                    'predictions': predictions_per_interval,
                    'is_peak': is_peak,
                    'priority': 'high' if is_peak else 'low',
                    'avg_reward': interval_data['avg_reward'],
                    'time_range': f"{hour:02d}:{quarter:02d}-{(hour + (quarter + 15) // 60):02d}:{((quarter + 15) % 60):02d}"
                }

                total_daily_predictions += predictions_per_interval

        return {
            'schedule': schedule,
            'peak_intervals': list(self.peak_intervals),
            'total_daily_predictions': total_daily_predictions,
            'peak_intervals_count': len(self.peak_intervals),
            'avg_predictions_per_peak_interval': total_daily_predictions / max(1, len(self.peak_intervals))
        }

    def get_next_peak_window(self, current_time: Optional[datetime] = None) -> Dict[str, Union[datetime, int, float]]:
        """Get information about the next peak prediction window"""
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        local_time = current_time + timedelta(hours=self.timezone_offset)
        current_minute = local_time.hour * 60 + local_time.minute

        # Find next peak interval
        next_peak_start = None
        min_wait_minutes = float('inf')

        for hour, quarter in self.peak_intervals:
            interval_start_minute = hour * 60 + quarter
            if interval_start_minute > current_minute:
                wait_minutes = interval_start_minute - current_minute
                if wait_minutes < min_wait_minutes:
                    min_wait_minutes = wait_minutes
                    next_peak_start = local_time.replace(hour=hour, minute=quarter, second=0, microsecond=0)

        if next_peak_start is None:
            # No peak intervals later today, get first one tomorrow
            first_peak = min(self.peak_intervals) if self.peak_intervals else (9, 0)
            next_peak_start = (local_time + timedelta(days=1)).replace(
                hour=first_peak[0], minute=first_peak[1], second=0, microsecond=0
            )
            min_wait_minutes = (next_peak_start - local_time).total_seconds() / 60

        return {
            'next_peak_start': next_peak_start,
            'wait_minutes': int(min_wait_minutes),
            'is_currently_peak': self.should_predict_now(current_time)[0],
            'peak_intervals_today': len([i for i in self.peak_intervals if i[0] >= local_time.hour or
                                       (i[0] == local_time.hour and i[1] > local_time.minute)])
        }

    def optimize_prediction_frequency(self) -> Dict[str, Union[float, int, List]]:
        """Optimize prediction frequency based on reward patterns"""
        analysis = self.analyze_peak_intervals()
        peak_intervals = analysis['peak_intervals']

        # Calculate optimal frequencies
        peak_freq = {}
        off_peak_freq = {}

        for interval in peak_intervals:
            data = self.time_intervals[interval]
            # Frequency based on reward and volume
            optimal_freq = int(data['avg_reward'] * 10000)  # Scale reward to frequency
            optimal_freq = max(2, min(10, optimal_freq))  # Clamp between 2-10
            peak_freq[interval] = optimal_freq

        # Off-peak: very low frequency
        for hour in range(24):
            for quarter in [0, 15, 30, 45]:
                interval = (hour, quarter)
                if interval not in peak_intervals:
                    off_peak_freq[interval] = 0  # No predictions off-peak

        total_predictions = sum(peak_freq.values())

        return {
            'peak_frequencies': peak_freq,
            'off_peak_frequencies': off_peak_freq,
            'total_daily_predictions': total_predictions,
            'avg_peak_frequency': total_predictions / max(1, len(peak_intervals)),
            'optimization_score': analysis['overall_peak_multiplier']
        }


class AdaptivePredictionScheduler:
    """
    Advanced scheduler that adapts prediction timing based on real-time conditions
    """

    def __init__(self, peak_optimizer: UltraPrecisePeakHourOptimizer,
                 market_regime_detector=None):
        self.peak_optimizer = peak_optimizer
        self.market_regime_detector = market_regime_detector

        # Scheduling state
        self.last_prediction_time = None
        self.prediction_queue = []
        self.daily_prediction_count = 0

        # Adaptive parameters
        self.min_interval_seconds = 30  # Minimum time between predictions
        self.max_daily_predictions = 200  # Safety limit

    def should_make_prediction(self, current_time: Optional[datetime] = None,
                             market_regime: str = 'unknown') -> Tuple[bool, Dict[str, Union[bool, float, str]]]:
        """
        Determine if a prediction should be made right now
        Considers peak hours, market regime, and timing constraints
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Check daily limit
        if self.daily_prediction_count >= self.max_daily_predictions:
            return False, {'reason': 'daily_limit_reached', 'wait_seconds': 3600}

        # Check minimum interval
        if self.last_prediction_time:
            time_since_last = (current_time - self.last_prediction_time).total_seconds()
            if time_since_last < self.min_interval_seconds:
                wait_seconds = self.min_interval_seconds - time_since_last
                return False, {'reason': 'too_soon', 'wait_seconds': wait_seconds}

        # Get peak hour status
        is_peak, confidence_multiplier, interval_info = self.peak_optimizer.should_predict_now(current_time)

        # Adjust for market regime
        regime_multiplier = self._get_regime_multiplier(market_regime)
        final_confidence = confidence_multiplier * regime_multiplier

        # Decision logic
        if is_peak and final_confidence > 0.8:
            # High confidence peak time - predict
            decision = True
            reason = 'peak_high_confidence'
        elif is_peak and final_confidence > 0.5:
            # Medium confidence peak time - predict with caution
            decision = True
            reason = 'peak_medium_confidence'
        elif final_confidence > 1.2:
            # Very high reward potential off-peak
            decision = True
            reason = 'off_peak_high_reward'
        else:
            # Not optimal time
            decision = False
            reason = 'not_optimal_time'

        decision_info = {
            'decision': decision,
            'reason': reason,
            'is_peak': is_peak,
            'confidence_multiplier': final_confidence,
            'regime_multiplier': regime_multiplier,
            'interval_info': interval_info,
            'daily_predictions': self.daily_prediction_count,
            'next_peak_info': self.peak_optimizer.get_next_peak_window(current_time)
        }

        return decision, decision_info

    def _get_regime_multiplier(self, regime: str) -> float:
        """Get prediction multiplier based on market regime"""
        regime_multipliers = {
            'bull': 1.3,      # More aggressive in bull markets
            'bear': 0.7,      # More conservative in bear markets
            'volatile': 0.5,  # Very conservative in volatile markets
            'ranging': 1.0,   # Normal in ranging markets
            'unknown': 0.8    # Slightly conservative when unknown
        }
        return regime_multipliers.get(regime, 0.8)

    def record_prediction(self, timestamp: Optional[datetime] = None):
        """Record that a prediction was made"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.last_prediction_time = timestamp
        self.daily_prediction_count += 1

        # Reset daily count at midnight UTC
        if timestamp.hour == 0 and timestamp.minute < 1:
            self.daily_prediction_count = 0

    def get_scheduler_status(self) -> Dict[str, Union[int, float, datetime, Dict]]:
        """Get current scheduler status"""
        current_time = datetime.now(timezone.utc)
        next_peak = self.peak_optimizer.get_next_peak_window(current_time)

        return {
            'daily_predictions': self.daily_prediction_count,
            'last_prediction': self.last_prediction_time,
            'time_since_last_minutes': (current_time - (self.last_prediction_time or current_time)).total_seconds() / 60,
            'next_peak_window': next_peak,
            'scheduler_active': True,
            'peak_intervals_active': len(self.peak_optimizer.peak_intervals)
        }


def create_ultra_precise_prediction_system(timezone_offset: int = 0):
    """Create complete ultra-precise prediction timing system"""
    peak_optimizer = UltraPrecisePeakHourOptimizer(timezone_offset)
    scheduler = AdaptivePredictionScheduler(peak_optimizer)

    system = {
        'peak_optimizer': peak_optimizer,
        'scheduler': scheduler,
        'timezone_offset': timezone_offset,
        'precision_level': '15_minute'
    }

    return system


if __name__ == "__main__":
    # Test ultra-precise peak hour optimization
    print("🎯 Testing Ultra-Precise Peak Hour Optimization (15-min granularity)")
    print("=" * 70)

    # Create optimizer
    optimizer = UltraPrecisePeakHourOptimizer(timezone_offset=0)

    # Simulate reward data for different 15-minute intervals
    print("\n📊 Simulating reward data across 15-minute intervals...")

    base_time = datetime.now(timezone.utc)
    for day in range(3):  # 3 days of data
        for hour in range(24):
            for quarter in [0, 15, 30, 45]:
                # Generate realistic rewards based on known peak hours
                interval_key = (hour, quarter)
                base_reward = 0.0003  # Base TAO per prediction

                if interval_key in optimizer.known_peak_periods:
                    # Peak hour multiplier
                    multiplier = optimizer.known_peak_periods[interval_key]
                    reward = base_reward * multiplier * (0.8 + np.random.random() * 0.4)  # Add noise
                else:
                    # Off-peak
                    reward = base_reward * (0.2 + np.random.random() * 0.3)

                # Add some successful predictions
                success = np.random.random() > 0.3 if reward > base_reward * 1.2 else np.random.random() > 0.6

                timestamp = base_time + timedelta(days=day, hours=hour, minutes=quarter)
                optimizer.update_reward_data(reward, timestamp, success)

    # Analyze peak intervals
    print("\n🔍 Analyzing peak intervals...")
    analysis = optimizer.analyze_peak_intervals()

    print(f"Top peak intervals: {analysis['peak_intervals'][:6]}")
    print(".2f")
    print(f"Total intervals analyzed: {analysis['total_analyzed_intervals']}")

    # Test scheduler
    print("\n⏰ Testing Adaptive Prediction Scheduler...")

    scheduler = AdaptivePredictionScheduler(optimizer)

    # Test different times
    test_times = [
        datetime.now(timezone.utc).replace(hour=10, minute=15),  # Peak time
        datetime.now(timezone.utc).replace(hour=3, minute=30),   # Off-peak time
        datetime.now(timezone.utc).replace(hour=14, minute=45),  # Peak time
    ]

    for test_time in test_times:
        should_predict, info = scheduler.should_make_prediction(test_time)
        print(f"{test_time.strftime('%H:%M')} - Predict: {should_predict} ({info['reason']})")

    # Get prediction schedule
    schedule = optimizer.get_prediction_schedule()
    print(f"\n📅 Daily prediction schedule: {schedule['total_daily_predictions']} predictions/day")
    print(f"Peak intervals: {schedule['peak_intervals_count']}")

    print("\n✅ Ultra-Precise Peak Hour Optimization Ready!")
