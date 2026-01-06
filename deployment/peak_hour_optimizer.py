#!/usr/bin/env python3
"""
PEAK HOUR OPTIMIZER
Automatically adjusts miner behavior during peak hours
Usage: Import in miner code or run standalone for testing
"""

import datetime
import pytz
import json
import os
from typing import Dict, Any

class PeakHourOptimizer:
    def __init__(self, timezone_str='UTC'):
        self.timezone = pytz.timezone(timezone_str)
        # Peak hours in UTC (when most trading activity occurs)
        self.peak_hours = [
            (9, 11),   # UTC 9-11 (London morning)
            (13, 15)   # UTC 13-15 (US morning)
        ]

        # Configuration for different periods
        self.settings = {
            'peak': {
                'prediction_frequency': 5,    # minutes
                'confidence_threshold': 0.75, # Lower threshold = more predictions
                'batch_size': 8,
                'timeout': 12,
                'description': 'High frequency, lower confidence threshold'
            },
            'normal': {
                'prediction_frequency': 15,   # minutes
                'confidence_threshold': 0.85, # Higher threshold = fewer predictions
                'batch_size': 4,
                'timeout': 16,
                'description': 'Standard frequency and confidence'
            }
        }

    def is_peak_hour(self, dt=None) -> bool:
        """Check if current time is peak hour"""
        if dt is None:
            dt = datetime.datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = self.timezone.localize(dt)

        current_hour = dt.hour
        current_minute = dt.minute

        for start_hour, end_hour in self.peak_hours:
            if start_hour <= current_hour < end_hour:
                return True
            # Check if we're in the last 30 minutes of peak hour
            elif current_hour == end_hour - 1 and current_minute >= 30:
                return True

        return False

    def get_current_settings(self) -> Dict[str, Any]:
        """Get current optimal settings based on time"""
        is_peak = self.is_peak_hour()

        settings = self.settings['peak'] if is_peak else self.settings['normal']
        settings_copy = settings.copy()
        settings_copy['is_peak_hour'] = is_peak
        settings_copy['current_time'] = datetime.datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S UTC')

        return settings_copy

    def get_next_peak_hour(self) -> datetime.datetime:
        """Get datetime of next peak hour start"""
        now = datetime.datetime.now(self.timezone)

        for start_hour, _ in self.peak_hours:
            peak_start = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            if peak_start > now:
                return peak_start

        # If no peak hours left today, get tomorrow's first peak hour
        tomorrow = now + datetime.timedelta(days=1)
        tomorrow = tomorrow.replace(hour=self.peak_hours[0][0], minute=0, second=0, microsecond=0)
        return tomorrow

    def get_time_until_peak(self) -> int:
        """Get minutes until next peak hour"""
        next_peak = self.get_next_peak_hour()
        now = datetime.datetime.now(self.timezone)
        delta = next_peak - now
        return int(delta.total_seconds() / 60)

    def should_make_prediction(self, confidence: float, current_settings: Dict = None) -> bool:
        """Determine if prediction should be made based on confidence and time"""
        if current_settings is None:
            current_settings = self.get_current_settings()

        threshold = current_settings['confidence_threshold']

        # During peak hours, be more aggressive with predictions
        if current_settings['is_peak_hour']:
            # Lower threshold and add some randomization for diversity
            import random
            adjusted_threshold = threshold * (0.9 + random.random() * 0.2)  # 0.9x to 1.1x threshold
            return confidence >= adjusted_threshold
        else:
            return confidence >= threshold

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        current = self.get_current_settings()
        next_peak = self.get_next_peak_hour()
        minutes_until = self.get_time_until_peak()

        return {
            'current_time': current['current_time'],
            'is_peak_hour': current['is_peak_hour'],
            'current_mode': 'PEAK' if current['is_peak_hour'] else 'NORMAL',
            'prediction_frequency': current['prediction_frequency'],
            'confidence_threshold': current['confidence_threshold'],
            'next_peak_hour': next_peak.strftime('%Y-%m-%d %H:%M UTC'),
            'minutes_until_peak': minutes_until,
            'description': current['description']
        }

# Global optimizer instance
_optimizer = None

def get_optimizer():
    """Get or create optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = PeakHourOptimizer()
    return _optimizer

# Utility functions for miner integration
def should_predict_now(confidence: float) -> bool:
    """Convenience function for miner to check if prediction should be made"""
    optimizer = get_optimizer()
    return optimizer.should_make_prediction(confidence)

def get_miner_settings() -> Dict[str, Any]:
    """Get current optimal miner settings"""
    optimizer = get_optimizer()
    return optimizer.get_current_settings()

def log_peak_hour_status():
    """Log current peak hour status (for debugging)"""
    import logging
    optimizer = get_optimizer()
    status = optimizer.get_status_report()

    logging.info(f"Peak Hour Status: {status['current_mode']} | "
                f"Threshold: {status['confidence_threshold']} | "
                f"Frequency: {status['prediction_frequency']}min | "
                f"Next peak: {status['minutes_until_peak']}min")

if __name__ == "__main__":
    # Standalone testing
    import time

    optimizer = PeakHourOptimizer()
    print("Peak Hour Optimizer Test")
    print("=" * 40)

    for i in range(5):
        status = optimizer.get_status_report()
        print(f"Time: {status['current_time']}")
        print(f"Mode: {status['current_mode']}")
        print(f"Peak Hour: {status['is_peak_hour']}")
        print(f"Confidence Threshold: {status['confidence_threshold']}")
        print(f"Prediction Frequency: {status['prediction_frequency']} minutes")
        print(f"Next Peak: {status['next_peak_hour']} ({status['minutes_until_peak']} minutes)")
        print("-" * 40)

        # Test prediction logic
        test_confidences = [0.7, 0.8, 0.9]
        print("Prediction decisions for test confidences:")
        for conf in test_confidences:
            decision = optimizer.should_make_prediction(conf)
            print(f"  Confidence {conf}: {'PREDICT' if decision else 'SKIP'}")

        print()

        if i < 4:  # Don't sleep on last iteration
            time.sleep(2)
