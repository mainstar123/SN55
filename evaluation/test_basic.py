#!/usr/bin/env python3
"""
Basic test script to validate the shadow evaluation system structure.
This doesn't run actual API calls but tests the code structure.
"""

import os
import sys
from datetime import datetime, timezone

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from collector import DataCollector
from ground_truth import GroundTruthResolver
from evaluator import Evaluator, EvaluationMetrics


def test_imports():
    """Test that all modules can be imported."""
    print("✓ All modules imported successfully")


def example_model(spot_price, timestamp):
    """Simple test model."""
    return {
        'point': spot_price,
        'low': spot_price * 0.98,
        'high': spot_price * 1.02
    }


def test_structure():
    """Test basic class instantiation (without API calls)."""
    print("Testing basic structure...")

    # Test collector initialization (will fail without API key, but that's expected)
    try:
        collector = DataCollector("test_data.csv")
        print("✗ Collector initialized without API key (unexpected)")
    except ValueError as e:
        print("✓ Collector properly requires API key")

    # Test evaluator
    evaluator = Evaluator("test_data.csv")
    print("✓ Evaluator initialized")

    # Test ground truth resolver
    try:
        resolver = GroundTruthResolver("test_data.csv")
        print("✗ Ground truth resolver initialized without API key (unexpected)")
    except ValueError as e:
        print("✓ Ground truth resolver properly requires API key")

    print("✓ Basic structure validation complete")


def test_metrics_calculation():
    """Test metrics calculation logic."""
    print("Testing metrics calculation...")

    evaluator = Evaluator()

    # Test point error
    error = evaluator.calculate_point_error(100, 105)
    assert error == 5.0, f"Expected 5.0, got {error}"

    # Test interval hit
    hit = evaluator.calculate_interval_hit(95, 105, 98, 102)
    assert hit == 1, f"Expected 1, got {hit}"

    hit = evaluator.calculate_interval_hit(95, 105, 90, 110)
    assert hit == 0, f"Expected 0, got {hit}"

    # Test interval width
    width = evaluator.calculate_interval_width(95, 105)
    assert width == 10, f"Expected 10, got {width}"

    # Test miss penalty
    penalty = evaluator.calculate_interval_miss_penalty(90, 110, 95, 105)
    assert penalty == 0, f"Expected 0, got {penalty}"  # predicted covers actual perfectly

    penalty = evaluator.calculate_interval_miss_penalty(95, 105, 85, 115)
    assert penalty == 20, f"Expected 20, got {penalty}"  # 10 below + 10 above

    # Test combined score
    score = evaluator.calculate_combined_score(5.0, 1, 0, 10)
    expected = 5.0 + 0 + 0.1 * 10  # point_error + alpha*miss_penalty + beta*width
    assert abs(score - expected) < 0.001, f"Expected {expected}, got {score}"

    print("✓ Metrics calculation tests passed")


def main():
    """Run all tests."""
    print("Running basic validation tests...\n")

    test_imports()
    test_structure()
    test_metrics_calculation()

    print("\n✓ All basic tests passed!")
    print("\nNext steps:")
    print("1. Set COINMETRICS_API_KEY environment variable")
    print("2. Implement your actual forecasting model in main.py")
    print("3. Test with real API calls: python main.py --mode once")


if __name__ == "__main__":
    main()
