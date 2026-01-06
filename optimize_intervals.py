#!/usr/bin/env python3
"""Optimize prediction intervals for better coverage and first-place performance"""

import os
os.environ['TRAINING_MODE'] = 'true'

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_top_miners_intervals():
    """Analyze interval patterns from top miners CSV data"""
    print("📊 ANALYZING TOP MINERS INTERVAL PATTERNS")
    print("=" * 60)

    assets = ['bitcoin', 'ethereum', 'tao']
    interval_analysis = {}

    for asset in assets:
        file_path = f'evaluation/csv_log/{asset}_full.csv'

        try:
            df = pd.read_csv(file_path)

            # Get top 10 miners data
            top_miners_data = df[df['Rank'] <= 10]

            # Calculate interval widths
            interval_widths = top_miners_data['Interval Upper Bound'] - top_miners_data['Interval Lower Bound']

            # Calculate statistics
            avg_width = interval_widths.mean()
            median_width = interval_widths.median()
            width_std = interval_widths.std()

            # Current prices for percentage calculation
            current_prices = {'bitcoin': 95000, 'ethereum': 3100, 'tao': 290}

            avg_width_pct = (avg_width / current_prices[asset]) * 100

            interval_analysis[asset.upper()] = {
                'avg_interval_width': avg_width,
                'median_interval_width': median_width,
                'interval_width_std': width_std,
                'avg_interval_width_pct': avg_width_pct,
                'current_price': current_prices[asset]
            }

            print(f"\n{asset.upper()} INTERVAL ANALYSIS:")
            print(".3f")
            print(".3f")
            print(".3f")

        except Exception as e:
            print(f"❌ Error analyzing {asset}: {e}")

    return interval_analysis

def optimize_intervals(interval_analysis):
    """Calculate optimal interval widths for 85-95% coverage based on top miners"""
    print("\n🎯 OPTIMIZING PREDICTION INTERVALS")
    print("=" * 60)

    optimized_intervals = {}

    for asset, stats in interval_analysis.items():
        print(f"\n{asset} INTERVAL OPTIMIZATION:")

        current_avg_width = stats['avg_interval_width']
        current_price = stats['current_price']

        # Top miners have ~77% coverage with current intervals
        # We need to increase interval width to achieve 85-95% coverage

        coverage_targets = [85, 90, 95]  # Target coverage percentages

        for target_coverage in coverage_targets:
            # Calculate required interval expansion
            # Assuming current coverage is ~77%, we need to expand intervals
            current_coverage = 77  # From our analysis
            coverage_ratio = target_coverage / current_coverage

            # Expand interval width to achieve target coverage
            # Using square root relationship (rough approximation)
            expansion_factor = np.sqrt(coverage_ratio)

            optimized_width = current_avg_width * expansion_factor

            # Add safety buffer for your model's uncertainty
            safety_buffer = 1.1  # 10% additional buffer for model uncertainty

            final_width = optimized_width * safety_buffer
            final_width_pct = (final_width / current_price) * 100

            optimized_intervals[f"{asset.lower()}_{target_coverage}pct"] = {
                'coverage_target': target_coverage,
                'interval_width_price': final_width,
                'interval_width_pct': final_width_pct,
                'expansion_factor': expansion_factor * safety_buffer
            }

            print(".1f")
            print(".3f")
            print(".3f")
            print(".2f")

    return optimized_intervals

def validate_intervals_with_top_miners(optimized_intervals):
    """Validate optimized intervals using top miners performance data"""
    print("\n🧪 VALIDATING OPTIMIZED INTERVALS")
    print("=" * 60)

    validation_results = {}

    for asset in ['bitcoin', 'ethereum', 'tao']:
        file_path = f'evaluation/csv_log/{asset}_full.csv'

        try:
            df = pd.read_csv(file_path)

            # Get top 10 miners data for validation
            top_miners_data = df[df['Rank'] <= 10]

            # Check coverage if actual prices are available
            valid_data = top_miners_data.dropna(subset=['CM Reference Rate at Eval Time'])

            if len(valid_data) > 0:
                actual_prices = valid_data['CM Reference Rate at Eval Time']
                lower_bounds = valid_data['Interval Lower Bound']
                upper_bounds = valid_data['Interval Upper Bound']

                # Calculate current coverage
                current_coverage = np.mean((actual_prices >= lower_bounds) & (actual_prices <= upper_bounds)) * 100

                print(f"\n{asset.upper()} VALIDATION:")
                print(".1f")

                # Validate our optimized intervals
                for interval_key, config in optimized_intervals.items():
                    if interval_key.startswith(asset.lower()):
                        target_coverage = config['coverage_target']
                        optimized_width = config['interval_width_price']

                        # Simulate coverage with optimized intervals
                        # Using point forecasts as center points
                        point_forecasts = valid_data['Point Forecast']

                        # Create optimized intervals around point forecasts
                        opt_lower = point_forecasts - optimized_width/2
                        opt_upper = point_forecasts + optimized_width/2

                        # Calculate expected coverage
                        expected_coverage = np.mean((actual_prices >= opt_lower) & (actual_prices <= opt_upper)) * 100

                        print(".1f")
                        print(".1f")

                        validation_results[interval_key] = {
                            'current_coverage': current_coverage,
                            'target_coverage': target_coverage,
                            'expected_coverage': expected_coverage,
                            'interval_width': optimized_width
                        }

        except Exception as e:
            print(f"❌ Error validating {asset}: {e}")

    return validation_results

def generate_interval_recommendations(optimized_intervals, validation_results):
    """Generate final recommendations for interval settings"""
    print("\n🎯 FINAL INTERVAL RECOMMENDATIONS FOR FIRST PLACE")
    print("=" * 60)

    recommendations = {}

    for asset in ['BTC', 'ETH', 'TAO']:
        print(f"\n{asset} RECOMMENDED SETTINGS:")

        # Find best performing interval for this asset
        best_config = None
        best_score = float('inf')

        for key, config in validation_results.items():
            if key.startswith(asset.lower()):
                # Score based on coverage achievement and interval efficiency
                coverage_achievement = config['expected_coverage'] / config['target_coverage']
                efficiency_score = 1 / config['interval_width']  # Prefer narrower intervals
                score = (1 - coverage_achievement) + (1 - efficiency_score * 1000)  # Lower is better

                if score < best_score:
                    best_score = score
                    best_config = config

        if best_config:
            recommendations[asset] = {
                'interval_width_price': best_config['interval_width'],
                'expected_coverage': best_config['expected_coverage'],
                'target_coverage': best_config['target_coverage']
            }

            print(".1f")
            print(".1f")
            print(".3f")

            # Provide implementation code
            print("  📝 Implementation in miner:")
            print(f"     interval_width = {best_config['interval_width']:.2f}")
            print("     lower_bound = prediction - interval_width/2")
            print("     upper_bound = prediction + interval_width/2")

        else:
            # Fallback to optimized intervals
            for key, config in optimized_intervals.items():
                if key.startswith(asset.lower()):
                    recommendations[asset] = config
                    print(".1f")
                    break

    print("\n" + "=" * 60)
    print("🎉 INTERVAL OPTIMIZATION COMPLETE!")
    print("💡 Integrate these intervals into enhanced_domination_miner.py")
    print("🚀 Then deploy with: ./deployment/deploy_first_place_miner.sh")
    print("=" * 60)

    return recommendations

def main():
    """Main optimization function"""
    print("🎯 PRECOG SUBNET 55: INTERVAL OPTIMIZATION FOR FIRST PLACE")
    print("=" * 70)

    # Step 1: Analyze top miners intervals
    interval_analysis = analyze_top_miners_intervals()

    if not interval_analysis:
        print("❌ Could not analyze intervals. Exiting.")
        return

    # Step 2: Calculate optimal intervals
    optimized_intervals = optimize_intervals(interval_analysis)

    # Step 3: Validate with top miners data
    validation_results = validate_intervals_with_top_miners(optimized_intervals)

    # Step 4: Generate recommendations
    recommendations = generate_interval_recommendations(optimized_intervals, validation_results)

    print("\n🚀 READY TO DOMINATE PRECOG SUBNET 55!")
    print("💡 Next: Integrate optimized intervals into your miner")
    print("🚀 Then: ./deployment/deploy_first_place_miner.sh")

if __name__ == "__main__":
    main()
