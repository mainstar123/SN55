#!/usr/bin/env python3
"""
Precog Competitor Analysis Tool
Analyze real competitor data to improve your mining performance
"""

import sys
import os
sys.path.append('.')

from mining_dashboard import get_metagraph_data
import pandas as pd
import numpy as np

def analyze_competitors():
    """Analyze competitor data for model improvement insights"""

    print("🔍 PRECOG COMPETITOR ANALYSIS")
    print("=" * 50)

    # Get real competitor data - use btcli directly for most reliable data
    from mining_dashboard import get_metagraph_via_btcli
    df = get_metagraph_via_btcli()

    if df.empty or len(df) == 0:
        print("❌ No competitor data available")
        return

    print(f"📊 Analyzing {len(df)} miners on subnet 55")
    print()

    # Top performers analysis
    print("🏆 TOP PERFORMERS ANALYSIS:")
    print("-" * 30)

    top5 = df.head(5)
    for _, miner in top5.iterrows():
        print(".4f"
              f"Trust: {miner['trust']:.1f}, Stake: {miner['stake']:.1f}")

    print()

    # Performance distribution
    print("📈 PERFORMANCE DISTRIBUTION:")
    print("-" * 30)

    emissions_ranges = [
        (0, 0.001, "Very Low (< 0.001 τ)"),
        (0.001, 0.01, "Low (0.001-0.01 τ)"),
        (0.01, 0.1, "Medium (0.01-0.1 τ)"),
        (0.1, 1.0, "High (0.1-1.0 τ)"),
        (1.0, float('inf'), "Elite (1.0+ τ)")
    ]

    for min_val, max_val, label in emissions_ranges:
        count = len(df[(df['emissions'] >= min_val) & (df['emissions'] < max_val)])
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count} miners ({percentage:5.1f}%)")
    print()

    # Trust vs Performance correlation
    print("🎯 TRUST vs PERFORMANCE CORRELATION:")
    print("-" * 40)

    # Calculate correlation
    correlation = df['trust'].corr(df['emissions'])
    print(f"Trust vs Emissions correlation: {correlation:.3f}")
    # High trust miners
    high_trust = df[df['trust'] > 50]
    if len(high_trust) > 0:
        print(f"High trust miners (>50): {len(high_trust)}")
        avg_emissions = high_trust['emissions'].mean()
        print(f"   Average emissions: {avg_emissions:.4f} τ")
    # Low trust miners
    low_trust = df[df['trust'] < 5]
    if len(low_trust) > 0:
        print(f"Low trust miners (<5): {len(low_trust)}")
        avg_emissions = low_trust['emissions'].mean()
        print(f"   Average emissions: {avg_emissions:.4f} τ")
    print()

    # Strategy insights
    print("💡 MODEL IMPROVEMENT INSIGHTS:")
    print("-" * 35)

    # Top strategies
    top_performer = df.iloc[0]
    print("🎯 TOP PERFORMER STRATEGY:")
    print(f"   Emissions: {top_performer['emissions']:.1f} τ")
    print(f"   Trust: {top_performer['trust']:.1f} (Reliability)")
    print(f"   Stake: {top_performer['stake']:.1f} τ (Network weight)")
    print()

    # Consistency analysis
    consistent_miners = df[df['emissions'] > df['emissions'].median()]
    avg_trust_consistent = consistent_miners['trust'].mean()
    avg_stake_consistent = consistent_miners['stake'].mean()

    print("📊 SUCCESSFUL MINER PROFILE:")
    print(f"   Average Trust: {avg_trust_consistent:.1f}")
    print(f"   Average Stake: {avg_stake_consistent:.1f}")
    print()

    # Your current position analysis
    your_position = len(df)  # Assuming you're at the bottom initially
    percentile = ((len(df) - your_position) / len(df)) * 100

    print("🎯 YOUR CURRENT POSITION:")
    print("-" * 25)
    print(f"Rank: {your_position} out of {len(df)} miners")
    print(f"Percentile: {percentile:.1f}%")
    print()

    # Improvement targets
    print("🚀 IMPROVEMENT TARGETS:")
    print("-" * 25)

    # Target the miner just above median
    median_emissions = df['emissions'].median()
    median_position = len(df) // 2
    target_miner = df.iloc[min(median_position, len(df)-1)]

    print("🎯 SHORT-TERM TARGET (Median Performance):")
    print(f"   Target Emissions: {median_emissions:.4f} τ")
    print(f"   Required Trust: {target_miner['trust']:.1f}")
    print()

    # Target top 25%
    top25_percentile = int(len(df) * 0.75)
    top25_miner = df.iloc[min(top25_percentile, len(df)-1)]

    print("🎯 MEDIUM-TERM TARGET (Top 25%):")
    print(f"   Target Emissions: {top25_miner['emissions']:.4f} τ")
    print(f"   Required Trust: {top25_miner['trust']:.1f}")
    print()

    print("🔧 MODEL IMPROVEMENT ACTIONS:")
    print("-" * 35)
    print("1. 📊 Focus on prediction accuracy (main factor for emissions)")
    print("2. ⚡ Optimize response time (affects trust scores)")
    print("3. 🎯 Target 0.01+ τ daily emissions for top 50% ranking")
    print("4. 🔄 Monitor competitor strategies weekly")
    print("5. 📈 Retrain model when accuracy drops below 0.15 MAPE")
    print()

    # Save analysis to file
    analysis_file = "competitor_analysis.txt"
    with open(analysis_file, 'w') as f:
        f.write("PRECROG COMPETITOR ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total miners analyzed: {len(df)}\n")
        f.write(".3f")
        f.write(".4f")
        f.write(".4f")
        f.write(f"\nYour current rank: {your_position}/{len(df)}\n")
        f.write(".1f")

    print(f"💾 Analysis saved to: {analysis_file}")
    print()
    print("🎉 Analysis complete! Use this data to improve your model's performance!")

if __name__ == "__main__":
    os.environ['HOME'] = '/home/ocean'
    analyze_competitors()
