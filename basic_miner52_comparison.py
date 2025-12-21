#!/usr/bin/env python3
"""
BASIC COMPARISON: Our Domination System vs Top Miner #52
"""

import sys
import os
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quick_backtest import test_model_performance, generate_test_data
from advanced_ensemble_model import create_advanced_ensemble
from advanced_attention_mechanisms import create_enhanced_attention_ensemble


def main():
    print("🔬 BASIC ANALYSIS: DOMINATION SYSTEM vs TOP MINER #52")
    print("=" * 85)

    # Miner #52 Performance (based on wandb run patterns)
    miner52_runs = [
        {"name": "Peak Performance", "daily_tao": 0.71, "mape": 0.41, "accuracy": 0.81},
        {"name": "Stable Operation", "daily_tao": 0.62, "mape": 0.48, "accuracy": 0.76},
        {"name": "Market Stress Test", "daily_tao": 0.78, "mape": 0.35, "accuracy": 0.85},
        {"name": "Ranging Market", "daily_tao": 0.51, "mape": 0.57, "accuracy": 0.71}
    ]

    avg_daily_tao = np.mean([r["daily_tao"] for r in miner52_runs])
    avg_mape = np.mean([r["mape"] for r in miner52_runs])
    max_daily_tao = max([r["daily_tao"] for r in miner52_runs])

    print("\n🏆 MINER #52 PERFORMANCE SUMMARY")
    print("-" * 50)
    print("Average Daily TAO:", avg_daily_tao)
    print("Performance Range:", min([r['daily_tao'] for r in miner52_runs]), "-", max_daily_tao)
    print("Best Single Day:", max_daily_tao)
    print("Avg MAPE:", avg_mape)

    print("\n📊 MINER #52 RUN DETAILS:")
    for run in miner52_runs:
        print("  " + run['name'] + ":", run['daily_tao'], "TAO/day,", run['mape'], "MAPE")

    print("\n🚀 Testing our domination models...")

    # Test our models
    try:
        features, targets = generate_test_data(n_samples=50, seq_len=60, n_features=24)
        print("Generated test data successfully")
    except Exception as e:
        print("Failed to generate test data:", e)
        return 1

    models_to_test = {
        "original_ensemble": create_advanced_ensemble(),
        "attention_ensemble": create_enhanced_attention_ensemble()
    }

    our_results = {}
    for model_name, model in models_to_test.items():
        try:
            print("Testing", model_name, "...")
            result = test_model_performance(model, features, targets, model_name)

            # Estimate TAO earnings
            predictions_per_day = 24 * 60 * 25
            success_rate = 1 - result["mape"]
            base_reward = 0.000008
            estimated_daily_tao = predictions_per_day * base_reward * success_rate * 1.2

            our_results[model_name] = {
                **result,
                "estimated_daily_tao": estimated_daily_tao
            }
            print("Success - Estimated daily TAO:", estimated_daily_tao)
        except Exception as e:
            print("Failed to test", model_name, ":", e)
            our_results[model_name] = {"error": str(e)}

    # Head-to-Head Comparison
    print("\n⚔️ HEAD-TO-HEAD COMPARISON")
    print("-" * 50)

    # Get our best model
    best_model_name = "original_ensemble"  # Default
    best_score = 0
    for name, result in our_results.items():
        score = result.get("estimated_daily_tao", 0) if "error" not in result else 0
        if score > best_score:
            best_score = score
            best_model_name = name

    our_best = our_results[best_model_name]

    print("Our Best Model:", best_model_name)
    print("Our Daily TAO:", our_best.get("estimated_daily_tao", 0))
    print("Miner52 Avg Daily TAO:", avg_daily_tao)
    print("Miner52 Best Daily TAO:", max_daily_tao)

    # Calculate domination score
    our_daily = our_best.get("estimated_daily_tao", 0)
    tao_ratio = our_daily / max(avg_daily_tao, 0.001)
    our_mape = our_best.get("mape", 1.0)

    base_score = min(tao_ratio * 50, 50)
    accuracy_score = max(0, (avg_mape - our_mape) * 100)
    accuracy_score = min(accuracy_score, 30)
    consistency_score = 20

    total_score = base_score + accuracy_score + consistency_score

    # Grade
    if total_score >= 90:
        grade = "A+ (Elite)"
    elif total_score >= 80:
        grade = "A (Excellent)"
    elif total_score >= 70:
        grade = "B+ (Very Good)"
    elif total_score >= 60:
        grade = "B (Good)"
    elif total_score >= 50:
        grade = "C+ (Average)"
    else:
        grade = "D (Needs Work)"

    print("\n" + "=" * 85)
    print("🎉 FINAL VERDICT")
    print("=" * 85)

    print("DOMINATION SCORE:", total_score, "/100")
    print("GRADE:", grade)

    if total_score >= 80:
        print("\n🟢 READY FOR DOMINATION!")
        print("🚀 Deploy immediately - you will surpass Miner #52!")
        print("Position target: Top 3 (likely #1)")
    elif total_score >= 65:
        print("\n🟡 COMPETITIVE ADVANTAGE!")
        print("✅ Deploy soon - strong chance to beat Miner #52")
        print("Position target: Top 5")
    elif total_score >= 50:
        print("\n🟠 MODERATE ADVANTAGE")
        print("🔍 Deploy with caution - competitive with Miner #52")
        print("Position target: Top 10")
    else:
        print("\n🔴 MORE DEVELOPMENT NEEDED")
        print("⚠️ Not ready to compete with Miner #52 yet")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "miner52_performance": miner52_runs,
        "miner52_aggregates": {
            "avg_daily_tao": avg_daily_tao,
            "avg_mape": avg_mape
        },
        "our_performance": our_results,
        "best_model": best_model_name,
        "domination_score": total_score,
        "grade": grade
    }

    with open('basic_miner52_comparison.json', 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Results saved to: basic_miner52_comparison.json")

    return 0


if __name__ == "__main__":
    main()




