#!/bin/bash
# PERFORMANCE COMPARISON WITH TOP MINERS
# Run after deployment to track improvements

cd /home/ocean/SN55

echo "📊 PERFORMANCE COMPARISON: You vs Top Miners"
echo "============================================="

# Your current performance (from logs)
echo "🎯 YOUR CURRENT PERFORMANCE:"
if pm2 list | grep -q "first_place_domination.*online"; then
    # Extract metrics from logs
    PREDICTIONS=$(pm2 logs first_place_domination --lines 1000 2>/dev/null | grep "Prediction made" | wc -l)
    HIT_RATE=$(pm2 logs first_place_domination --lines 1000 2>/dev/null | grep "Precision.*✓" | wc -l)
    TOTAL_PREDS=$(pm2 logs first_place_domination --lines 1000 2>/dev/null | grep "Prediction made" | wc -l)

    if [ "$TOTAL_PREDS" -gt 0 ]; then
        HIT_PCT=$((HIT_RATE * 100 / TOTAL_PREDS))
        echo "  • Predictions made: $PREDICTIONS"
        echo "  • Hit rate (≤1%): $HIT_PCT%"
    fi

    # Rewards
    REWARDS=$(pm2 logs first_place_domination --lines 1000 2>/dev/null | grep "Avg Reward" | tail -1 | awk '{print $4}')
    if [ -n "$REWARDS" ]; then
        echo "  • Average reward: $REWARDS TAO"
    fi
else
    echo "  Miner not running - deploy first"
fi

echo ""
echo "🏆 TOP MINER BENCHMARKS (from analysis):"
echo "  • Hit rate (≤1%): 65-73%"
echo "  • Predictions/hour: 11.8"
echo "  • Interval coverage: 45-50%"
echo "  • Avg reward/prediction: 0.027 TAO"
echo ""

echo "🎯 IMPROVEMENT TARGETS:"
echo "  Week 1: Achieve 60%+ hit rate"
echo "  Week 2: Reach top 10"
echo "  Week 3: Surpass top 5"
echo "  Week 4: Claim #1 position"
echo ""

echo "💡 OPTIMIZATION TIPS:"
echo "  • If hit rate <65%: Run ./deployment/automated_retraining.sh"
echo "  • If rewards low: Check ./deployment/competition_monitor.sh"
echo "  • If coverage wrong: Adjust INTERVAL_STABILITY_FACTOR in code"
echo ""
