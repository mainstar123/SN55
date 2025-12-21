#!/bin/bash
# Monitor Domination Miner Progress
# Track journey to #1 position

cd /home/ocean/nereus/precog

echo "📊 DOMINATION MINER MONITOR"
echo "============================"
echo "🎯 Tracking progress to #1 position"
echo ""

# Check if miner is running
MINER_PID=$(cat miner_domination.pid 2>/dev/null)
if ps -p $MINER_PID > /dev/null 2>&1; then
    echo "✅ Domination miner is running (PID: $MINER_PID)"
    echo ""

    # Show current time and peak hour status
    CURRENT_HOUR=$(date -u +%H)
    if [[ "$CURRENT_HOUR" =~ ^(09|10|13|14)$ ]]; then
        echo "⏰ CURRENT STATUS: PEAK HOUR ACTIVE ($CURRENT_HOUR UTC)"
        echo "⚡ 3x prediction frequency enabled"
    else
        echo "⏰ CURRENT STATUS: Normal hour ($CURRENT_HOUR UTC)"
        echo "📊 Standard prediction frequency"
    fi
    echo ""

    # Show recent logs
    echo "📝 RECENT ACTIVITY:"
    echo "-------------------"
    tail -10 miner_domination.log | grep -E "(DOMINATION|Market|Prediction|Performance|TARGET|synapse)" | tail -5

    echo ""
    echo "🎯 DOMINATION TARGETS:"
    echo "• Hour 12: 0.08+ TAO (Surpass UID 31)"
    echo "• Hour 24: 0.12+ TAO (Enter Top 3)"
    echo "• Hour 48: 0.15+ TAO (Dominate UID 31)"
    echo ""

    echo "🔍 CONTINUOUS MONITORING:"
    echo "tail -f miner_domination.log"
    echo ""
    echo "📊 PERFORMANCE CHECK:"
    echo "grep 'Performance Update' miner_domination.log | tail -5"
    echo ""
    echo "🏆 ACHIEVEMENT CHECK:"
    echo "grep 'TARGET ACHIEVED' miner_domination.log"

else
    echo "❌ Domination miner is not running"
    echo ""
    echo "🚀 RESTART COMMAND:"
    echo "./start_domination_miner.sh"
    echo ""
    echo "📊 CHECK LOGS:"
    echo "tail -50 miner_domination.log"
fi

echo ""
echo "⚡ DOMINATION FEATURES ACTIVE:"
echo "• Peak hour optimization (9-11, 13-15 UTC)"
echo "• Market regime detection"
echo "• Ensemble predictions"
echo "• Adaptive thresholds"
echo "• Performance tracking"
