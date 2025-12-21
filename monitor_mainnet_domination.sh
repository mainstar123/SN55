#!/bin/bash
# MONITOR MAINNET DOMINATION MINER
# Track progress toward #1 position on mainnet

cd /home/ocean/nereus/precog

echo "🌐 MAINNET DOMINATION MONITOR"
echo "=============================="
echo "🎯 Tracking mainnet progress to #1 position"
echo ""

# Check if mainnet miner is running
MINER_PID=$(cat miner_mainnet_domination.pid 2>/dev/null)
if ps -p $MINER_PID > /dev/null 2>&1; then
    echo "✅ Mainnet domination miner is running (PID: $MINER_PID)"
    echo ""

    # Show current time and peak hour status
    CURRENT_HOUR=$(date -u +%H)
    if [[ "$CURRENT_HOUR" =~ ^(09|10|13|14)$ ]]; then
        echo "⏰ CURRENT STATUS: PEAK HOUR ACTIVE ($CURRENT_HOUR UTC)"
        echo "⚡ 3x prediction frequency enabled on mainnet"
    else
        echo "⏰ CURRENT STATUS: Normal hour ($CURRENT_HOUR UTC)"
        echo "📊 Standard prediction frequency"
    fi
    echo ""

    # Show recent mainnet activity
    echo "📝 RECENT MAINNET ACTIVITY:"
    echo "---------------------------"
    tail -10 miner_mainnet_domination.log | grep -E "(DOMINATION|Market|Prediction|Performance|TARGET|synapse)" | tail -5

    # Show performance summary
    echo ""
    echo "📊 PERFORMANCE SUMMARY:"
    echo "-----------------------"
    PREDICTIONS=$(grep -c "Prediction made" miner_mainnet_domination.log 2>/dev/null || echo "0")
    AVG_REWARD=$(grep "Avg Reward:" miner_mainnet_domination.log | tail -1 | grep -o "[0-9.]*" || echo "0.000")
    TARGETS_ACHIEVED=$(grep -c "TARGET ACHIEVED" miner_mainnet_domination.log 2>/dev/null || echo "0")

    echo "• Total Predictions: $PREDICTIONS"
    echo "• Current Avg Reward: ${AVG_REWARD} TAO"
    echo "• Targets Achieved: $TARGETS_ACHIEVED"

    echo ""
    echo "🎯 MAINNET DOMINATION TARGETS:"
    echo "• Hour 12: 0.08+ TAO (Surpass UID 31) - $(if (( $(echo "$AVG_REWARD >= 0.08" | bc -l) )); then echo "✅ ACHIEVED"; else echo "⏳ Pending"; fi)"
    echo "• Hour 24: 0.12+ TAO (Enter Top 3) - $(if (( $(echo "$AVG_REWARD >= 0.12" | bc -l) )); then echo "✅ ACHIEVED"; else echo "⏳ Pending"; fi)"
    echo "• Hour 48: 0.15+ TAO (Dominate UID 31) - $(if (( $(echo "$AVG_REWARD >= 0.15" | bc -l) )); then echo "✅ ACHIEVED"; else echo "⏳ Pending"; fi)"
    echo ""

    echo "🔍 MAINNET MONITORING COMMANDS:"
    echo "tail -f miner_mainnet_domination.log"
    echo ""
    echo "📊 Performance check:"
    echo "grep 'Performance Update' miner_mainnet_domination.log | tail -5"
    echo ""
    echo "🏆 Achievement check:"
    echo "grep 'TARGET ACHIEVED' miner_mainnet_domination.log"

else
    echo "❌ Mainnet domination miner is not running"
    echo ""
    echo "🚀 RESTART COMMAND:"
    echo "./start_mainnet_domination_miner.sh"
    echo ""
    echo "📊 CHECK LOGS:"
    echo "tail -50 miner_mainnet_domination.log"
    echo ""
    echo "💡 If mainnet connection issues:"
    echo "   • Check wallet: btcli wallet overview"
    echo "   • Check registration: btcli subnet list --netuid 55"
    echo "   • Try testnet first: ./start_domination_miner.sh"
fi

echo ""
echo "⚡ MAINNET DOMINATION FEATURES ACTIVE:"
echo "• Peak hour optimization (9-11, 13-15 UTC)"
echo "• Market regime detection"
echo "• Ensemble predictions (trained model)"
echo "• Adaptive thresholds"
echo "• Real-time mainnet performance tracking"
echo "• UID 31 domination targeting"
