#!/bin/bash
# COMPETITION MONITORING DASHBOARD
# Analyze competitor performance and adapt strategy
# Usage: ./deployment/competition_monitor.sh

cd /home/ocean/SN55
source venv/bin/activate

echo "🏆 COMPETITION MONITORING DASHBOARD"
echo "=================================="
echo "Time: $(date)"
echo ""

# Get current subnet information
echo "1. SUBNET OVERVIEW:"
TOTAL_MINERS=$(btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 2>/dev/null | wc -l)
if [ "$TOTAL_MINERS" -gt 0 ]; then
    echo "Total miners on subnet 55: $TOTAL_MINERS"
else
    echo "Unable to fetch subnet data"
    exit 1
fi
echo ""

# Get top performers
echo "2. TOP 10 PERFORMERS:"
echo "Rank | UID | Stake (τ) | Emissions (τ/day) | Incentive"
echo "-----|-----|------------|-------------------|----------"
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 2>/dev/null | head -11 | tail -10 | awk '{
    rank=NR
    uid=$1
    stake=$2
    emissions=$3
    incentive=$4
    printf "%-4d | %-3s | %-10s | %-17s | %s\n", rank, uid, stake, emissions, incentive
}' || echo "Unable to fetch top performers"
echo ""

# Analyze your position
echo "3. YOUR POSITION ANALYSIS:"
if [ -f ".env.miner" ]; then
    source .env.miner
    if [ -n "$COLDKEY" ]; then
        YOUR_POSITION=$(btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 2>/dev/null | grep -n "$COLDKEY\|$MINER_HOTKEY" | head -1 | cut -d: -f1 || echo "Not found")

        if [ "$YOUR_POSITION" != "Not found" ] && [ -n "$YOUR_POSITION" ]; then
            echo "Your current position: #$YOUR_POSITION"

            # Performance comparison
            TOP_PERFORMER=$(btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 2>/dev/null | head -2 | tail -1 | awk '{print $4}' || echo "N/A")
            YOUR_PERFORMANCE=$(btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 2>/dev/null | sed -n "${YOUR_POSITION}p" | awk '{print $4}' || echo "N/A")

            echo "Top performer incentive: $TOP_PERFORMER"
            echo "Your incentive: $YOUR_PERFORMER"

            if [ "$YOUR_PERFORMANCE" != "N/A" ] && [ "$TOP_PERFORMER" != "N/A" ]; then
                PERFORMANCE_RATIO=$(echo "scale=2; $YOUR_PERFORMANCE / $TOP_PERFORMER" | bc -l 2>/dev/null || echo "0")
                PERFORMANCE_PCT=$(echo "scale=0; $PERFORMANCE_RATIO * 100 / 1" | bc -l 2>/dev/null || echo "0")
                echo "Performance vs #1: ${PERFORMANCE_PCT}%"
            fi
        else
            echo "Your miner not found in subnet"
        fi
    fi
fi
echo ""

# Competition analysis
echo "4. COMPETITION ANALYSIS:"
# Get incentive distribution
INCENTIVES=$(btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 2>/dev/null | awk '{print $4}' | grep -v "incentive" | sort -nr | head -20)

if [ -n "$INCENTIVES" ]; then
    echo "Top 20 incentives distribution:"
    echo "$INCENTIVES" | awk '{printf "%.6f ", $1} END {print ""}'
    echo ""

    # Calculate competition metrics
    TOTAL_INCENTIVE=$(echo "$INCENTIVES" | awk '{sum+=$1} END {print sum}')
    TOP_5_TOTAL=$(echo "$INCENTIVES" | head -5 | awk '{sum+=$1} END {print sum}')
    TOP_5_PCT=$(echo "scale=1; $TOP_5_TOTAL / $TOTAL_INCENTIVE * 100" | bc -l 2>/dev/null || echo "0")

    echo "Competition Insights:"
    echo "• Top 5 miners capture: ${TOP_5_PCT}% of total incentives"
    echo "• Incentive concentration: $([ "$(echo "$TOP_5_PCT > 50" | bc -l)" -eq 1 ] && echo "HIGH" || echo "MODERATE")"

    # Calculate Gini coefficient (inequality measure)
    SORTED_INCENTIVES=$(echo "$INCENTIVES" | sort -n)
    N=$(echo "$SORTED_INCENTIVES" | wc -l)
    if [ "$N" -gt 1 ]; then
        MEAN=$(echo "$SORTED_INCENTIVES" | awk '{sum+=$1} END {print sum/NR}')
        GINI_NUMERATOR=0
        GINI_DENOMINATOR=0
        I=1
        while read -r VAL; do
            GINI_NUMERATOR=$(echo "$GINI_NUMERATOR + ($I - 0.5) * $VAL" | bc -l)
            GINI_DENOMINATOR=$(echo "$GINI_DENOMINATOR + $VAL" | bc -l)
            I=$(($I + 1))
        done <<< "$SORTED_INCENTIVES"

        GINI=$(echo "scale=3; 1 - (2 * $GINI_NUMERATOR) / ($N * $GINI_DENOMINATOR)" | bc -l 2>/dev/null || echo "0.5")
        echo "• Incentive inequality (Gini): $GINI ($(if [ "$(echo "$GINI > 0.7" | bc -l)" -eq 1 ]; then echo "HIGH"; elif [ "$(echo "$GINI > 0.5" | bc -l)" -eq 1 ]; then echo "MODERATE"; else echo "LOW"; fi))"
    fi
else
    echo "Unable to analyze competition"
fi
echo ""

# Strategy recommendations
echo "5. STRATEGY RECOMMENDATIONS:"

if [ -n "$YOUR_POSITION" ] && [ "$YOUR_POSITION" != "Not found" ]; then
    if [ "$YOUR_POSITION" -le 5 ]; then
        echo "🎯 EXCELLENT: You're in top 5!"
        echo "   Strategy: Focus on maintaining position with conservative optimization"
        echo "   Action: Monitor closely, small parameter adjustments only"
    elif [ "$YOUR_POSITION" -le 20 ]; then
        echo "👍 GOOD: You're in top 20!"
        echo "   Strategy: Aggressive optimization to reach top 10"
        echo "   Action: Enable competition intelligence, increase prediction frequency"
    elif [ "$YOUR_POSITION" -le 50 ]; then
        echo "⚡ IMPROVING: Top 50 position"
        echo "   Strategy: Implement advanced features, focus on hit rate"
        echo "   Action: Deploy enhanced miner, enable online learning"
    else
        echo "🚀 CLIMBING: Outside top 50"
        echo "   Strategy: Deploy first-place miner, focus on fundamentals"
        echo "   Action: ./deployment/deploy_first_place_miner.sh"
    fi
else
    echo "📍 Position unknown - miner may not be registered or running"
    echo "   Action: Check registration and restart miner"
fi
echo ""

# Market analysis
echo "6. MARKET REGIME ANALYSIS:"
# Simple market analysis based on recent price action
# This is a placeholder - in real implementation would use actual market data

CURRENT_HOUR=$(date +%H)
if [ "$CURRENT_HOUR" -ge 9 ] && [ "$CURRENT_HOUR" -le 11 ]; then
    echo "🌅 MARKET: London session active (high volatility expected)"
elif [ "$CURRENT_HOUR" -ge 13 ] && [ "$CURRENT_HOUR" -le 15 ]; then
    echo "🌎 MARKET: US session active (peak activity time)"
elif [ "$CURRENT_HOUR" -ge 0 ] && [ "$CURRENT_HOUR" -le 6 ]; then
    echo "🌙 MARKET: Asian session (lower volatility)"
else
    echo "📊 MARKET: Off-peak hours (moderate activity)"
fi
echo ""

# Performance targets
echo "7. PERFORMANCE TARGETS:"
echo "Daily targets based on your current position:"
if [ -n "$YOUR_POSITION" ] && [ "$YOUR_POSITION" != "Not found" ]; then
    if [ "$YOUR_POSITION" -le 10 ]; then
        echo "• Maintain position: >0.1 TAO daily"
        echo "• Hit rate target: >52%"
        echo "• MAPE target: <0.09%"
    elif [ "$YOUR_POSITION" -le 50 ]; then
        echo "• Reach top 10: >0.15 TAO daily"
        echo "• Hit rate target: >50%"
        echo "• MAPE target: <0.10%"
    else
        echo "• Reach top 50: >0.05 TAO daily"
        echo "• Hit rate target: >45%"
        echo "• MAPE target: <0.12%"
    fi
else
    echo "• Registration target: Get on subnet"
    echo "• First rewards: >0.01 TAO daily"
    echo "• Stability target: Consistent predictions"
fi
echo ""

echo "=========================================="
echo "📋 NEXT ACTIONS:"
echo "   • Monitor position: ./deployment/competition_monitor.sh"
echo "   • Check performance: ./deployment/monitor_precog.sh"
echo "   • Retrain if needed: ./deployment/automated_retraining.sh"
echo "   • Backup regularly: ./deployment/backup_and_recover.sh"
echo "=========================================="
