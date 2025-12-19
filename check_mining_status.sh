#!/bin/bash
# Comprehensive Precog Testnet Mining Status Checker
# Run this script to get complete overview of your mining performance

echo "========================================"
echo "🔍 PRECOG TESTNET MINING STATUS CHECKER"
echo "========================================"
echo ""

# Set environment
cd /home/ocean/nereus/precog
source venv/bin/activate
export BITTENSOR_CONFIG_DIR=/home/ocean/.bittensor
export HOME=/home/ocean

echo "📅 $(date)"
echo ""

echo "💰 WALLET STATUS (Subnet 55 - Precog Testnet):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
btcli wallet overview --netuid 55 --subtensor.network finney
echo ""

echo "📊 DETAILED BALANCE:"
echo "━━━━━━━━━━━━━━━━━━━━"
btcli wallet balance --wallet.name cold_draven --subtensor.network finney
echo ""

echo "🏆 YOUR POSITION IN METAGRAPH:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Get your hotkey address
YOUR_HOTKEY=$(btcli wallet list | grep -A 1 "Hotkey default" | tail -1 | awk '{print $NF}')
echo "Your Hotkey: $YOUR_HOTKEY"
echo ""

# Check metagraph for your position
METAGRAPH_INFO=$(btcli subnet metagraph --netuid 55 --subtensor.network finney | grep -n "$YOUR_HOTKEY")
if [ ! -z "$METAGRAPH_INFO" ]; then
    LINE_NUM=$(echo $METAGRAPH_INFO | cut -d: -f1)
    echo "Your Rank: #$LINE_NUM in metagraph"
    echo "Details: $METAGRAPH_INFO"
else
    echo "❌ Your hotkey not found in metagraph (not registered or not mining)"
fi
echo ""

echo "🌐 SUBNET HEALTH:"
echo "━━━━━━━━━━━━━━━━━"
btcli subnet list --subtensor.network finney | grep -A 2 -B 2 "55"
echo ""

echo "⚙️  MINER STATUS:"
echo "━━━━━━━━━━━━━━━━━"
if pm2 list | grep -q precog_testnet_miner; then
    echo "✅ Miner is running"
    pm2 jlist | jq -r '.[] | select(.name=="precog_testnet_miner") | "PID: \(.pid), CPU: \(.monit.cpu)%, Memory: \(.monit.memory)MB, Status: \(.pm2_env.status)"'
else
    echo "❌ Miner is not running"
fi
echo ""

echo "📈 RECENT LOGS:"
echo "━━━━━━━━━━━━━━━━"
pm2 logs precog_testnet_miner --lines 5 --nostream
echo ""

echo "🎯 PERFORMANCE METRICS (Last Hour):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/validate_performance.py --hours 1 2>/dev/null || echo "Performance script not available or no data yet"
echo ""

echo "🏅 TOP 10 COMPETITORS:"
echo "━━━━━━━━━━━━━━━━━━━━━━"
python scripts/monitor_competitors.py --netuid 55 --top-n 10 2>/dev/null || echo "Competitor monitoring not available"
echo ""

echo "📋 QUICK STATUS SUMMARY:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Wallet: $(btcli wallet balance --wallet.name cold_draven --subtensor.network finney | grep -o '[0-9.]* TAO' | head -1)"
echo "✅ Incentive: $(btcli wallet overview --netuid 55 --subtensor.network finney | grep -o 'Incentive: [0-9.]*' | head -1)"
echo "✅ Trust: $(btcli wallet overview --netuid 55 --subtensor.network finney | grep -o 'Trust: [0-9.]*' | head -1)"
echo "✅ Emission: $(btcli wallet overview --netuid 55 --subtensor.network finney | grep -o 'Emission: [0-9.]*' | head -1)"
echo ""

echo "💡 NEXT STEPS:"
echo "━━━━━━━━━━━━━"
if pm2 list | grep -q precog_testnet_miner; then
    echo "✅ Miner is running - monitor performance over next 24 hours"
    echo "✅ Check metagraph position daily"
    echo "✅ Retrain model if accuracy below 0.15% MAPE"
else
    echo "❌ Start your miner: make miner_custom ENV_FILE=.env.miner.testnet"
fi

echo ""
echo "🔄 Run this script anytime: ./check_mining_status.sh"
