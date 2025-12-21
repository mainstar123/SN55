#!/bin/bash
# Comprehensive Precog Mining Dashboard

cd /home/ocean/nereus/precog
source venv/bin/activate

export HOME=/home/ocean
export BITTENSOR_CONFIG_DIR=/home/ocean/.bittensor

while true; do
    clear
    echo "🎯 PRECOG MINING DASHBOARD"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Time: $(date)"
    echo ""

    echo "💰 WALLET STATUS (Testnet):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━"
    btcli wallet balance --wallet.name cold_draven --subtensor.network test 2>/dev/null | grep -E "(cold_|Total)" || echo "Balance check failed"
    echo ""

    echo "🏆 YOUR POSITION (Subnet 256):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    YOUR_HOTKEY=$(btcli wallet list 2>/dev/null | grep "Hotkey default" -A1 | tail -1 | awk '{print $NF}' || echo "Unknown")
    echo "Your Hotkey: $YOUR_HOTKEY"
    
    METAGRAPH_INFO=$(btcli subnet metagraph --netuid 256 --subtensor.network test 2>/dev/null | grep -n "$YOUR_HOTKEY" || echo "Not found")
    if [ "$METAGRAPH_INFO" != "Not found" ]; then
        LINE_NUM=$(echo $METAGRAPH_INFO | cut -d: -f1)
        echo "Rank: #$LINE_NUM in subnet 256"
    else
        echo "❌ Not registered or not mining on subnet 256"
    fi
    echo ""

    echo "⚙️  MINER STATUS:"
    echo "━━━━━━━━━━━━━━━━━"
    if pgrep -f "miner.py" > /dev/null; then
        echo "✅ Miner is running"
        echo "PID: $(pgrep -f "miner.py")"
    else
        echo "❌ Miner is not running"
        echo "💡 Run: ./start_testnet_miner.sh"
    fi
    echo ""

    echo "📊 PERFORMANCE METRICS (Last Hour):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python scripts/validate_performance.py --hours 1 2>/dev/null | tail -10 || echo "Performance data not available yet"
    echo ""

    echo "🏅 TOP 5 COMPETITORS:"
    echo "━━━━━━━━━━━━━━━━━━━━"
    python scripts/monitor_competitors.py --netuid 256 --top-n 5 2>/dev/null | head -20 || echo "Competitor data not available"
    echo ""

    echo "🔄 Refreshing in 30 seconds... (Ctrl+C to exit)"
    sleep 30
done
