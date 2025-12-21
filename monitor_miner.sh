#!/bin/bash
# Real-time Precog Miner Monitor

cd /home/ocean/nereus/precog
source venv/bin/activate

export HOME=/home/ocean
export BITTENSOR_CONFIG_DIR=/home/ocean/.bittensor

while true; do
    clear
    echo "🔥 PRECOG MINER MONITOR - TESTNET SUBNET 55"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Time: $(date)"
    echo ""

    # Miner Status
    echo "⚙️  MINER STATUS:"
    echo "━━━━━━━━━━━━━━━━"
    if ps aux | grep -v grep | grep "miner.py" > /dev/null; then
        MINER_PID=$(ps aux | grep "miner.py" | grep -v grep | awk '{print $2}')
        CPU_USAGE=$(ps aux | grep "miner.py" | grep -v grep | awk '{print $3}')
        MEM_USAGE=$(ps aux | grep "miner.py" | grep -v grep | awk '{print $4}')
        echo "✅ Miner is RUNNING (PID: $MINER_PID)"
        echo "   CPU: ${CPU_USAGE}% | Memory: ${MEM_USAGE}MB"
    else
        echo "❌ Miner is NOT running"
        echo "   💡 Start with: ./start_miner.sh"
    fi
    echo ""

    # Wallet Balance
    echo "💰 WALLET BALANCE (Testnet):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━"
    BALANCE=$(btcli wallet balance --wallet.name cold_draven --subtensor.network test 2>/dev/null | grep -o "[0-9]*\.[0-9]* τ" | head -1 || echo "Check failed")
    echo "Balance: $BALANCE"
    echo ""

    # Registration Status
    echo "📝 REGISTRATION STATUS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━"
    OVERVIEW=$(btcli wallet overview --wallet.name cold_draven --subtensor.network test 2>/dev/null | grep -A2 -B2 "55:" || echo "Check failed")
    if echo "$OVERVIEW" | grep -q "55:"; then
        echo "✅ Registered on subnet 55 (Precog)"
        echo "$OVERVIEW" | grep -A2 "55:"
    else
        echo "❌ Not registered on subnet 55"
    fi
    echo ""

    # Recent Logs
    echo "📋 RECENT LOGS:"
    echo "━━━━━━━━━━━━━━━"
    if [ -f miner.log ]; then
        tail -5 miner.log | head -10
    else
        echo "No log file yet..."
    fi
    echo ""

    # Performance Metrics
    echo "📊 PERFORMANCE METRICS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━"
    if [ -f miner.log ] && [ $(wc -l < miner.log) -gt 0 ]; then
        PREDICTIONS=$(grep -c "prediction" miner.log 2>/dev/null || echo "0")
        REQUESTS=$(grep -c "request" miner.log 2>/dev/null || echo "0")
        echo "Predictions made: $PREDICTIONS"
        echo "Requests processed: $REQUESTS"
    else
        echo "Waiting for first predictions..."
    fi
    echo ""

    echo "🔄 Refreshing in 10 seconds... (Ctrl+C to exit)"
    sleep 10
done
