#!/bin/bash
# PRECOG MINER DASHBOARD
# Comprehensive monitoring script
# Usage: ./deployment/monitor_precog.sh

cd /home/ocean/SN55
source venv/bin/activate

echo "=========================================="
echo "📊 PRECOG MINER DASHBOARD"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Miner Status
echo "1. MINER STATUS:"
pm2 jlist 2>/dev/null | jq -r '.[] | select(.name | contains("precog")) | "\(.name): \(.pm2_env.status) (PID: \(.pid))"' 2>/dev/null || echo "No precog miners found"

# Check process health
MINER_PID=$(pgrep -f miner.py || echo "")
if [ -n "$MINER_PID" ]; then
    echo "✅ Miner process running (PID: $MINER_PID)"
else
    echo "❌ Miner process not found"
fi
echo ""

# Wallet Balance
echo "2. WALLET BALANCE:"
if [ -f ".env.miner" ]; then
    source .env.miner
    if [ -n "$COLDKEY" ]; then
        BALANCE=$(btcli wallet overview --wallet.name $COLDKEY 2>/dev/null | grep -E "(Balance|τ)" | head -1 || echo "Unable to check balance")
        echo "$COLDKEY: $BALANCE"
    else
        echo "❌ Wallet name not configured in .env.miner"
    fi
else
    echo "❌ .env.miner not found"
fi
echo ""

# Recent Predictions
echo "3. PREDICTIONS (Last Hour):"
if pm2 list | grep -q "precog.*online"; then
    PREDICTIONS=$(pm2 logs precog_best_miner --lines 1000 2>/dev/null | grep "Prediction made" | wc -l || echo "0")
    RECENT_PREDS=$(pm2 logs precog_best_miner --lines 100 2>/dev/null | grep "Prediction made" | tail -5 || echo "No recent predictions")
    echo "Total predictions (last 1000 lines): $PREDICTIONS"
    echo "Recent predictions:"
    echo "$RECENT_PREDS"
else
    echo "Miner not running"
fi
echo ""

# Performance Metrics
echo "4. PERFORMANCE METRICS:"
if pm2 list | grep -q "precog.*online"; then
    # Response time
    RESPONSE_TIME=$(pm2 logs precog_best_miner --lines 500 2>/dev/null | grep "response_time" | tail -10 | awk '{sum+=$2; count++} END {if(count>0) print sum/count " ms avg"; else print "No data"}' || echo "No data")
    echo "Average response time: $RESPONSE_TIME"

    # Success rate
    SUCCESS_RATE=$(pm2 logs precog_best_miner --lines 500 2>/dev/null | grep -c "success\|Success" || echo "0")
    TOTAL_REQUESTS=$(pm2 logs precog_best_miner --lines 500 2>/dev/null | grep -c "synapse\|request" || echo "0")
    if [ "$TOTAL_REQUESTS" -gt 0 ]; then
        SUCCESS_PCT=$((SUCCESS_RATE * 100 / TOTAL_REQUESTS))
        echo "Success rate: $SUCCESS_PCT% ($SUCCESS_RATE/$TOTAL_REQUESTS)"
    fi

    # MAPE if available
    MAPE=$(pm2 logs precog_best_miner --lines 500 2>/dev/null | grep "MAPE" | tail -1 | awk '{print $2}' || echo "Not available")
    echo "Latest MAPE: $MAPE"
else
    echo "Miner not running"
fi
echo ""

# Subnet Position
echo "5. SUBNET POSITION:"
if [ -f ".env.miner" ]; then
    source .env.miner
    if [ -n "$COLDKEY" ]; then
        YOUR_UID=$(btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 2>/dev/null | grep -n "$COLDKEY\|$MINER_HOTKEY" | head -1 | cut -d: -f1 || echo "Not found")
        if [ "$YOUR_UID" != "Not found" ] && [ -n "$YOUR_UID" ]; then
            echo "Your position: #$YOUR_UID"
            # Show top 5 for context
            echo "Top 5 miners:"
            btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 2>/dev/null | head -5 || echo "Unable to fetch"
        else
            echo "Your miner not found in subnet"
        fi
    fi
fi
echo ""

# System Resources
echo "6. SYSTEM RESOURCES:"
# CPU usage
CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')
echo "CPU Usage: $CPU"

# Memory usage
MEM=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')
echo "Memory Usage: $MEM"

# Disk space
DISK=$(df / | tail -1 | awk '{print $5}')
echo "Disk Usage: $DISK"

# GPU if available
if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    echo "GPU Usage: ${GPU}%"
fi
echo ""

# Network Status
echo "7. NETWORK STATUS:"
# Check connectivity to mainnet
if ping -c 1 -W 2 archive.substrate.network >/dev/null 2>&1; then
    echo "✅ Mainnet connectivity: OK"
else
    echo "❌ Mainnet connectivity: FAILED"
fi

# Check local axon
if [ -f ".env.miner" ]; then
    source .env.miner
    if [ -n "$MINER_PORT" ]; then
        if curl -s --max-time 2 http://localhost:$MINER_PORT/ping >/dev/null 2>&1; then
            echo "✅ Local axon (port $MINER_PORT): OK"
        else
            echo "❌ Local axon (port $MINER_PORT): FAILED"
        fi
    fi
fi

# Active connections
CONNECTIONS=$(netstat -t | grep ESTABLISHED | wc -l)
echo "Active connections: $CONNECTIONS"
echo ""

# Recommendations
echo "8. RECOMMENDATIONS:"
if pm2 list | grep -q "precog.*stopped\|errored"; then
    echo "🔴 ACTION NEEDED: Miner is not running"
    echo "   Run: ./deployment/deploy_best_model.sh"
elif [ "$PREDICTIONS" = "0" ] 2>/dev/null; then
    echo "🟡 MONITOR: No predictions in recent logs"
    echo "   Check: pm2 logs precog_best_miner --follow"
elif (( $(echo "$CPU > 90" | bc -l) )); then
    echo "🟡 OPTIMIZE: High CPU usage detected"
    echo "   Consider: Reducing model complexity or upgrading hardware"
elif (( $(echo "$MEM > 90" | bc -l) )); then
    echo "🟡 OPTIMIZE: High memory usage detected"
    echo "   Consider: Model quantization or memory optimization"
else
    echo "✅ STATUS: All systems normal"
    echo "   Continue monitoring and consider daily retraining"
fi
echo ""

echo "=========================================="
echo "📋 QUICK COMMANDS:"
echo "   • Full logs: pm2 logs precog_best_miner --follow"
echo "   • Restart: pm2 restart precog_best_miner"
echo "   • Retrain: ./deployment/automated_retraining.sh"
echo "   • Backup: ./deployment/backup_and_recover.sh"
echo "=========================================="
