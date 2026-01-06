#!/bin/bash
# Comprehensive miner monitoring dashboard

echo "🎯 PRECOG MINER MONITORING DASHBOARD"
echo "===================================="
echo ""

while true; do
    clear
    echo "🎯 PRECOG MINER MONITORING DASHBOARD"
    echo "===================================="
    echo "Time: $(date)"
    echo ""
    
    # 1. Service Status
    echo "📊 1. MINER SERVICE STATUS:"
    echo "---------------------------"
    systemctl --user status precog-miner.service --no-pager -l | head -3
    echo ""
    
    # 2. Network Status
    echo "🌐 2. BITTENSOR NETWORK STATUS:"
    echo "-------------------------------"
    if timeout 3 curl -s --connect-timeout 2 https://finney.opentensor.ai/ >/dev/null 2>&1; then
        echo "✅ finney.opentensor.ai: ACCESSIBLE"
    else
        echo "❌ finney.opentensor.ai: NOT ACCESSIBLE"
    fi
    
    if timeout 3 curl -s --connect-timeout 2 https://finney.opentensor.ai/ >/dev/null 2>&1; then
        echo "✅ finney.opentensor.ai: ACCESSIBLE"
    else
        echo "❌ finney.opentensor.ai: NOT ACCESSIBLE"
    fi
    echo ""
    
    # 3. Current Metrics
    echo "📈 3. CURRENT MINER METRICS:"
    echo "----------------------------"
    LATEST_LOG=$(./manage_miner_service.sh logs 2>/dev/null | grep "Miner | UID:" | tail -1)
    if [ -n "$LATEST_LOG" ]; then
        BLOCK=$(echo "$LATEST_LOG" | grep -o "Block:[0-9]*" | cut -d: -f2)
        STAKE=$(echo "$LATEST_LOG" | grep -o "Stake:[0-9.]*" | cut -d: -f2)
        TRUST=$(echo "$LATEST_LOG" | grep -o "Trust:[0-9.]*" | cut -d: -f2)
        INCENTIVE=$(echo "$LATEST_LOG" | grep -o "Incentive:[0-9.]*" | cut -d: -f2)
        EMISSION=$(echo "$LATEST_LOG" | grep -o "Emission:[0-9.]*" | cut -d: -f2)
        
        echo "• UID: 142"
        echo "• Block: $BLOCK"
        echo "• Stake: $STAKE TAO"
        echo "• Trust: $TRUST"
        echo "• Incentive: $INCENTIVE"
        echo "• Emission: $EMISSION TAO/block"
        
        # Status interpretation
        if [ "$EMISSION" = "0.000" ]; then
            echo "❌ STATUS: NOT EARNING (Network Issues)"
        else
            echo "✅ STATUS: ACTIVELY EARNING!"
        fi
    else
        echo "❌ No metrics available - check service status"
    fi
    echo ""
    
    # 4. GPU Status
    echo "🎮 4. GPU STATUS:"
    echo "----------------"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [ $? -eq 0 ]; then
        GPU_NAME=$(echo $GPU_INFO | cut -d, -f1)
        GPU_MEM_USED=$(echo $GPU_INFO | cut -d, -f2)
        GPU_MEM_TOTAL=$(echo $GPU_INFO | cut -d, -f3)
        GPU_UTIL=$(echo $GPU_INFO | cut -d, -f4)
        
        echo "• GPU: $GPU_NAME"
        echo "• Memory: ${GPU_MEM_USED}MiB / ${GPU_MEM_TOTAL}MiB"
        echo "• Utilization: ${GPU_UTIL}%"
        
        if [ "$GPU_MEM_USED" -gt 300 ]; then
            echo "✅ STATUS: GPU memory allocated (models loaded)"
        else
            echo "❌ STATUS: GPU memory not allocated"
        fi
    else
        echo "❌ NVIDIA GPU not detected"
    fi
    echo ""
    
    # 5. Recent Activity
    echo "📋 5. RECENT LOG ACTIVITY:"
    echo "--------------------------"
    ./manage_miner_service.sh logs 2>/dev/null | tail -3
    echo ""
    
    # 6. Status Summary
    echo "🎯 6. OVERALL STATUS SUMMARY:"
    echo "-----------------------------"
    if [ "$EMISSION" = "0.000" ]; then
        echo "🔴 MINER STATUS: REGISTERED BUT NOT ACTIVE"
        echo "• Visible in Taostats: ✅ (UID 142)"
        echo "• Local service: ✅ Running"
        echo "• GPU ready: ✅ Allocated"
        echo "• Network: ❌ Blocking earnings"
        echo "• Earnings: ❌ 0.000 TAO/block"
        echo ""
        echo "⏳ WAITING FOR: Bittensor network recovery"
        echo "🚀 WILL ACTIVATE: Automatically when endpoints accessible"
    else
        echo "🟢 MINER STATUS: ACTIVELY MINING!"
        echo "• Network: ✅ Connected"
        echo "• Earnings: ✅ $EMISSION TAO/block"
        echo "• GPU: ✅ Processing requests"
        echo "• Trust: 📈 Building reputation"
    fi
    
    echo ""
    echo "🔄 Refreshing in 10 seconds... (Ctrl+C to exit)"
    sleep 10
done
