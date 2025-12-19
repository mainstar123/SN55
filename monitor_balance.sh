#!/bin/bash

# Monitor Bittensor testnet balance
# Usage: ./monitor_balance.sh

cd /home/ocean/nereus/precog
source venv/bin/activate

export BITTENSOR_CONFIG_DIR=/home/ocean/.bittensor
export HOME=/home/ocean

echo "🔍 MONITORING TESTNET BALANCE..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Checking every 30 seconds... (Ctrl+C to stop)"
echo ""

while true; do
    echo "$(date '+%H:%M:%S') - Checking balance..."
    BALANCE=$(btcli wallet balance --wallet-path /home/ocean/.bittensor/wallets --wallet-name cold_draven --network finney 2>/dev/null | grep -o "‎[0-9]*\.[0-9]*" | head -1)

    if [[ $BALANCE =~ ^[0-9]+\.[0-9]+$ ]] && (( $(echo "$BALANCE > 0" | bc -l) )); then
        echo "🎉 BALANCE DETECTED: $BALANCE τ"
        echo ""
        echo "🚀 READY TO REGISTER!"
        echo "━━━━━━━━━━━━━━━━━━━━"
        echo "btcli subnet register --netuid 55 --wallet-path /home/ocean/.bittensor/wallets --wallet-name cold_draven --wallet-hotkey default --network finney --yes"
        echo ""
        echo "Then deploy: make miner_custom ENV_FILE=.env.miner.testnet"
        break
    else
        echo "Balance: 0.0000 τ (still waiting...)"
    fi

    sleep 30
done
