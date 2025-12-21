#!/bin/bash

# Wait for finney testnet TAO balance to arrive
cd /home/ocean/nereus/precog
source venv/bin/activate

export BITTENSOR_CONFIG_DIR=/home/ocean/.bittensor
export HOME=/home/ocean

echo "🔍 WAITING FOR FINNEY TESTNET TAO BALANCE..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Address: 5DJCeqFEQ59XhDK4kfxssE8jnwK3Y3Tq36SBphc1ufc6FjWf"
echo "Needed: ~0.01 τ for testnet registration"
echo "Note: 'test' network TAO ≠ 'finney' network TAO"
echo ""
echo "Checking balance every 30 seconds... (Ctrl+C to stop)"
echo ""

while true; do
    echo "$(date '+%H:%M:%S') - Checking finney balance..."

    # Extract balance from btcli output
    BALANCE_OUTPUT=$(btcli wallet balance --wallet-path /home/ocean/.bittensor/wallets --wallet-name cold_draven --network finney 2>/dev/null)
    BALANCE=$(echo "$BALANCE_OUTPUT" | grep -o "‎[0-9]*\.[0-9]*" | head -1 | sed 's/‎//g')

    if [[ $BALANCE =~ ^[0-9]+\.[0-9]+$ ]] && (( $(echo "$BALANCE >= 0.01" | bc -l) )); then
        echo ""
        echo "🎉 SUCCESS! FINNEY BALANCE DETECTED: $BALANCE τ"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Testnet registration needs ~0.01 τ"
        echo ""
        echo "🚀 READY TO REGISTER ON PRECOG SUBNET 55 (TESTNET)!"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Command:"
        echo "btcli subnet register --netuid 55 --wallet-path /home/ocean/.bittensor/wallets --wallet-name cold_draven --wallet-hotkey default --network finney --yes"
        echo ""
        echo "Then deploy miner:"
        echo "make miner_custom ENV_FILE=.env.miner.testnet"
        echo ""
        echo "🎯 YOUR ENHANCED GRU MINER IS READY TO EARN TAO!"
        break
    else
        echo "Finney balance: ${BALANCE:-0.0000} τ (need 0.01 τ)"
    fi

    sleep 30
done
