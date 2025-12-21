#!/bin/bash
# Fix Miner Setup - Complete testnet registration and startup

cd /home/ocean/nereus/precog
source venv/bin/activate

echo "🔧 FIXING MINER SETUP FOR TESTNET"
echo "=================================="

# Step 1: Kill any existing miner processes
echo "1. Stopping existing miner processes..."
pkill -f miner.py
sleep 2

# Step 2: Check wallet registration
echo "2. Checking wallet registration..."
export HOME=/home/ocean
export BITTENSOR_CONFIG_DIR=/home/ocean/.bittensor

# Check if registered
python3 -c "
import os
os.environ['HOME'] = '/home/ocean'
import bittensor as bt
wallet = bt.wallet(name='cold_draven', hotkey='default')
subtensor = bt.subtensor(network='test')
is_registered = subtensor.is_hotkey_registered(netuid=256, hotkey_ss58=wallet.hotkey.ss58_address)
print(f'Wallet registered: {is_registered}')
if not is_registered:
    print('NEEDS_REGISTRATION')
else:
    print('ALREADY_REGISTERED')
" > /tmp/reg_status.txt

REG_STATUS=$(cat /tmp/reg_status.txt | grep -E "(NEEDS_REGISTRATION|ALREADY_REGISTERED)")
echo "   Registration status: $REG_STATUS"

if echo "$REG_STATUS" | grep -q "NEEDS_REGISTRATION"; then
    echo "3. Registering wallet on testnet subnet 256..."

    # Register wallet (this costs ~0.1 TAO)
    echo "   Running: btcli subnet register --netuid 256 --wallet.name cold_draven --wallet.hotkey default --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443"

    btcli subnet register \
        --netuid 256 \
        --wallet.name cold_draven \
        --wallet.hotkey default \
        --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
        --yes

    if [ $? -eq 0 ]; then
        echo "   ✅ Registration successful!"
        echo "   ⏳ Waiting 30 seconds for registration to propagate..."
        sleep 30
    else
        echo "   ❌ Registration failed!"
        echo "   💡 Check your wallet balance and try again"
        exit 1
    fi
else
    echo "3. ✅ Wallet already registered, skipping..."
fi

# Step 4: Create logs directory
echo "4. Setting up logging..."
mkdir -p logs

# Step 5: Start miner
echo "5. Starting miner..."
echo "   Command: ./start_testnet_miner.sh"
./start_testnet_miner.sh &

MINER_PID=$!
echo "   Miner started with PID: $MINER_PID"

# Step 6: Monitor startup
echo "6. Monitoring miner startup..."
sleep 10

if ps -p $MINER_PID > /dev/null; then
    echo "   ✅ Miner is running successfully!"
else
    echo "   ❌ Miner failed to start"
    exit 1
fi

echo ""
echo "🎯 MINER STATUS SUMMARY"
echo "======================"
echo "✅ Wallet registered on testnet subnet 256"
echo "✅ Miner process running (PID: $MINER_PID)"
echo "⏳ Waiting for metagraph sync and first prediction requests..."
echo ""
echo "📊 MONITOR YOUR MINER:"
echo "======================"
echo "• Live status: python3 check_miner_status.py"
echo "• Prediction monitor: python3 monitor_predictions.py"
echo "• View logs: tail -f miner.log"
echo "• Stop miner: pkill -f miner.py"
echo ""
echo "🎯 EXPECTED BEHAVIOR:"
echo "===================="
echo "• Miner syncs with metagraph (may take 1-2 minutes)"
echo "• Gets assigned UID in subnet 256"
echo "• Starts receiving prediction requests every few minutes"
echo "• Logs predictions to logs/predictions.log"
echo "• Monitor will show real-time request processing"




