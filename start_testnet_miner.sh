#!/bin/bash
# Start Precog Testnet Miner with monitoring

cd /home/ocean/nereus/precog
source venv/bin/activate

export HOME=/home/ocean
export BITTENSOR_CONFIG_DIR=/home/ocean/.bittensor

echo "🚀 STARTING PRECOG TESTNET MINER (Subnet 256)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Wallet: cold_draven (default hotkey)"
echo "Network: testnet (subnet 256)"
echo "Forward function: custom_model"
echo ""

# Start miner in background
echo "Starting miner..."
python3 precog/miners/miner.py \
    --neuron.name miner \
    --wallet.name cold_draven \
    --wallet.hotkey default \
    --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
    --axon.port 8092 \
    --netuid 256 \
    --logging.level info \
    --timeout 16 \
    --vpermit_tao_limit 2 \
    --forward_function custom_model &
MINER_PID=$!

echo "Miner started with PID: $MINER_PID"
echo ""

# Monitor for 30 seconds
echo "Monitoring miner startup (30 seconds)..."
sleep 5

echo ""
echo "📊 MINER STATUS:"
echo "━━━━━━━━━━━━━━━━"
if ps -p $MINER_PID > /dev/null; then
    echo "✅ Miner is running (PID: $MINER_PID)"
else
    echo "❌ Miner failed to start"
    exit 1
fi

echo ""
echo "📈 MONITORING DASHBOARD:"
echo "━━━━━━━━━━━━━━━━━━━━━━━"
echo "• Logs: tail -f logs/miner.log (if created)"
echo "• Performance: python scripts/validate_performance.py --hours 1"
echo "• Competitors: python scripts/monitor_competitors.py --netuid 256 --top-n 10"
echo ""
echo "🎯 Press Ctrl+C to stop monitoring (miner will continue running)"
echo ""

# Continuous monitoring
while true; do
    echo "$(date '+%H:%M:%S') - Miner status: $(ps -p $MINER_PID > /dev/null && echo 'RUNNING' || echo 'STOPPED')"
    sleep 10
done
