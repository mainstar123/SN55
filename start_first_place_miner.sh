#!/bin/bash
# START FIRST-PLACE DOMINATION MINER
# Uses the proper Bittensor miner framework with enhanced functions

cd /home/ocean/SN55
source venv/bin/activate

echo "🏆 STARTING FIRST-PLACE DOMINATION MINER"
echo "======================================="
echo "Time: $(date)"
echo ""

# Check if miner is already running
MINER_PID=$(ps aux | grep "precog/miners/miner.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$MINER_PID" ]; then
    echo "⚠️ Miner already running (PID: $MINER_PID)"
    echo "   Stop with: kill $MINER_PID"
    echo "   Or continue to start another instance"
fi

# Enable domination mode for advanced features
export DOMINATION_MODE=true

# Start the miner with Bittensor framework
echo "🚀 STARTING MINER WITH BITTENSOR FRAMEWORK (DOMINATION MODE)..."
LOG_FILE="logs/first_place_miner_$(date +%Y%m%d_%H%M%S).log"

python3 precog/miners/miner.py \
    --wallet.name precog_coldkey \
    --wallet.hotkey miner_hotkey \
    --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
    --netuid 55 \
    --logging.info \
    --axon.port 8091 \
    --neuron.device cuda \
    --forward_function custom_model > "$LOG_FILE" 2>&1 &

MINER_PID=$!
echo "✅ First-place miner started (PID: $MINER_PID)"
echo "📝 Logs: $LOG_FILE"
echo ""
echo "🔍 MONITORING:"
echo "   Check status: ps aux | grep precog/miners/miner.py"
echo "   View logs: tail -f $LOG_FILE"
echo "   Stop miner: kill $MINER_PID"
echo ""
echo "📊 Check taostats.io/subnets/subnet-55/ for UID 142 performance!"
echo ""
echo "🎯 Your enhanced domination miner is now serving predictions!"
echo "   Target: #1 position in Precog Subnet 55! 🏆"
