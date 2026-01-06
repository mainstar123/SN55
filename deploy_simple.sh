#!/bin/bash
# SIMPLE DEPLOYMENT WITHOUT PM2
# Usage: ./deploy_simple.sh

cd /home/ocean/SN55
source venv/bin/activate

echo "🚀 STARTING FIRST-PLACE DOMINATION MINER (NO PM2)"
echo "================================================="
echo "Time: $(date)"
echo ""

# Check if miner is already running
MINER_PID=$(ps aux | grep "enhanced_domination_miner.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$MINER_PID" ]; then
    echo "⚠️ Miner already running (PID: $MINER_PID)"
    echo "   Stop with: kill $MINER_PID"
    echo "   Or continue to start another instance"
fi

# Start the miner directly
echo "🔥 STARTING MINER..."
LOG_FILE="logs/first_place_domination_$(date +%Y%m%d_%H%M%S).log"

python3 deployment/enhanced_domination_miner.py \
    --neuron.name first_place_domination \
    --wallet.name precog_coldkey \
    --wallet.hotkey miner_hotkey \
    --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
    --axon.port 8091 \
    --netuid 55 \
    --logging.level info \
    --timeout 10 \
    --vpermit_tao_limit 4096 \
    --forward_function safe_domination_forward \
    --neuron.device cuda > "$LOG_FILE" 2>&1 &

MINER_PID=$!
echo "✅ Miner started (PID: $MINER_PID)"
echo "📝 Logs: $LOG_FILE"
echo ""
echo "🔍 MONITORING:"
echo "   Check status: ps aux | grep enhanced_domination_miner"
echo "   View logs: tail -f $LOG_FILE"
echo "   Stop miner: kill $MINER_PID"
echo ""
echo "🎯 Your first-place miner is now running!"
echo "   UID 142 will start earning rewards soon! 🏆"
