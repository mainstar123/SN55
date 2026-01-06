#!/bin/bash
# Start miner in screen session as backup option

SESSION_NAME="precog-miner"

echo "📺 Starting miner in screen session: $SESSION_NAME"
echo "=================================================="

# Check if screen session already exists
if screen -list | grep -q "$SESSION_NAME"; then
    echo "⚠️  Screen session '$SESSION_NAME' already exists"
    echo "   Attach to it with: screen -r $SESSION_NAME"
    echo "   Or kill and restart with: screen -X -S $SESSION_NAME quit && $0"
    exit 1
fi

# Start miner in detached screen session
cd /home/ocean/SN55
source venv/bin/activate

screen -dmS "$SESSION_NAME" python3 precog/miners/miner.py \
    --wallet.name precog_coldkey \
    --wallet.hotkey miner_hotkey \
    --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
    --netuid 55 \
    --logging.info \
    --axon.port 8091 \
    --neuron.device cuda \
    --forward_function custom_model

echo "✅ Miner started in screen session '$SESSION_NAME'"
echo ""
echo "📋 Screen Management Commands:"
echo "• Attach to session:  screen -r $SESSION_NAME"
echo "• Detach from session: Ctrl+A, D"
echo "• Kill session:       screen -X -S $SESSION_NAME quit"
echo "• List sessions:      screen -list"
echo ""
echo "📊 Monitor logs in screen or use: tail -f logs/*.log"
