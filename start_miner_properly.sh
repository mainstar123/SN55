#!/bin/bash
# Proper miner startup script with correct environment

cd /home/ocean/nereus/precog

echo "🛑 Stopping any existing miners..."
pkill -f miner.py
sleep 2

echo "🔧 Setting up environment..."
export HOME=/home/ocean
export PYTHONPATH=/home/ocean/nereus/precog
export PATH=/home/ocean/nereus/precog/venv/bin:$PATH

echo "🐍 Activating virtual environment..."
source venv/bin/activate

echo "🚀 Starting miner with GPU support..."
python3 precog/miners/miner.py \
    --neuron.name miner \
    --wallet.name cold_draven \
    --wallet.hotkey default \
    --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
    --axon.port 8092 \
    --netuid 55 \
    --logging.level info \
    --timeout 16 \
    --vpermit_tao_limit 2 \
    --forward_function custom_model \
    --neuron.device cuda > miner.log 2>&1 &

MINER_PID=$!
echo $MINER_PID > miner.pid

echo "✅ Miner started with PID: $MINER_PID"
echo "📊 Logs will be written to: miner.log"
echo ""
echo "🔍 Monitoring startup..."
sleep 5

if ps -p $MINER_PID > /dev/null; then
    echo "✅ Miner is running successfully!"
    echo "📈 Check logs with: tail -f miner.log"
    echo "📊 Monitor with: ./comprehensive_dashboard.sh"
else
    echo "❌ Miner failed to start. Check miner.log for errors."
    cat miner.log
fi
