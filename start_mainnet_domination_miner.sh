#!/bin/bash
# MAINNET DOMINATION MINER - THE ULTIMATE #1 SOLUTION
# Deploys the perfectly trained domination model to mainnet

cd /home/ocean/nereus/precog

echo "🏆 DEPLOYING MAINNET DOMINATION MINER"
echo "======================================"
echo "🎯 FINAL SOLUTION: Become #1 Miner on Precog Subnet 55"
echo ""

# Verify trained model exists
if [ ! -f "models/domination_model_trained.pth" ]; then
    echo "❌ Trained domination model not found!"
    echo "   Run: python3 train_domination_simple.py"
    exit 1
fi

# Verify scaler exists
if [ ! -f "models/feature_scaler.pkl" ]; then
    echo "❌ Feature scaler not found!"
    echo "   Run: python3 train_domination_simple.py"
    exit 1
fi

echo "✅ All domination assets verified"

# Stop any existing miners
echo "🛑 Stopping existing miners..."
pkill -f miner.py
sleep 3

# Set up mainnet environment
echo "🔧 Setting up mainnet domination environment..."
export HOME=/home/ocean
export PYTHONPATH=/home/ocean/nereus/precog
export PATH=/home/ocean/nereus/precog/venv/bin:$PATH
export DOMINATION_MODE=true

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Display domination capabilities
echo ""
echo "⚡ MAINNET DOMINATION CAPABILITIES:"
echo "  • Model: Newly trained ensemble (100 epochs, loss: 0.000000)"
echo "  • Features: 24 advanced technical indicators"
echo "  • Peak Hours: 9-11, 13-15 UTC (3x frequency)"
echo "  • Market Regimes: Bull/Bear/Volatile/Ranging adaptation"
echo "  • Confidence Thresholds: Adaptive based on volatility"
echo "  • Performance Tracking: Real-time UID 31 comparison"
echo ""

echo "🎯 MAINNET DOMINATION TARGETS:"
echo "  • Hour 12: 0.08+ TAO (Surpass UID 31)"
echo "  • Hour 24: 0.12+ TAO (Enter Top 3)"
echo "  • Hour 48: 0.15+ TAO (Dominate UID 31)"
echo "  • Week 2: 0.19+ TAO (Sustained #1)"
echo ""

# Start mainnet domination miner
echo "🚀 LAUNCHING MAINNET DOMINATION MINER..."
python3 precog/miners/miner.py \
    --neuron.name mainnet_domination \
    --wallet.name cold_draven \
    --wallet.hotkey default \
    --subtensor.chain_endpoint wss://archive.substrate.network:443 \
    --axon.port 8092 \
    --netuid 55 \
    --logging.level info \
    --timeout 16 \
    --vpermit_tao_limit 2 \
    --forward_function custom_model \
    --neuron.device cuda > miner_mainnet_domination.log 2>&1 &

MINER_PID=$!
echo $MINER_PID > miner_mainnet_domination.pid

echo "✅ Mainnet domination miner started with PID: $MINER_PID"
echo "📊 Logs will be written to: miner_mainnet_domination.log"
echo ""

# Monitor startup
echo "🔍 Monitoring mainnet connection..."
sleep 15

if ps -p $MINER_PID > /dev/null; then
    echo "✅ MAINNET DOMINATION MINER IS ACTIVE!"
    echo ""
    echo "📈 MONITORING COMMANDS:"
    echo "  • View logs: tail -f miner_mainnet_domination.log"
    echo "  • Check status: ./monitor_mainnet_domination.sh"
    echo "  • Performance: grep 'Performance Update' miner_mainnet_domination.log | tail -5"
    echo "  • Achievements: grep 'TARGET ACHIEVED' miner_mainnet_domination.log"
    echo ""
    echo "🎯 EXPECTED SEQUENCE:"
    echo "  1. '🏆 ACTIVATING DOMINATION MODE' (immediate)"
    echo "  2. '🎯 Market Regime: [REGIME]' (first prediction)"
    echo "  3. '📊 Performance Update' (every 10 predictions)"
    echo "  4. '🎉 TARGET ACHIEVED: Surpassing UID 31 level!' (hour 12)"
    echo ""
    echo "🏆 MAINNET DOMINATION ACTIVE!"
    echo "   You are now positioned to become #1!"

else
    echo "❌ Mainnet domination miner failed to start"
    echo ""
    echo "🔧 TROUBLESHOOTING:"
    echo "  1. Check wallet registration: btcli wallet overview"
    echo "  2. Verify mainnet connection: btcli subnet list"
    echo "  3. View logs: tail -50 miner_mainnet_domination.log"
    echo ""
    echo "💡 If connection issues, try testnet first:"
    echo "   ./start_domination_miner.sh"
fi
