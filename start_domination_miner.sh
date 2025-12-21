#!/bin/bash
# DOMINATION MINER STARTUP SCRIPT
# Activate domination mode and start miner for #1 positioning

cd /home/ocean/nereus/precog

echo "🏆 STARTING DOMINATION MINER"
echo "============================"
echo "🎯 Target: Surpass UID 31 and become #1"
echo ""

# Stop any existing miners
echo "🛑 Stopping existing miners..."
pkill -f miner.py
sleep 3

# Set up environment
echo "🔧 Setting up domination environment..."
export HOME=/home/ocean
export PYTHONPATH=/home/ocean/nereus/precog
export PATH=/home/ocean/nereus/precog/venv/bin:$PATH
export DOMINATION_MODE=true

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Verify domination model exists
if [ ! -f "models/domination_ensemble.pth" ]; then
    echo "❌ Domination model not found! Run upgrade script first:"
    echo "python3 scripts/direct_domination_upgrade.py"
    exit 1
fi

echo "✅ Domination model verified"
echo ""

# Display domination features
echo "⚡ DOMINATION FEATURES ACTIVATED:"
echo "  • Peak hour optimization (9-11, 13-15 UTC)"
echo "  • 3x prediction frequency during peaks"
echo "  • Market regime detection (Bull/Bear/Volatile/Ranging)"
echo "  • Ensemble predictions (GRU + Transformer)"
echo "  • Real-time UID 31 performance tracking"
echo "  • Adaptive confidence thresholds"
echo ""

echo "🎯 DOMINATION TARGETS:"
echo "  • Hour 12: 0.08+ TAO (Surpass UID 31)"
echo "  • Hour 24: 0.12+ TAO (Enter Top 3)"
echo "  • Hour 48: 0.15+ TAO (Dominate UID 31)"
echo "  • Week 2: 0.19+ TAO (Sustained #1)"
echo ""

# Start miner with domination mode
echo "🚀 Launching domination miner..."
python3 precog/miners/miner.py \
    --neuron.name domination_miner \
    --wallet.name cold_draven \
    --wallet.hotkey default \
    --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443 \
    --axon.port 8092 \
    --netuid 55 \
    --logging.level info \
    --timeout 16 \
    --vpermit_tao_limit 2 \
    --forward_function custom_model \
    --neuron.device cuda > miner_domination.log 2>&1 &

MINER_PID=$!
echo $MINER_PID > miner_domination.pid

echo "✅ Domination miner started with PID: $MINER_PID"
echo "📊 Logs will be written to: miner_domination.log"
echo ""

# Monitor startup
echo "🔍 Monitoring startup..."
sleep 10

if ps -p $MINER_PID > /dev/null; then
    echo "✅ Domination miner is running successfully!"
    echo ""
    echo "📈 MONITORING COMMANDS:"
    echo "  • View logs: tail -f miner_domination.log"
    echo "  • Check performance: ./comprehensive_dashboard.sh"
    echo "  • Stop miner: kill $MINER_PID"
    echo ""
    echo "🎯 WATCH FOR THESE LOG MESSAGES:"
    echo "  • '🏆 ACTIVATING DOMINATION MODE'"
    echo "  • '🎯 Market Regime: [REGIME] | Peak Hour: [True/False]'"
    echo "  • '🎯 Prediction made: [TAO] TAO | Confidence: [SCORE]'"
    echo "  • '📊 Performance Update: [N] predictions | Avg Reward: [TAO] TAO'"
    echo "  • '🎉 TARGET ACHIEVED: Surpassing UID 31 level!'"
    echo ""
    echo "⚡ DOMINATION MODE ACTIVE - BECOMING #1!"
else
    echo "❌ Domination miner failed to start. Check logs:"
    echo "tail -50 miner_domination.log"
fi
