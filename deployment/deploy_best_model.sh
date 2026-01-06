#!/bin/bash
# DEPLOY BEST MODEL MINER
# This script deploys the domination model miner (best performance)
# Usage: ./deployment/deploy_best_model.sh

cd /home/ocean/SN55
source venv/bin/activate

echo "🚀 DEPLOYING PRECOG BEST MODEL MINER"
echo "===================================="
echo "Time: $(date)"
echo ""

# Pre-deployment checks
echo "1. PRE-DEPLOYMENT CHECKS:"

# Check model files
if [ ! -f "models/domination_model_trained.pth" ]; then
    echo "❌ Trained model not found!"
    echo "   Run: python3 train_domination_simple.py"
    exit 1
fi

if [ ! -f "models/feature_scaler.pkl" ]; then
    echo "❌ Feature scaler not found!"
    echo "   Run: python3 train_domination_simple.py"
    exit 1
fi

echo "✅ Model files verified"

# Check environment file
if [ ! -f ".env.miner" ]; then
    echo "❌ .env.miner not found!"
    echo "   Copy from .env.miner.example and edit"
    exit 1
fi

# Validate environment variables
source .env.miner
if [ -z "$COLDKEY" ] || [ -z "$MINER_HOTKEY" ]; then
    echo "❌ Wallet configuration missing in .env.miner"
    echo "   Edit .env.miner with your wallet names"
    exit 1
fi

echo "✅ Environment configuration verified"

# Check wallet registration
echo "2. WALLET VERIFICATION:"
btcli wallet overview --wallet.name $COLDKEY >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Coldkey wallet not found: $COLDKEY"
    echo "   Create with: btcli wallet new_coldkey --wallet.name $COLDKEY"
    exit 1
fi

# Check hotkey registration
btcli wallet overview --wallet.name $COLDKEY | grep -q $MINER_HOTKEY
if [ $? -ne 0 ]; then
    echo "❌ Hotkey not found: $MINER_HOTKEY"
    echo "   Create with: btcli wallet new_hotkey --wallet.name $COLDKEY --wallet.hotkey $MINER_HOTKEY"
    exit 1
fi

echo "✅ Wallets verified"

# Check TAO balance
TAO_BALANCE=$(btcli wallet overview --wallet.name $COLDKEY | grep "τ" | head -1 | awk '{print $2}' | sed 's/τ//')
if (( $(echo "$TAO_BALANCE < 0.001" | bc -l) )); then
    echo "❌ Insufficient TAO balance: $TAO_BALANCE τ"
    echo "   Need at least 0.001 τ for operations"
    echo "   Get TAO from friends or faucet"
    exit 1
fi

echo "✅ TAO balance sufficient: $TAO_BALANCE τ"

# Check subnet registration
echo "3. SUBNET REGISTRATION CHECK:"
REGISTERED=$(btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 | grep -c "$COLDKEY\|$MINER_HOTKEY" 2>/dev/null || echo "0")
if [ "$REGISTERED" -eq "0" ]; then
    echo "❌ Not registered on subnet 55"
    echo "   Register with: btcli subnet register --netuid 55 --wallet.name $COLDKEY --wallet.hotkey $MINER_HOTKEY --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443"
    exit 1
fi

echo "✅ Subnet 55 registration verified"

# Stop any existing miners
echo "4. CLEANUP:"
echo "Stopping existing miners..."
pm2 delete all 2>/dev/null || true
pkill -f miner.py 2>/dev/null || true
sleep 3
echo "✅ Cleanup complete"

# Set up environment for best model
echo "5. MODEL CONFIGURATION:"
export PYTHONPATH=/home/ocean/SN55:$PYTHONPATH
export HOME=/home/ocean
export DOMINATION_MODE=true
export ONLINE_RETRAINING=true

# Create logs directory
mkdir -p logs

echo "✅ Environment configured"

# Deploy the miner
echo "6. DEPLOYING BEST MODEL MINER:"

# Start with PM2
pm2 start --name precog_best_miner python3 -- precog/miners/miner.py \
    --neuron.name best_miner \
    --wallet.name $COLDKEY \
    --wallet.hotkey $MINER_HOTKEY \
    --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
    --axon.port $MINER_PORT \
    --netuid 55 \
    --logging.level info \
    --timeout $TIMEOUT \
    --vpermit_tao_limit $VPERMIT_TAO_LIMIT \
    --forward_function custom_model \
    --neuron.device cuda > logs/best_miner_$(date +%Y%m%d_%H%M%S).log 2>&1 &

MINER_PID=$!
echo $MINER_PID > best_miner.pid

echo "✅ Miner deployment initiated (PID: $MINER_PID)"

# Wait for startup
echo "7. STARTUP VERIFICATION:"
sleep 15

# Check if miner is running
if ps -p $MINER_PID > /dev/null; then
    echo "✅ Miner process is running"

    # Test axon connectivity
    sleep 10
    if curl -s --max-time 5 http://localhost:$MINER_PORT/ping > /dev/null 2>&1; then
        echo "✅ Axon responding on port $MINER_PORT"
    else
        echo "⚠️ Axon not responding yet - may take longer to start"
    fi

    # Save PM2 configuration
    pm2 save

    echo ""
    echo "🎉 BEST MODEL MINER DEPLOYMENT SUCCESSFUL!"
    echo "=========================================="
    echo ""
    echo "📊 MONITORING COMMANDS:"
    echo "   • Live logs: pm2 logs precog_best_miner --follow"
    echo "   • Status: pm2 monit"
    echo "   • Performance: ./deployment/monitor_precog.sh"
    echo "   • Balance: watch -n 300 'btcli wallet overview --wallet.name $COLDKEY'"
    echo ""
    echo "🔧 MANAGEMENT COMMANDS:"
    echo "   • Restart: pm2 restart precog_best_miner"
    echo "   • Stop: pm2 stop precog_best_miner"
    echo "   • Delete: pm2 delete precog_best_miner"
    echo ""
    echo "📈 EXPECTED BEHAVIOR:"
    echo "   • First 5-10 min: Model loading and connection"
    echo "   • 10-30 min: First predictions and rewards"
    echo "   • 1-2 hours: Performance optimization begins"
    echo "   • 24 hours: Stable performance with auto-improvements"
    echo ""
    echo "🔄 AUTO-IMPROVEMENT FEATURES ACTIVE:"
    echo "   • Online retraining every 5 minutes"
    echo "   • Peak hour optimization (UTC 9-11, 13-15)"
    echo "   • Market regime adaptation"
    echo "   • Ensemble model updates"
    echo ""
    echo "📋 NEXT STEPS:"
    echo "   1. Monitor logs for first predictions"
    echo "   2. Check balance after 1 hour"
    echo "   3. Run automated retraining daily: ./deployment/automated_retraining.sh"
    echo "   4. Weekly: Review performance and consider hyperparameter tuning"

else
    echo "❌ Miner failed to start"
    echo ""
    echo "🔧 TROUBLESHOOTING:"
    echo "   • Check logs: tail -50 logs/best_miner_*.log"
    echo "   • Verify wallet: btcli wallet overview --wallet.name $COLDKEY"
    echo "   • Check registration: btcli subnet metagraph --netuid 55"
    echo "   • Restart: ./deployment/deploy_best_model.sh"
fi

echo ""
echo "=========================================="
