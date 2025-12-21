#!/bin/bash

# 🎯 PRECOG DOMINATION SYSTEM - DEPLOYMENT SCRIPT
# Based on comprehensive analysis vs Miner #52
# Deployment Readiness: B+ (Very Good) - 70/105 domination score

echo "🎯 PRECOG DOMINATION SYSTEM DEPLOYMENT"
echo "======================================="
echo "Analysis Complete: B+ Competitive Rating vs Miner #52"
echo "Expected Performance: 0.55-0.65 TAO/day initially"
echo "Position Target: Top 10 (competitive)"
echo ""

# Configuration
DEPLOYMENT_DIR="/home/ocean/nereus/precog"
VENV_PATH="$DEPLOYMENT_DIR/venv"
MINER_PID_FILE="$DEPLOYMENT_DIR/miner.pid"

echo "📋 PRE-DEPLOYMENT CHECKS"
echo "------------------------"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "   Run: python3 -m venv $VENV_PATH"
    exit 1
fi
echo "✅ Virtual environment found"

# Check required files
required_files=(
    "start_domination_miner.py"
    "advanced_ensemble_model.py"
    "advanced_attention_mechanisms.py"
    "market_regime_detector.py"
    "peak_hour_optimizer.py"
    "performance_tracking_system.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$DEPLOYMENT_DIR/$file" ]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done
echo "✅ All required files present"

# Check if miner is already running
if [ -f "$MINER_PID_FILE" ]; then
    OLD_PID=$(cat "$MINER_PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "⚠️  Existing miner process found (PID: $OLD_PID)"
        echo "   Stopping existing miner..."
        kill "$OLD_PID"
        sleep 5
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "   Force killing..."
            kill -9 "$OLD_PID"
        fi
    fi
    rm -f "$MINER_PID_FILE"
    echo "✅ Existing miner stopped"
fi

echo ""
echo "🚀 DEPLOYMENT SEQUENCE"
echo "----------------------"

# Activate virtual environment
echo "1. Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Navigate to deployment directory
cd "$DEPLOYMENT_DIR"

# Set environment variables for domination mode
export DOMINATION_MODE=true
export CUDA_VISIBLE_DEVICES=0  # Use GPU if available
export PYTHONPATH="$DEPLOYMENT_DIR:$PYTHONPATH"

echo "2. Environment configured:"
echo "   - DOMINATION_MODE: $DOMINATION_MODE"
echo "   - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   - Working directory: $(pwd)"

# Start miner in background with monitoring
echo "3. Starting domination miner..."
echo "   Command: python3 start_domination_miner.py --deploy --monitor"

python3 start_domination_miner.py --deploy --monitor > domination_deploy.log 2>&1 &
MINER_PID=$!

# Save PID for monitoring
echo $MINER_PID > "$MINER_PID_FILE"

echo "✅ Miner started with PID: $MINER_PID"
echo "   Logs: domination_deploy.log"
echo "   PID file: $MINER_PID_FILE"

# Wait a moment for startup
sleep 3

# Verify miner is running
if kill -0 "$MINER_PID" 2>/dev/null; then
    echo "✅ Miner process verified running"

    # Get initial status
    echo ""
    echo "📊 INITIAL STATUS CHECK"
    echo "----------------------"
    ps aux | grep "python3 start_domination_miner.py" | grep -v grep

    echo ""
    echo "🎯 DEPLOYMENT SUCCESSFUL!"
    echo "========================="
    echo "🎯 System Status: ACTIVE"
    echo "🏆 Target Performance: 0.55-0.65 TAO/day"
    echo "🎯 Competitive Position: Top 10 contender"
    echo "📊 Domination Score: B+ (70/105)"
    echo ""
    echo "📋 MONITORING INSTRUCTIONS:"
    echo "• Check logs: tail -f domination_deploy.log"
    echo "• Monitor earnings: watch subnet rankings"
    echo "• Performance tracking: check performance logs"
    echo "• Emergency stop: kill $MINER_PID or ./stop_miner.sh"
    echo ""
    echo "🚨 FIRST 24 HOURS - INTENSIVE MONITORING:"
    echo "• Monitor TAO earnings hourly"
    echo "• Check system stability"
    echo "• Validate prediction accuracy"
    echo "• Compare vs Miner #52 performance"
    echo ""
    echo "🎉 DOMINATION SYSTEM IS NOW LIVE!"
    echo "   Expected to surpass Miner #52 within weeks 🚀"

else
    echo "❌ Miner failed to start properly"
    echo "   Check logs: cat domination_deploy.log"
    exit 1
fi

# Deactivate virtual environment
deactivate

exit 0




