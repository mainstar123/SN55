#!/bin/bash
# PRECOG DEPLOYMENT SETUP VERIFICATION
# Run: ./deployment/verify_setup.sh

echo "=========================================="
echo "🔍 PRECOG SUBNET 55 - SETUP VERIFICATION"
echo "=========================================="
echo ""

# Check Python environment
echo "1. PYTHON ENVIRONMENT:"
python3 --version
if [ $? -eq 0 ]; then
    echo "   ✅ Python 3 available"
else
    echo "   ❌ Python 3 not found"
fi

# Check virtual environment
if [ -d "venv" ]; then
    echo "   ✅ Virtual environment exists"
    source venv/bin/activate
    if [ $? -eq 0 ]; then
        echo "   ✅ Virtual environment activated"
    else
        echo "   ❌ Cannot activate virtual environment"
    fi
else
    echo "   ❌ Virtual environment not found"
fi
echo ""

# Check Bittensor CLI
echo "2. BITTENSOR CLI:"
btcli --version >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Bittensor CLI available"
else
    echo "   ❌ Bittensor CLI not found - run: pip install bittensor"
fi
echo ""

# Check wallet
echo "3. WALLET STATUS:"
if [ -d "~/.bittensor/wallets" ]; then
    echo "   ✅ Bittensor wallets directory exists"
    ls ~/.bittensor/wallets/ | head -5
else
    echo "   ❌ No wallets found - create wallet first"
fi
echo ""

# Check environment file
echo "4. ENVIRONMENT CONFIGURATION:"
if [ -f ".env.miner" ]; then
    echo "   ✅ .env.miner file exists"
    echo "   Wallet name: $(grep COLDKEY .env.miner | cut -d'=' -f2)"
    echo "   Hotkey name: $(grep MINER_HOTKEY .env.miner | cut -d'=' -f2)"
    echo "   Network: $(grep NETWORK .env.miner | cut -d'=' -f2)"
else
    echo "   ❌ .env.miner not found - copy from .env.miner.example"
fi
echo ""

# Check model files
echo "5. MODEL FILES:"
if [ -f "models/domination_model_trained.pth" ]; then
    echo "   ✅ Trained model exists"
else
    echo "   ❌ Trained model not found - run training first"
fi

if [ -f "models/feature_scaler.pkl" ]; then
    echo "   ✅ Feature scaler exists"
else
    echo "   ❌ Feature scaler not found"
fi
echo ""

# Check PM2
echo "6. PM2 PROCESS MANAGER:"
pm2 --version >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ PM2 available"
    pm2 list | grep -E "(miner|precog)" || echo "   ℹ️  No precog miners running"
else
    echo "   ❌ PM2 not found - install with: npm install -g pm2"
fi
echo ""

# Network connectivity
echo "7. NETWORK CONNECTIVITY:"
ping -c 1 archive.substrate.network >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Mainnet connectivity OK"
else
    echo "   ❌ Cannot reach mainnet - check internet"
fi

curl -s --max-time 5 http://localhost:8092/ping >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Local miner port accessible"
else
    echo "   ℹ️  Local miner not running (port 8092)"
fi
echo ""

# TAO balance check
echo "8. TAO BALANCE CHECK:"
if [ -f ".env.miner" ] && grep -q "COLDKEY" .env.miner; then
    WALLET_NAME=$(grep COLDKEY .env.miner | cut -d'=' -f2)
    echo "   Checking balance for wallet: $WALLET_NAME"
    btcli wallet overview --wallet.name "$WALLET_NAME" 2>/dev/null | grep -E "(Balance|τ)" || echo "   ❌ Cannot check balance - wallet may not exist"
else
    echo "   ❌ Cannot determine wallet name"
fi
echo ""

echo "=========================================="
echo "🎯 NEXT STEPS:"
echo "=========================================="

# Determine next steps based on verification
if [ ! -d "~/.bittensor/wallets" ]; then
    echo "1. Create wallet: btcli wallet new_coldkey --wallet.name precog_wallet"
    echo "2. Create hotkey: btcli wallet new_hotkey --wallet.name precog_wallet --wallet.hotkey miner_key"
    echo "3. Get TAO from friends and register"
elif ! grep -q "0\..*τ" <(btcli wallet overview --wallet.name "$WALLET_NAME" 2>/dev/null); then
    echo "1. Get TAO from friends to your wallet address"
    echo "2. Register on subnet 55"
    echo "3. Deploy miner"
elif ! pm2 list | grep -q "online.*miner"; then
    echo "1. Deploy miner: make miner_custom ENV_FILE=.env.miner"
    echo "2. Monitor performance: pm2 logs miner --follow"
else
    echo "✅ Setup appears complete!"
    echo "1. Monitor performance: ./deployment/quick_commands.sh"
    echo "2. Check earnings regularly"
    echo "3. Consider model improvements after 24-48 hours"
fi

echo ""
echo "📖 Full guide: deployment/COMPLETE_PRECOG_DEPLOYMENT_GUIDE.md"
echo "=========================================="
