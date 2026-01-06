#!/bin/bash
# QUICK COMMANDS REFERENCE FOR PRECOG DEPLOYMENT
# Run: ./deployment/quick_commands.sh

echo "=========================================="
echo "🚀 PRECOG SUBNET 55 - QUICK COMMANDS"
echo "=========================================="
echo ""

echo "1. WALLET MANAGEMENT:"
echo "   Create coldkey:     btcli wallet new_coldkey --wallet.name precog_wallet"
echo "   Create hotkey:      btcli wallet new_hotkey --wallet.name precog_wallet --wallet.hotkey miner_key"
echo "   Check balance:      btcli wallet overview --wallet.name precog_wallet"
echo "   Get address:        btcli wallet overview --wallet-name precog_wallet"
echo ""

echo "2. REGISTRATION:"
echo "   Register subnet:    btcli subnet register --netuid 55 --wallet.name precog_wallet --wallet.hotkey miner_key --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443"
echo "   Verify register:    btcli wallet overview --wallet.name precog_wallet"
echo "   Check position:     btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 | grep precog"
echo ""

echo "3. DEPLOYMENT:"
echo "   Deploy miner:       make miner_custom ENV_FILE=.env.miner"
echo "   Check status:       pm2 list"
echo "   View logs:          pm2 logs miner"
echo "   Restart miner:      pm2 restart miner"
echo "   Stop miner:         pm2 stop miner"
echo ""

echo "4. MONITORING:"
echo "   Live logs:          pm2 logs miner --follow"
echo "   Performance:        pm2 logs miner | grep 'Prediction made' | wc -l"
echo "   Balance check:      watch -n 60 'btcli wallet overview --wallet.name precog_wallet'"
echo "   Network position:   btcli subnet metagraph --netuid 55 | head -20"
echo ""

echo "5. TROUBLESHOOTING:"
echo "   Check processes:    ps aux | grep miner"
echo "   Port check:         netstat -tlnp | grep 8092"
echo "   Restart network:    pm2 restart miner"
echo "   Full logs:          pm2 logs miner > debug.log"
echo ""

echo "6. MODEL IMPROVEMENT:"
echo "   Retrain model:      python3 automated_retraining.sh"
echo "   Performance check:  pm2 logs miner | grep MAPE | tail -5"
echo "   Backup model:       cp models/*.pth models/backup/"
echo ""

echo "=========================================="
echo "📖 FULL GUIDE: deployment/COMPLETE_PRECOG_DEPLOYMENT_GUIDE.md"
echo "=========================================="
