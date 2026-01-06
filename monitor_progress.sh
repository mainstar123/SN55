#!/bin/bash
echo "🏆 PRECOG MINER PROGRESS DASHBOARD"
echo "==================================="
echo "Time: $(date)"
echo ""

# Miner Status
echo "🤖 MINER STATUS:"
ps aux | grep "miner.py" | grep -v grep | awk '{print "   • PID:", $2, "| CPU:", $3"%", "| Memory:", $4"%"}' || echo "   ❌ Miner not running!"
echo ""

# Recent Logs
echo "📝 RECENT ACTIVITY:"
tail -5 logs/first_place_miner_*.log 2>/dev/null | head -5
echo ""

# Wallet Balance
echo "💰 WALLET BALANCE:"
btcli wallet overview --wallet.name precog_coldkey 2>/dev/null | grep -E "(τ|free)" | head -1
echo ""

# Network Position (if available)
echo "🌐 NETWORK POSITION:"
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 2>/dev/null | grep -A1 -B1 "142" || echo "   Position data loading..."
echo ""

echo "📊 Check https://taostats.io/subnets/subnet-55/ for full leaderboard!"
