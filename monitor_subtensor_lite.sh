#!/bin/bash
echo "🎯 SUBTENSOR LITE NODE MONITOR"
echo "=============================="
echo "Time: $(date)"
echo ""

# Check service status
echo "📊 SERVICE STATUS:"
echo "=================="
sudo systemctl status subtensor-lite --no-pager | head -3
echo ""

# Check node health
echo "🏥 NODE HEALTH:"
echo "==============="
HEALTH=$(curl -s -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"system_health","params":[],"id":1}' \
  http://localhost:9933 2>/dev/null)

if [ ! -z "$HEALTH" ]; then
    echo "✅ Node responding"
    echo "$HEALTH" | jq . 2>/dev/null || echo "$HEALTH"
else
    echo "⏳ Node starting up..."
fi
echo ""

# Check sync status
echo "🔄 SYNC STATUS:"
echo "==============="
BLOCK=$(curl -s -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"chain_getHeader","params":[],"id":1}' \
  http://localhost:9933 2>/dev/null | jq -r '.result.number' 2>/dev/null)

if [ ! -z "$BLOCK" ] && [ "$BLOCK" != "null" ]; then
    echo "✅ Current block: $BLOCK"
else
    echo "⏳ Syncing blockchain..."
fi
echo ""

# Check peers
echo "👥 PEER COUNT:"
echo "=============="
PEERS=$(curl -s -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"system_peers","params":[],"id":1}' \
  http://localhost:9933 2>/dev/null | jq '.result | length' 2>/dev/null)

if [ ! -z "$PEERS" ]; then
    echo "✅ Connected peers: $PEERS"
else
    echo "⏳ Finding peers..."
fi
echo ""

# Check storage
echo "💾 STORAGE USAGE:"
echo "================="
if [ -d "/var/lib/subtensor-lite" ]; then
    STORAGE=$(du -sh /var/lib/subtensor-lite 2>/dev/null | cut -f1)
    echo "📁 Blockchain data: $STORAGE"
else
    echo "⏳ Storage not yet created"
fi
echo ""

# Recent logs
echo "📋 RECENT LOGS:"
echo "==============="
sudo journalctl -u subtensor-lite -n 5 --no-pager --since "1 minute ago"
echo ""

# Readiness check
echo "🎯 READINESS STATUS:"
echo "===================="
if [ ! -z "$BLOCK" ] && [ "$BLOCK" != "null" ] && [ ! -z "$PEERS" ] && [ "$PEERS" -gt 0 ]; then
    echo "✅ NODE IS READY FOR MINING!"
    echo "============================="
    echo "🌐 WebSocket endpoint: ws://localhost:9944"
    echo "🌐 RPC endpoint: http://localhost:9933"
    echo ""
    echo "🚀 READY TO UPDATE MINER:"
    echo "========================="
    echo "Run these commands:"
    echo "sed -i 's|wss://test\.finney\.opentensor\.ai:443|ws://localhost:9944|g' ~/.config/systemd/user/precog-miner.service"
    echo "systemctl --user daemon-reload"
    echo "./manage_miner_service.sh restart"
    echo ""
    echo "Then check: ./miner_monitor.sh"
else
    echo "⏳ NODE STILL SYNCING..."
    echo "======================="
    echo "Wait for block number and peers to appear."
    echo "This takes 2-4 hours for first sync."
    echo ""
    echo "Monitor progress: ./monitor_subtensor_lite.sh"
fi
