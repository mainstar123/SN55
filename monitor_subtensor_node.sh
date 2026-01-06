#!/bin/bash
echo "🎯 SUBTENSOR NODE MONITOR"
echo "========================"

# Check if container is running
echo "Container Status:"
sudo docker ps | grep subtensor-lite || echo "❌ Container not running"

echo ""
echo "Node Health:"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"system_health","params":[],"id":1}' \
  http://localhost:9933 2>/dev/null | jq . 2>/dev/null || echo "Node not ready yet"

echo ""
echo "Current Block:"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"chain_getHeader","params":[],"id":1}' \
  http://localhost:9933 2>/dev/null | jq -r '.result.number' 2>/dev/null || echo "Block data not available"

echo ""
echo "Peer Count:"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"system_peers","params":[],"id":1}' \
  http://localhost:9933 2>/dev/null | jq '.result | length' 2>/dev/null || echo "Peer data not available"

echo ""
echo "Disk Usage:"
sudo docker system df 2>/dev/null | grep subtensor || echo "Volume data not available"
