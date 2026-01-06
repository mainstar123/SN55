#!/bin/bash
# Watch for validator requests in real-time

echo "👀 WATCHING FOR VALIDATOR REQUESTS..."
echo "====================================="
echo "Monitoring logs for request activity..."
echo "Press Ctrl+C to stop"
echo ""

# Watch for request patterns in real-time
tail -f logs/first_place_miner_20260106_150503.log | grep -E --line-buffered "(synapse|challenge|request|prediction|processing|response|emission|trust|incentive)" || echo "No requests detected yet..."
