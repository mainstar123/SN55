#!/bin/bash
# Monitor Bittensor network recovery

echo "🌐 BITTENSOR NETWORK RECOVERY MONITOR"
echo "====================================="

while true; do
    echo ""
    echo "⏰ $(date)"
    
    # Test Bittensor endpoints
    echo "Testing endpoints:"
    
    # Test current endpoint
    timeout 5 curl -s --connect-timeout 3 https://archive.substrate.network/ >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ archive.substrate.network: ACCESSIBLE"
        ENDPOINT_STATUS="✅"
    else
        echo "❌ archive.substrate.network: NOT ACCESSIBLE"
        ENDPOINT_STATUS="❌"
    fi
    
    # Test alternative endpoint
    timeout 5 curl -s --connect-timeout 3 https://archive.opentensor.ai/ >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ archive.opentensor.ai: ACCESSIBLE"
        ALT_STATUS="✅"
    else
        echo "❌ archive.opentensor.ai: NOT ACCESSIBLE"
        ALT_STATUS="❌"
    fi
    
    # If any endpoint works, start miner
    if [ "$ENDPOINT_STATUS" = "✅" ] || [ "$ALT_STATUS" = "✅" ]; then
        echo ""
        echo "🚀 NETWORK RECOVERED! Starting miner..."
        
        # Kill any existing miners
        pkill -f "miner.py" 2>/dev/null
        
        # Start miner
        source venv/bin/activate
        cd /home/ocean/SN55
        python3 precog/miners/miner.py --wallet.name precog_coldkey --wallet.hotkey miner_hotkey --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 --netuid 55 --logging.info --axon.port 8091 --neuron.device cuda --forward_function custom_model > logs/miner_recovery_$(date +%Y%m%d_%H%M%S).log 2>&1 &
        
        echo "✅ Miner started! Monitor logs for emission recovery."
        echo "📊 Check Taostats.io UID 142 for emission recovery."
        break
    else
        echo "⏳ Network still down - checking again in 60 seconds..."
        sleep 60
    fi
done
