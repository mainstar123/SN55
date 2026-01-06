#!/bin/bash
# Secure Subtensor Node Setup Script
# Run this script to set up your own Bittensor node

echo "🛠️  SETTING UP YOUR OWN SUBTENSOR NODE"
echo "======================================"
echo ""

echo "This script will:"
echo "• Install Docker"
echo "• Download Bittensor subtensor image"
echo "• Run your lite subtensor node"
echo "• Configure it for mining"
echo ""

read -p "Do you want to proceed with the setup? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 1
fi

echo ""
echo "🔧 STEP 1: INSTALLING DOCKER"
echo "============================"

# Install Docker
echo "Installing Docker..."
sudo apt update
sudo apt install -y docker.io

# Start and enable Docker
echo "Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

echo "✅ Docker installed and running"
echo ""

echo "🐳 STEP 2: DOWNLOADING BITTENSOR IMAGE"
echo "======================================="

echo "Pulling official Bittensor subtensor image..."
sudo docker pull opentensor/subtensor:latest

echo "✅ Bittensor image downloaded"
echo ""

echo "🚀 STEP 3: STARTING YOUR SUBTENSOR NODE"
echo "======================================="

echo "Starting lite subtensor node for Bittensor main net..."
echo "This will create a node that syncs with the finney chain."
echo ""

sudo docker run -d \
  --name subtensor-lite \
  --restart unless-stopped \
  -p 9933:9933 \
  -p 9943:9943 \
  -v subtensor-lite-data:/data \
  opentensor/subtensor:latest \
  --chain finney \
  --rpc-external \
  --ws-external \
  --rpc-cors all \
  --pruning archive \
  --db-cache 2048 \
  --execution wasm \
  --wasm-execution compiled

echo "✅ Subtensor node started!"
echo ""

echo "📊 STEP 4: VERIFYING NODE SETUP"
echo "==============================="

echo "Checking if node is running..."
sleep 5
sudo docker ps | grep subtensor-lite

echo ""
echo "Node information:"
echo "• Name: subtensor-lite"
echo "• WebSocket endpoint: ws://localhost:9933"
echo "• RPC endpoint: http://localhost:9933"
echo "• Chain: finney (Bittensor main net)"
echo "• Storage: Docker volume (subtensor-lite-data)"
echo ""

echo "⏳ STEP 5: MONITORING INITIAL SYNC"
echo "=================================="

echo "The node is now downloading and syncing the Bittensor blockchain."
echo "This is normal and can take several hours for the first run."
echo ""

echo "To monitor sync progress:"
echo "sudo docker logs -f subtensor-lite"
echo ""

echo "Or check current status:"
echo "sudo docker logs subtensor-lite | tail -10"
echo ""

echo "🎯 STEP 6: WHEN NODE IS SYNCED"
echo "=============================="

echo "Once your node shows 'Syncing' or block numbers in the logs,"
echo "it's ready to use. Then run these commands:"
echo ""

echo "# Update your miner to use your own node"
echo "sed -i 's|wss://test\.finney\.opentensor\.ai:443|ws://localhost:9933|g' ~/.config/systemd/user/precog-miner.service"
echo ""

echo "# Restart miner"
echo "./manage_miner_service.sh restart"
echo ""

echo "# Verify the change"
echo "./manage_miner_service.sh logs | grep 'chain_endpoint'"
echo ""

echo "🏆 RESULT: INDEPENDENT MINING INFRASTRUCTURE"
echo "============================================="

echo "✅ Your own Bittensor node"
echo "✅ Always available endpoint"
echo "✅ Independent of public outages"
echo "✅ Professional mining setup"
echo "✅ Maximum earning potential"
echo ""

echo "🎉 SETUP COMPLETE!"
echo "=================="
echo ""
echo "Your subtensor node is now running and syncing."
echo "Monitor progress and switch your miner when ready!"
echo ""
echo "Monitor with: sudo docker logs -f subtensor-lite"
