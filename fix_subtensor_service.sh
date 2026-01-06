#!/bin/bash
# Fix Subtensor Service Script
# Remove invalid --ws-external argument

echo "🔧 FIXING SUBTENSOR SERVICE ARGUMENTS"
echo "====================================="
echo ""

echo "Removing invalid --ws-external flag..."
sudo sed -i 's/--ws-external //' /etc/systemd/system/subtensor-lite.service

echo ""
echo "Reloading systemd..."
sudo systemctl daemon-reload

echo ""
echo "Restarting subtensor node..."
sudo systemctl restart subtensor-lite

echo ""
echo "Checking status..."
sleep 3
sudo systemctl status subtensor-lite --no-pager | head -5

echo ""
echo "Checking recent logs..."
sudo journalctl -u subtensor-lite -n 5 --no-pager

echo ""
echo "✅ SERVICE FIXED!"
echo "================="

echo "Monitor sync progress with:"
echo "./monitor_subtensor_lite.sh"
