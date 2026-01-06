#!/bin/bash
# Check TAO rewards and wallet balance

echo "💰 TAO REWARDS CHECK"
echo "===================="
echo ""

echo "🔍 METHOD 1: TAOSTATS.IO (Real-time)"
echo "===================================="
echo "1. Open: https://taostats.io/"
echo "2. Search UID: 142"
echo "3. View your balance and earnings"
echo ""

echo "🔍 METHOD 2: LOCAL WALLET CHECK"
echo "==============================="
echo "Adding btcli to PATH..."
export PATH="$HOME/.local/bin:$PATH"

echo "Checking wallet balance..."
btcli wallet overview --wallet.name precog_coldkey 2>/dev/null && echo "✅ Wallet check successful!" || echo "❌ Wallet check failed - use Taostats.io instead"

echo ""
echo "📊 YOUR CURRENT STATUS:"
echo "• Earning: 0.236 TAO/block"
echo "• Estimated daily: ~0-60 USD"
echo "• Network: Precog Subnet 55"
echo "• UID: 142"

echo ""
echo "💡 TIP: Use Taostats.io for real-time earnings tracking!"
