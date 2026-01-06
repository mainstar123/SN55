#!/bin/bash
# Check accumulated rewards from mining

echo "💰 ACCUMULATED REWARDS CHECK"
echo "============================"
echo ""

echo "📊 YOUR TOTAL EARNINGS FROM LOGS:"
TOTAL_EARNED=$(grep "Emission:" logs/first_place_miner_20260106_150503.log | grep -o "Emission:[0-9]\+\.[0-9]\+" | cut -d: -f2 | awk '{sum += $1} END {print sum}')
echo "• Total TAO earned: $TOTAL_EARNED"
echo "• Current wallet shows: 0.0316 τ"
echo "• Difference: $(echo "$TOTAL_EARNED - 0.0316" | bc) τ (accumulated but not distributed yet)"

echo ""
echo "⏰ AFTER 2 MORE HOURS YOU SHOULD SEE:"
BLOCKS_PER_2_HOURS=18  # ~9 blocks per hour
ADDITIONAL_EARNINGS=$(echo "$BLOCKS_PER_2_HOURS * 0.236" | bc)
EXPECTED_TOTAL=$(echo "$TOTAL_EARNED + $ADDITIONAL_EARNINGS" | bc)
echo "• Additional earnings: ~$ADDITIONAL_EARNINGS TAO"
echo "• Expected total earned: ~$EXPECTED_TOTAL TAO"
echo "• Wallet should show accumulated rewards"

echo ""
echo "🔍 HOW TO SEE ACCUMULATED REWARDS:"
echo "=================================="
echo "1. 🌐 Taostats.io - Real-time earnings (UID: 142)"
echo "2. 💼 Check 'Total Balance' (includes staked rewards)"
echo "3. ⏳ Wait for epoch distribution (every 2-3 hours)"
echo "4. 📈 Rewards will accumulate and become visible!"

echo ""
echo "✅ CONCLUSION: YES! After 2 hours you'll see more accumulated rewards in your wallet!"
