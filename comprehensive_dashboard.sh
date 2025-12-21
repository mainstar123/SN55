#!/bin/bash
# Comprehensive Precog Mining Dashboard

cd /home/ocean/nereus/precog
source venv/bin/activate

export HOME=/home/ocean
export BITTENSOR_CONFIG_DIR=/home/ocean/.bittensor

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                        🚀 PRECOG MINING DASHBOARD                        ║"
    echo "║                          Subnet 55 - Testnet                             ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║ Time: $(date '+%Y-%m-%d %H:%M:%S')                                       ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    echo "╔═══════════════════════════════ WALLET STATUS ═══════════════════════════════╗"
    BALANCE=$(btcli wallet balance --wallet.name cold_draven --subtensor.network test 2>/dev/null | grep -o "[0-9]*\.[0-9]* τ" | head -1 || echo "0.0000 τ")
    echo "║ Wallet: cold_draven (default hotkey)                                       ║"
    echo "║ Balance: $BALANCE                                                         ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    echo "╔═══════════════════════════════ MINER STATUS ════════════════════════════════╗"
    if ps aux | grep -v grep | grep "miner.py" > /dev/null; then
        MINER_PID=$(ps aux | grep "miner.py" | grep -v grep | awk '{print $2}')
        CPU_USAGE=$(ps aux | grep "miner.py" | grep -v grep | awk '{print $3}')
        MEM_USAGE=$(ps aux | grep "miner.py" | grep -v grep | awk '{print $4}')
        echo "║ Status: 🟢 RUNNING (PID: $MINER_PID)                                      ║"
        echo "║ CPU: ${CPU_USAGE}% | Memory: ${MEM_USAGE}MB                                   ║"
    else
        echo "║ Status: 🔴 NOT RUNNING                                                   ║"
        echo "║ 💡 Run: ./start_miner.sh                                                ║"
    fi
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    echo "╔═══════════════════════════════ MINING METRICS ══════════════════════════════╗"
    OVERVIEW=$(btcli wallet overview --wallet.name cold_draven --subtensor.network test 2>/dev/null)
    INCENTIVE=$(echo "$OVERVIEW" | grep -o 'Incentive: [0-9.]*' | head -1 | cut -d' ' -f2 || echo "0.0000")
    EMISSIONS=$(echo "$OVERVIEW" | grep -o 'Emission: [0-9.]*' | head -1 | cut -d' ' -f2 || echo "0.000000")
    TRUST=$(echo "$OVERVIEW" | grep -o 'Trust: [0-9.]*' | head -1 | cut -d' ' -f2 || echo "0.0000")

    echo "║ Incentive Score: $INCENTIVE (Target: >0.001)                              ║"
    echo "║ Daily Emissions: $EMISSIONS τ                                             ║"
    echo "║ Trust Score: $TRUST (Reliability)                                        ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    echo "╔═══════════════════════════════ YOUR RANKING ════════════════════════════════╗"
    # Get your position in metagraph
    YOUR_HOTKEY=$(btcli wallet list 2>/dev/null | grep "Hotkey default" -A1 | tail -1 | awk '{print $NF}' || echo "Unknown")
    METAGRAPH_INFO=$(btcli subnet metagraph --netuid 55 --subtensor.network test 2>/dev/null | grep -n "$YOUR_HOTKEY" || echo "Not found")

    if [ "$METAGRAPH_INFO" != "Not found" ]; then
        LINE_NUM=$(echo $METAGRAPH_INFO | cut -d: -f1)
        TOTAL_MINERS=$(btcli subnet metagraph --netuid 55 --subtensor.network test 2>/dev/null | wc -l || echo "40")
        echo "║ Your Rank: #$LINE_NUM out of ~$TOTAL_MINERS miners                         ║"
        echo "║ Hotkey: ${YOUR_HOTKEY:0:20}...                                           ║"
    else
        echo "║ Status: Not registered or not mining on subnet 55                        ║"
        echo "║ Hotkey: ${YOUR_HOTKEY:0:20}...                                           ║"
    fi
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    echo "╔═══════════════════════════════ TOP 10 COMPETITORS ══════════════════════════╗"
    echo "║ Rank │ UID  │ Incentive │ Emissions │ Trust   │ Status                    ║"
    echo "╠══════╬══════╬═══════════╬═══════════╬═════════╬════════════════════════════╣"

    # Get top 10 from metagraph (this is a simplified version)
    METAGRAPH_DATA=$(btcli subnet metagraph --netuid 55 --subtensor.network test 2>/dev/null | head -15 | tail -10 || echo "Data unavailable")
    if [ "$METAGRAPH_DATA" != "Data unavailable" ]; then
        echo "$METAGRAPH_DATA" | nl -v0 | head -10 | while read line; do
            if [[ $line =~ ([0-9]+).*τ\ ([0-9]+\.[0-9]+).*τ\ ([0-9]+\.[0-9]+).*τ\ ([0-9]+\.[0-9]+) ]]; then
                rank=$(( ${BASH_REMATCH[1]} + 1 ))
                incentive=${BASH_REMATCH[2]}
                emissions=${BASH_REMATCH[3]}
                trust=${BASH_REMATCH[4]}
                printf "║ %4d │ %4d │ %9.4f │ %9.6f │ %7.4f │                            ║\n" $rank $rank $incentive $emissions $trust
            fi
        done
    else
        echo "║ Data temporarily unavailable...                                        ║"
    fi

    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    echo "╔═══════════════════════════════ PERFORMANCE ANALYSIS ═══════════════════════╗"
    # Performance analysis based on metrics
    INCENTIVE_NUM=$(echo "$INCENTIVE" | bc -l 2>/dev/null || echo "0")
    if (( $(echo "$INCENTIVE_NUM > 0.001" | bc -l) )); then
        echo "║ Mining Performance: 🟢 EXCELLENT (Incentive > 0.001)                     ║"
    elif (( $(echo "$INCENTIVE_NUM > 0.0001" | bc -l) )); then
        echo "║ Mining Performance: 🟡 GOOD (Incentive > 0.0001)                        ║"
    else
        echo "║ Mining Performance: 🔴 NEEDS IMPROVEMENT (Low incentive)                 ║"
    fi

    TRUST_NUM=$(echo "$TRUST" | bc -l 2>/dev/null || echo "0")
    if (( $(echo "$TRUST_NUM > 0.8" | bc -l) )); then
        echo "║ Trust Reliability: 🟢 HIGH TRUST                                        ║"
    elif (( $(echo "$TRUST_NUM > 0.5" | bc -l) )); then
        echo "║ Trust Reliability: 🟡 MEDIUM TRUST                                      ║"
    else
        echo "║ Trust Reliability: 🔴 LOW TRUST                                         ║"
    fi
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    echo "╔═══════════════════════════════ SYSTEM INFO ════════════════════════════════╗"
    echo "║ Network: Testnet (wss://test.finney.opentensor.ai:443)                    ║"
    echo "║ Subnet: 55 (Precog - Bitcoin Price Prediction)                           ║"
    echo "║ Block Time: ~12 seconds                                                   ║"
    echo "║ Emission Rate: ~0.0002 τ per block                                       ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    echo "╔═══════════════════════════════ CONTROLS ════════════════════════════════════╗"
    echo "║ [R] Refresh now  [Q] Quit  [M] Start Miner  [S] Stop Miner                 ║"
    echo "║ [L] View Logs   [P] Performance Test  [C] Competitor Analysis             ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    # Auto refresh every 30 seconds unless user presses a key
    echo -n "Auto-refreshing in 30 seconds... (or press any key for menu)"
    read -t 30 -n 1 key

    if [[ $key == "q" ]] || [[ $key == "Q" ]]; then
        echo ""
        echo "Goodbye! 👋"
        exit 0
    elif [[ $key == "r" ]] || [[ $key == "R" ]]; then
        continue
    elif [[ $key == "m" ]] || [[ $key == "M" ]]; then
        echo ""
        echo "Starting miner..."
        ./start_miner.sh &
        sleep 2
    elif [[ $key == "s" ]] || [[ $key == "S" ]]; then
        echo ""
        echo "Stopping miner..."
        pkill -f miner.py
        sleep 2
    elif [[ $key == "l" ]] || [[ $key == "L" ]]; then
        echo ""
        echo "Recent miner logs:"
        tail -20 miner.log 2>/dev/null || echo "No logs available"
        echo ""
        read -p "Press Enter to continue..."
    elif [[ $key == "p" ]] || [[ $key == "P" ]]; then
        echo ""
        echo "Running performance validation..."
        python scripts/validate_performance.py --hours 1
        echo ""
        read -p "Press Enter to continue..."
    elif [[ $key == "c" ]] || [[ $key == "C" ]]; then
        echo ""
        echo "Running competitor analysis..."
        python scripts/monitor_competitors.py --netuid 55 --top-n 20
        echo ""
        read -p "Press Enter to continue..."
    fi

done
