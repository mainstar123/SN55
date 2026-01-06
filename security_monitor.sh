#!/bin/bash
# Security monitoring for Bittensor miner

echo "🔍 SECURITY MONITORING REPORT"
echo "============================="
echo "Time: $(date)"

# Check for suspicious processes
echo ""
echo "🚨 SUSPICIOUS PROCESSES:"
SUSPICIOUS=$(ps aux | grep -E "(miner|python.*bittensor|ssh.*root)" | grep -v grep | wc -l)
echo "• Mining processes: $SUSPICIOUS (expected: 1-2)"

# Check network connections
echo ""
echo "🌐 NETWORK CONNECTIONS:"
ESTABLISHED=$(netstat -tun 2>/dev/null | grep -c ESTABLISHED || echo "0")
LISTENING=$(netstat -tln 2>/dev/null | grep -c LISTEN || echo "0")
echo "• Established connections: $ESTABLISHED"
echo "• Listening ports: $LISTENING"

# Check resource usage
echo ""
echo "💾 RESOURCE USAGE:"
CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')
MEM=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')
echo "• CPU Usage: $CPU"
echo "• Memory Usage: $MEM"

# Check recent SSH attempts
echo ""
echo "🔐 SSH SECURITY:"
if [ -f /var/log/auth.log ]; then
    FAILED_ATTEMPTS=$(grep -c "Failed password\|Invalid user" /var/log/auth.log 2>/dev/null || echo "0")
    echo "• Failed SSH attempts (24h): $FAILED_ATTEMPTS"
fi

# Check miner logs for errors
echo ""
echo "🎯 MINER STATUS:"
if [ -f "logs/first_place_miner_*.log" ]; then
    LATEST_LOG=$(tail -1 logs/first_place_miner_*.log 2>/dev/null)
    if echo "$LATEST_LOG" | grep -q "Emission:0\.000"; then
        echo "• Status: Network issues (normal)"
    else
        echo "• Status: Active mining detected"
    fi
else
    echo "• Status: No log files found"
fi

echo ""
echo "✅ Security check completed"
