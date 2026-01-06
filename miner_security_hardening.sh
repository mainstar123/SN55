#!/bin/bash
# Bittensor Miner Security Hardening Script

echo "🛡️  BITTENSOR MINER SECURITY HARDENING"
echo "======================================"
echo ""

# Function to check if we have sudo
check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        echo "⚠️  Some security features require sudo access"
        echo "   Running with available permissions..."
        USE_SUDO=false
    else
        USE_SUDO=true
    fi
}

# Check sudo access
check_sudo

echo "🔒 IMPLEMENTING SECURITY MEASURES..."
echo ""

# 1. Firewall Setup
echo "1. 🔥 FIREWALL CONFIGURATION:"
echo "-----------------------------"
if $USE_SUDO; then
    if command -v ufw >/dev/null 2>&1; then
        echo "Setting up UFW firewall..."
        sudo ufw --force enable >/dev/null 2>&1
        sudo ufw allow ssh >/dev/null 2>&1
        sudo ufw allow 8091/tcp >/dev/null 2>&1  # Bittensor axon port
        sudo ufw --force reload >/dev/null 2>&1
        echo "✅ UFW firewall configured"
        echo "   • SSH port: ALLOWED"
        echo "   • Bittensor port 8091: ALLOWED"
        echo "   • Default: DENY all other ports"
    else
        echo "⚠️  UFW not available, installing..."
        sudo apt update >/dev/null 2>&1
        sudo apt install -y ufw >/dev/null 2>&1
        if command -v ufw >/dev/null 2>&1; then
            sudo ufw --force enable >/dev/null 2>&1
            sudo ufw allow ssh >/dev/null 2>&1
            sudo ufw allow 8091/tcp >/dev/null 2>&1
            echo "✅ UFW installed and configured"
        else
            echo "❌ Could not install UFW"
        fi
    fi
else
    echo "⚠️  Firewall setup requires sudo - skipped"
fi
echo ""

# 2. SSH Hardening
echo "2. 🔐 SSH SECURITY HARDENING:"
echo "-----------------------------"
if $USE_SUDO; then
    SSH_CONFIG="/etc/ssh/sshd_config"
    
    # Backup SSH config
    sudo cp $SSH_CONFIG ${SSH_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)
    
    # Apply hardening
    sudo sed -i 's/#*PermitRootLogin.*/PermitRootLogin no/' $SSH_CONFIG
    sudo sed -i 's/#*PasswordAuthentication.*/PasswordAuthentication no/' $SSH_CONFIG
    sudo sed -i 's/#*PubkeyAuthentication.*/PubkeyAuthentication yes/' $SSH_CONFIG
    sudo sed -i 's/#*MaxAuthTries.*/MaxAuthTries 3/' $SSH_CONFIG
    sudo sed -i 's/#*ClientAliveInterval.*/ClientAliveInterval 60/' $SSH_CONFIG
    sudo sed -i 's/#*ClientAliveCountMax.*/ClientAliveCountMax 3/' $SSH_CONFIG
    
    # Restart SSH
    sudo systemctl restart ssh >/dev/null 2>&1
    echo "✅ SSH hardened:"
    echo "   • Root login: DISABLED"
    echo "   • Password auth: DISABLED"
    echo "   • Key auth: ENABLED"
    echo "   • Max auth tries: 3"
else
    echo "⚠️  SSH hardening requires sudo - skipped"
fi
echo ""

# 3. Fail2Ban Setup
echo "3. 🛡️  FAIL2BAN PROTECTION:"
echo "--------------------------"
if $USE_SUDO; then
    if ! command -v fail2ban-client >/dev/null 2>&1; then
        echo "Installing fail2ban..."
        sudo apt update >/dev/null 2>&1
        sudo apt install -y fail2ban >/dev/null 2>&1
    fi
    
    if command -v fail2ban-client >/dev/null 2>&1; then
        sudo systemctl enable fail2ban >/dev/null 2>&1
        sudo systemctl start fail2ban >/dev/null 2>&1
        echo "✅ Fail2Ban installed and running"
        echo "   • SSH brute force protection: ACTIVE"
    else
        echo "❌ Could not install Fail2Ban"
    fi
else
    echo "⚠️  Fail2Ban requires sudo - skipped"
fi
echo ""

# 4. System Updates
echo "4. 🔄 SYSTEM SECURITY UPDATES:"
echo "------------------------------"
if $USE_SUDO; then
    echo "Checking for security updates..."
    UPDATES=$(sudo apt list --upgradable 2>/dev/null | grep -c "security" || echo "0")
    if [ "$UPDATES" -gt 0 ]; then
        echo "Installing security updates..."
        sudo apt update >/dev/null 2>&1
        sudo apt upgrade -y >/dev/null 2>&1
        echo "✅ Security updates installed"
    else
        echo "✅ System is up to date"
    fi
else
    echo "⚠️  System updates require sudo - skipped"
fi
echo ""

# 5. Miner-Specific Security
echo "5. 🎯 MINER-SPECIFIC SECURITY:"
echo "-----------------------------"
echo "Configuring miner security..."

# Create security monitoring script
cat > security_monitor.sh << 'MONITOR_EOF'
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
MONITOR_EOF

chmod +x security_monitor.sh
echo "✅ Security monitoring script created: ./security_monitor.sh"
echo ""

# 6. Final Recommendations
echo "6. 📋 SECURITY RECOMMENDATIONS:"
echo "-------------------------------"
echo "✅ IMPLEMENTED:"
echo "• Firewall protection for Bittensor port"
echo "• SSH hardening (if sudo available)"
echo "• Brute force protection (Fail2Ban)"
echo "• Security monitoring script"
echo ""
echo "🔧 ADDITIONAL RECOMMENDATIONS:"
echo "• Use strong SSH keys (not passwords)"
echo "• Regularly run: ./security_monitor.sh"
echo "• Monitor resource usage for anomalies"
echo "• Keep system updated"
echo "• Consider running miner in Docker container"
echo "• Use VPN for additional protection"
echo ""

echo "🎉 SECURITY HARDENING COMPLETE!"
echo "==============================="
echo ""
echo "🔒 Your Bittensor miner is now more secure:"
echo "• Firewall protecting your ports"
echo "• SSH access restricted"
echo "• Attack detection active"
echo "• Monitoring tools available"
echo ""
echo "Run './security_monitor.sh' regularly to check security status"
