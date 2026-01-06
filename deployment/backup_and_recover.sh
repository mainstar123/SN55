#!/bin/bash
# BACKUP AND RECOVERY SCRIPT
# Creates comprehensive backups and provides recovery options
# Usage: ./deployment/backup_and_recover.sh

cd /home/ocean/SN55

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "=== CREATING PRECOG BACKUP ==="
echo "Backup directory: $BACKUP_DIR"

# Backup models and scalers
echo "📦 Backing up models..."
cp -r models/ "$BACKUP_DIR/"

# Backup configuration
echo "⚙️ Backing up configuration..."
cp .env.miner "$BACKUP_DIR/" 2>/dev/null || echo "No .env.miner found"

# Backup wallet files (encrypted keys - handle with care!)
echo "🔑 Backing up wallet files..."
if [ -d "~/.bittensor/wallets" ]; then
    mkdir -p "$BACKUP_DIR/wallets"
    cp -r ~/.bittensor/wallets/precog_coldkey "$BACKUP_DIR/wallets/" 2>/dev/null || echo "No wallet files to backup"
fi

# Backup wallet configuration
echo "⚙️ Backing up wallet configuration..."
if [ -f ".env.miner" ]; then
    source .env.miner
    echo "COLDKEY=$COLDKEY" > "$BACKUP_DIR/wallet_config.txt"
    echo "MINER_HOTKEY=$MINER_HOTKEY" >> "$BACKUP_DIR/wallet_config.txt"
fi

# Backup logs
echo "📝 Backing up logs..."
mkdir -p "$BACKUP_DIR/logs"
cp logs/*.log "$BACKUP_DIR/logs/" 2>/dev/null || echo "No logs to backup"
pm2 logs precog_best_miner > "$BACKUP_DIR/pm2_logs.txt" 2>/dev/null || echo "No PM2 logs"

# Backup PM2 configuration
echo "💾 Backing up PM2 configuration..."
pm2 save
cp ~/.pm2/dump.pm2 "$BACKUP_DIR/" 2>/dev/null || echo "No PM2 dump found"

# Backup deployment scripts
echo "📜 Backing up deployment scripts..."
cp -r deployment/ "$BACKUP_DIR/"

# Create backup manifest
cat > "$BACKUP_DIR/backup_manifest.txt" << EOF
PRECOG SUBNET BACKUP MANIFEST
Created: $(date)
Location: $BACKUP_DIR

CONTENTS:
- models/: Trained models and scalers
- wallets/: Encrypted wallet files (secure backup)
- .env.miner: Environment configuration
- wallet_config.txt: Wallet names and configuration
- logs/: Miner logs
- pm2_logs.txt: PM2 process logs
- dump.pm2: PM2 configuration
- deployment/: All deployment scripts

RECOVERY INSTRUCTIONS:
1. Stop current miner: pm2 stop precog_best_miner
2. Restore models: cp -r models/* /home/ocean/SN55/models/
3. Restore config: cp .env.miner /home/ocean/SN55/
4. Restore PM2: pm2 kill && cp dump.pm2 ~/.pm2/ && pm2 resurrect
5. Restart miner: ./deployment/deploy_best_model.sh
EOF

# Calculate backup size
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "Backup size: $BACKUP_SIZE"

echo "✅ Backup created successfully at: $BACKUP_DIR"
echo ""

# Create recovery script
cat > "$BACKUP_DIR/restore.sh" << 'EOF'
#!/bin/bash
echo "=== RESTORING PRECOG BACKUP ==="
echo "From: $(pwd)"

# Confirm restoration
read -p "This will overwrite current models and configuration. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restoration cancelled"
    exit 1
fi

# Stop current miner
echo "Stopping current miner..."
pm2 stop precog_best_miner 2>/dev/null || true

# Restore models
echo "Restoring models..."
cp -r models/* /home/ocean/SN55/models/

# Restore configuration
echo "Restoring configuration..."
cp .env.miner /home/ocean/SN55/ 2>/dev/null || echo "No .env.miner to restore"

# Restore PM2 configuration
echo "Restoring PM2 configuration..."
pm2 kill 2>/dev/null || true
cp dump.pm2 ~/.pm2/ 2>/dev/null || echo "No PM2 dump to restore"
pm2 resurrect 2>/dev/null || echo "PM2 resurrection failed"

# Restore deployment scripts
echo "Restoring deployment scripts..."
cp -r deployment/* /home/ocean/SN55/deployment/ 2>/dev/null || echo "No deployment scripts to restore"

echo "✅ Restoration complete!"
echo "Start miner with: ./deployment/deploy_best_model.sh"
EOF

chmod +x "$BACKUP_DIR/restore.sh"

echo "RECOVERY OPTIONS:"
echo "1. Quick restore: cd $BACKUP_DIR && ./restore.sh"
echo "2. Manual restore: Follow instructions in backup_manifest.txt"
echo "3. Selective restore: Copy individual files as needed"
echo ""

# List all backups
echo "ALL BACKUPS:"
ls -la backups/ | head -10
echo ""

echo "=========================================="
