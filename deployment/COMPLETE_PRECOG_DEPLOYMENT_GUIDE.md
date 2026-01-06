# 🚀 **COMPLETE PRECOG SUBNET 55 DEPLOYMENT GUIDE**

## **Table of Contents**
1. [Wallet Creation & Key Management](#wallet-creation)
2. [TAO Acquisition & Transfer](#tao-acquisition)
3. [Subnet Registration](#subnet-registration)
4. [Environment Configuration](#environment-config)
5. [Miner Deployment](#miner-deployment)
6. [Performance Monitoring](#performance-monitoring)
7. [Post-Deployment Model Improvement](#model-improvement)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Optimization](#advanced-optimization)

---

## **1. WALLET CREATION & KEY MANAGEMENT** <a name="wallet-creation"></a>

### **Prerequisites**
```bash
# Ensure virtual environment is activated
cd /home/ocean/SN55
source venv/bin/activate

# Verify bittensor CLI is installed
btcli --version
```

### **Step 1: Create Coldkey (Secure Wallet)**
```bash
# Create your main coldkey wallet
btcli wallet new_coldkey --wallet.name precog_coldkey

# Expected prompts:
# - Enter password for coldkey (use strong password, save securely)
# - Confirm password
# - Optional: Add description

# Verify coldkey creation
btcli wallet list
```

**Output Example:**
```
Wallets
├── precog_coldkey (coldkey)
```

### **Step 2: Create Hotkey (Mining Key)**
```bash
# Create hotkey for mining operations
btcli wallet new_hotkey --wallet.name precog_coldkey --wallet.hotkey miner_hotkey

# Expected prompts:
# - Enter coldkey password (from step 1)
# - Optional: Add hotkey description

# Verify hotkey creation
btcli wallet overview --wallet.name precog_coldkey
```

**Output Example:**
```
Wallet: precog_coldkey
├── Balance: τ 0.0000
├── Hotkeys:
│   └── miner_hotkey (τ 0.0000)
```

### **Step 3: Get Your Wallet Address**
```bash
# Get your coldkey address for TAO transfers
btcli wallet overview --wallet-name precog_coldkey

# Alternative: List all wallets with addresses
btcli wallet list
```

**The wallet address appears directly in the output (e.g., 5Hft1rGjPQrUvW9Q8ZFq6A1AqQE8qgEdYKLokF7x7QjD5kPf)**
**Share this address with friends to receive TAO for registration**

### **Step 4: Backup Your Wallets**
```bash
# Wallet files are already encrypted and stored securely
# Copy wallet directory to secure backup location

# Create backup directory
mkdir -p ~/wallet_backup

# Backup entire wallet directory (contains encrypted keys)
cp -r ~/.bittensor/wallets/precog_coldkey ~/wallet_backup/

# Verify backup
ls -la ~/wallet_backup/precog_coldkey/

# Store backup in secure location (encrypted drive, offline storage)
# NEVER share these files or your password with anyone
```

### **Security Best Practices**
- **Never share your coldkey password or JSON files**
- **Use different passwords for coldkey and hotkey**
- **Store coldkey backups offline (USB drive in safe)**
- **Only use hotkey for mining operations**

---

## **2. TAO ACQUISITION & TRANSFER** <a name="tao-acquisition"></a>

### **Method 1: From Friends (Recommended)**
```bash
# Share your coldkey address with friends
echo "My Precog wallet address: $(btcli wallet overview --wallet.name precog_coldkey | grep 'Address:' | awk '{print $2}')"

# After receiving TAO, verify balance
btcli wallet overview --wallet.name precog_coldkey
```

### **Method 2: Testnet Faucet (For Testing)**
```bash
# Get testnet TAO first
btcli wallet faucet --wallet.name precog_coldkey --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443

# Bridge to mainnet (if available) or use testnet for practice
```

### **Method 3: Purchase TAO**
- **Exchanges**: Binance, KuCoin, Gate.io
- **Minimum Required**: ~0.001 TAO for registration
- **Recommended**: 0.1+ TAO for initial operations

### **Transfer Verification**
```bash
# Monitor incoming transfers
watch -n 30 "btcli wallet overview --wallet.name precog_coldkey"

# Check transaction history
btcli wallet history --wallet.name precog_coldkey
```

---

## **3. SUBNET REGISTRATION** <a name="subnet-registration"></a>

### **Step 1: Verify Network Connection**
```bash
# Test mainnet connection
btcli subnet list --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443

# Verify subnet 55 exists
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 | head -20
```

### **Step 2: Register on Subnet 55**
```bash
# Register your hotkey on Precog subnet
btcli subnet register \
  --netuid 55 \
  --wallet.name precog_coldkey \
  --wallet.hotkey miner_hotkey \
  --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443

# Expected cost: ~0.001 TAO
# Wait for confirmation (1-2 minutes)
```

### **Step 3: Verify Registration**
```bash
# Check registration status
btcli wallet overview --wallet.name precog_coldkey

# Verify hotkey appears in subnet
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 | grep -i precog

# Get your UID
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 | nl | grep -i precog
```

### **Step 4: Enable Auto-Staking (Optional)**
```bash
# Enable automatic staking of rewards
btcli stake set_auto_stake --wallet.name precog_coldkey --netuid 55

# Verify auto-stake is enabled
btcli stake list --wallet.name precog_coldkey
```

---

## **4. ENVIRONMENT CONFIGURATION** <a name="environment-config"></a>

### **Create .env.miner File**
```bash
# Copy and edit the environment file
cp .env.miner.example .env.miner

# Edit with your details
nano .env.miner
```

**.env.miner contents:**
```bash
# Network Configuration
# Options: localnet, testnet, finney
NETWORK=finney

# Wallet Configuration
COLDKEY=precog_coldkey
MINER_HOTKEY=miner_hotkey

# Node Configuration
MINER_NAME=miner
# This port must be open to accept incoming TCP connections.
MINER_PORT=8092

# Miner Settings
TIMEOUT=16
VPERMIT_TAO_LIMIT=2

#Adjust this function if you would like to specify a custom forward function
FORWARD_FUNCTION=custom_model

# Logging
# Options: info, debug, trace
LOGGING_LEVEL=debug

# Local Subtensor Configuration
# Only used if you run your own subtensor node
# LOCALNET=ws://127.0.0.1:9945
```

### **Verify Configuration**
```bash
# Test environment loading
make miner ENV_FILE=.env.miner --dry-run

# Check for any configuration errors
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv('.env.miner')
print('COLDKEY:', os.getenv('COLDKEY'))
print('HOTKEY:', os.getenv('MINER_HOTKEY'))
print('NETWORK:', os.getenv('NETWORK'))
"
```

---

## **5. MINER DEPLOYMENT** <a name="miner-deployment"></a>

### **Method 1: Using Makefile (Recommended)**
```bash
# Deploy with custom model
make miner_custom ENV_FILE=.env.miner

# Alternative: Deploy base miner
make miner ENV_FILE=.env.miner
```

### **Method 2: Direct PM2 Deployment**
```bash
# Start miner with PM2
pm2 start --name precog_miner python3 -- precog/miners/miner.py \
  --neuron.name miner \
  --wallet.name precog_coldkey \
  --wallet.hotkey miner_hotkey \
  --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
  --axon.port 8092 \
  --netuid 55 \
  --logging.level debug \
  --timeout 16 \
  --vpermit_tao_limit 2 \
  --forward_function custom_model

# Save PM2 configuration
pm2 save
pm2 startup
```

### **Method 3: Domination Miner (Advanced)**
```bash
# For the domination model (if available)
./start_mainnet_domination_miner.sh

# Or manually
DOMINATION_MODE=true pm2 start --name domination_miner python3 -- precog/miners/miner.py \
  --neuron.name domination_miner \
  --wallet.name precog_coldkey \
  --wallet.hotkey miner_hotkey \
  --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
  --axon.port 8092 \
  --netuid 55 \
  --logging.level info \
  --forward_function custom_model
```

### **Verify Deployment**
```bash
# Check PM2 status
pm2 list

# View miner logs
pm2 logs precog_miner

# Check network connectivity
curl -s http://localhost:8092/ping

# Verify subnet connection
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 | grep your_uid
```

---

## **6. PERFORMANCE MONITORING** <a name="performance-monitoring"></a>

### **Real-Time Monitoring Commands**

#### **Miner Health Check**
```bash
# PM2 status
pm2 list
pm2 monit precog_miner

# Process monitoring
ps aux | grep miner.py
top -p $(pgrep -f miner.py)

# Network connectivity
netstat -tlnp | grep 8092
```

#### **Live Log Monitoring**
```bash
# Real-time logs
pm2 logs precog_miner --lines 50

# Follow logs continuously
pm2 logs precog_miner --follow

# Filter specific messages
pm2 logs precog_miner --follow | grep -E "(Prediction|Reward|Error)"
```

#### **Performance Metrics**
```bash
# Prediction frequency
pm2 logs precog_miner | grep "Prediction made" | wc -l

# Average response time
pm2 logs precog_miner | grep "response_time" | awk '{sum+=$2; count++} END {print "Avg:", sum/count, "ms"}'

# Success rate
pm2 logs precog_miner | grep -c "success\|Success"
```

### **Wallet & Earnings Monitoring**
```bash
# Current balance
btcli wallet overview --wallet.name precog_coldkey

# Recent transactions
btcli wallet history --wallet.name precog_coldkey --limit 10

# Staking information
btcli stake list --wallet.name precog_coldkey

# Daily earnings calculation
btcli wallet history --wallet.name precog_coldkey --days 1 | grep -E "[0-9]+\.[0-9]+.*τ" | awk '{sum+=$2} END {print "Daily earnings: τ", sum}'
```

### **Subnet Position Monitoring**
```bash
# Your position in subnet
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 | grep -A 5 -B 5 precog

# Top performers
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 | head -20

# Network statistics
btcli subnet list | grep "55"

# Validator information
btcli subnet metagraph --netuid 55 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 | tail -10
```

### **Advanced Monitoring Scripts**
```bash
# Create monitoring dashboard
cat > monitor_precog.sh << 'EOF'
#!/bin/bash
echo "=== PRECOG MINER DASHBOARD ==="
echo "Time: $(date)"
echo ""

echo "1. MINER STATUS:"
pm2 jlist | jq -r '.[] | select(.name=="precog_miner") | "\(.name): \(.pm2_env.status) (PID: \(.pid))"'
echo ""

echo "2. WALLET BALANCE:"
btcli wallet overview --wallet.name precog_coldkey | grep -E "(Balance|τ)"
echo ""

echo "3. PREDICTIONS (Last Hour):"
pm2 logs precog_miner --lines 1000 2>/dev/null | grep "Prediction made" | tail -10
echo ""

echo "4. SUBNET POSITION:"
YOUR_UID=$(btcli subnet metagraph --netuid 55 | grep -n precog | cut -d: -f1)
echo "Your approximate position: #$YOUR_UID"
echo ""

echo "5. SYSTEM RESOURCES:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
echo "Memory: $(free | grep Mem | awk '{printf "%.2f%%", $3/$2 * 100.0}')"
echo ""

echo "6. NETWORK STATS:"
echo "Active connections: $(netstat -t | grep ESTABLISHED | wc -l)"
echo "Port 8092 status: $(netstat -tlnp | grep 8092 | wc -l) connections"
EOF

chmod +x monitor_precog.sh
```

---

## **7. POST-DEPLOYMENT MODEL IMPROVEMENT** <a name="model-improvement"></a>

### **Performance Analysis**
```bash
# Extract performance data
pm2 logs precog_miner > miner_logs.txt

# Analyze prediction accuracy
grep "Prediction made" miner_logs.txt | awk '{print $6}' > predictions.txt
grep "Actual price" miner_logs.txt | awk '{print $3}' > actuals.txt

# Calculate MAPE
python3 -c "
import numpy as np
pred = np.loadtxt('predictions.txt')
actual = np.loadtxt('actuals.txt')
mape = np.mean(np.abs((actual - pred) / actual)) * 100
print(f'MAPE: {mape:.4f}%')
"
```

### **Data Collection for Retraining**
```bash
# Collect live market data
python3 scripts/fetch_training_data.py --days 7 --output live_training_data.csv

# Extract miner performance data
python3 -c "
import pandas as pd
import re

# Parse logs for training data
with open('miner_logs.txt', 'r') as f:
    logs = f.readlines()

training_data = []
for line in logs:
    if 'Prediction made' in line:
        # Extract features and prediction
        match = re.search(r'features: (.+?) prediction: (.+?) actual: (.+)', line)
        if match:
            training_data.append({
                'features': eval(match.group(1)),
                'prediction': float(match.group(2)),
                'actual': float(match.group(3))
            })

df = pd.DataFrame(training_data)
df.to_csv('miner_performance_data.csv', index=False)
print(f'Collected {len(training_data)} training samples')
"
```

### **Model Retraining Strategies**

#### **Strategy 1: Online Learning**
```bash
# Implement online learning updates
cat > online_retrainer.py << 'EOF'
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import bittensor as bt

class OnlineRetrainer:
    def __init__(self, model_path, scaler_path):
        self.model = torch.load(model_path)
        self.scaler = pd.read_pickle(scaler_path)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()
        self.performance_buffer = []
        
    def update_model(self, features, prediction, actual):
        # Add to performance buffer
        self.performance_buffer.append({
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual) / actual
        })
        
        # Retrain if buffer is full
        if len(self.performance_buffer) >= 100:
            self._retrain_on_buffer()
            
    def _retrain_on_buffer(self):
        bt.logging.info(f"Retraining on {len(self.performance_buffer)} samples")
        
        # Prepare data
        data = pd.DataFrame(self.performance_buffer)
        X = self.scaler.transform(data['features'].tolist())
        y = data['actual'].values
        
        # Fine-tune model
        self.model.train()
        for epoch in range(10):
            self.optimizer.zero_grad()
            outputs = self.model(torch.FloatTensor(X))
            loss = self.criterion(outputs.squeeze(), torch.FloatTensor(y))
            loss.backward()
            self.optimizer.step()
            
        # Save updated model
        torch.save(self.model, 'models/updated_model.pth')
        bt.logging.success("Model updated successfully")
        
        # Clear buffer
        self.performance_buffer = []

# Usage in miner
retrainer = OnlineRetrainer('models/domination_model_trained.pth', 'models/feature_scaler.pkl')
# Call retrainer.update_model() after each prediction
EOF
```

#### **Strategy 2: Periodic Retraining**
```bash
# Create automated retraining script
cat > automated_retraining.sh << 'EOF'
#!/bin/bash
cd /home/ocean/SN55

echo "=== AUTOMATED MODEL RETRAINING ==="
echo "Time: $(date)"

# Check miner performance
echo "Checking miner performance..."
PERFORMANCE=$(pm2 logs precog_miner --lines 1000 2>/dev/null | grep "MAPE" | tail -1 | awk '{print $2}' | sed 's/%//')

if (( $(echo "$PERFORMANCE > 0.15" | bc -l) )); then
    echo "Performance degraded (MAPE: $PERFORMANCE%), initiating retraining..."
    
    # Collect fresh data
    echo "Collecting fresh training data..."
    python3 scripts/fetch_training_data.py --days 3 --output fresh_data.csv
    
    # Retrain model
    echo "Retraining model..."
    python3 -c "
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('fresh_data.csv')
X = data.drop(['target'], axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Load existing model
model = torch.load('models/domination_model_trained.pth')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Fine-tune
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(torch.FloatTensor(X_train.values))
    loss = torch.nn.MSELoss()(outputs.squeeze(), torch.FloatTensor(y_train.values))
    loss.backward()
    optimizer.step()
    
print(f'Final loss: {loss.item():.6f}')

# Save updated model
torch.save(model, 'models/retrained_model.pth')
print('Model retrained successfully')
"

    # Backup old model
    cp models/domination_model_trained.pth models/backup_$(date +%Y%m%d_%H%M%S).pth
    
    # Deploy new model
    cp models/retrained_model.pth models/domination_model_trained.pth
    
    # Restart miner with new model
    pm2 restart precog_miner
    
    echo "Model retrained and deployed successfully!"
else
    echo "Performance good (MAPE: $PERFORMANCE%), no retraining needed"
fi
EOF

# Make executable and schedule
chmod +x automated_retraining.sh
# Add to crontab for daily retraining: 0 2 * * * /home/ocean/SN55/automated_retraining.sh
```

#### **Strategy 3: Ensemble Model Updates**
```bash
# Create ensemble update script
cat > ensemble_updater.py << 'EOF'
import torch
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

class EnsembleUpdater:
    def __init__(self, model_paths):
        self.models = [torch.load(path) for path in model_paths]
        self.weights = np.ones(len(self.models)) / len(self.models)
        self.performance_history = []
        
    def predict(self, X):
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(torch.FloatTensor(X)).item()
                predictions.append(pred)
        
        # Weighted ensemble prediction
        return np.average(predictions, weights=self.weights)
    
    def update_weights(self, X, y_true):
        # Calculate individual model errors
        errors = []
        for model in self.models:
            pred = self.predict(X.reshape(1, -1))
            error = abs(pred - y_true) / y_true
            errors.append(error)
        
        # Update weights based on performance
        total_error = sum(errors)
        if total_error > 0:
            self.weights = np.array([1/error if error > 0 else 1.0 for error in errors])
            self.weights = self.weights / self.weights.sum()
        
        self.performance_history.append({
            'errors': errors,
            'weights': self.weights.copy(),
            'avg_error': np.average(errors, weights=self.weights)
        })

# Usage
updater = EnsembleUpdater([
    'models/gru_model.pth',
    'models/transformer_model.pth',
    'models/lstm_model.pth'
])

# Update weights after each prediction
# updater.update_weights(features, actual_price)
EOF
```

### **Hyperparameter Optimization**
```bash
# Automated hyperparameter tuning
cat > hyperparameter_optimizer.py << 'EOF'
import optuna
import torch
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def objective(trial):
    # Define hyperparameter search space
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    # Train model with these parameters
    model = create_model(hidden_size, num_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Cross-validation on recent data
    cv_scores = []
    tscv = TimeSeriesSplit(n_splits=3)
    
    for train_idx, val_idx in tscv.split(X_recent):
        # Train on subset
        model_copy = create_model(hidden_size, num_layers, dropout)
        train_model(model_copy, X_recent[train_idx], y_recent[train_idx])
        
        # Validate
        val_pred = model_copy(torch.FloatTensor(X_recent[val_idx]))
        val_mape = mean_absolute_percentage_error(y_recent[val_idx], val_pred.detach().numpy())
        cv_scores.append(val_mape)
    
    return np.mean(cv_scores)

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get best parameters
best_params = study.best_params
print(f"Best parameters: {best_params}")
print(f"Best MAPE: {study.best_value:.4f}%")

# Retrain with best parameters
best_model = create_model(**best_params)
train_model(best_model, X_full, y_full)
torch.save(best_model, 'models/optimized_model.pth')
EOF
```

---

## **8. TROUBLESHOOTING** <a name="troubleshooting"></a>

### **Common Issues & Solutions**

#### **Miner Won't Start**
```bash
# Check dependencies
python3 -c "import bittensor, torch, pandas; print('Dependencies OK')"

# Check wallet access
btcli wallet overview --wallet.name precog_coldkey

# Check port availability
netstat -tlnp | grep 8092

# Restart with debug logging
LOGGING_LEVEL=trace make miner_custom ENV_FILE=.env.miner
```

#### **No Predictions Being Made**
```bash
# Check validator connections
pm2 logs precog_miner | grep "validator\|synapse"

# Verify market data access
pm2 logs precog_miner | grep "market\|price\|data"

# Check confidence thresholds
pm2 logs precog_miner | grep "confidence\|threshold"

# Test manual prediction
python3 -c "
from precog.miners.custom_model import forward
import pandas as pd
# Test with sample data
"
```

#### **Low Rewards**
```bash
# Analyze prediction accuracy
pm2 logs precog_miner | grep -A 5 -B 5 "Reward\|reward"

# Check response time
pm2 logs precog_miner | grep "response_time" | awk '{print $2}' | sort -n

# Compare with competitors
btcli subnet metagraph --netuid 55 | head -20

# Adjust parameters
nano .env.miner
# Increase timeout, adjust vpermit limits
pm2 restart precog_miner
```

#### **Connection Issues**
```bash
# Test network connectivity
ping archive.substrate.network

# Check endpoint availability
curl -s wss://entrypoint-finney.opentensor.ai:443 >/dev/null && echo "OK" || echo "FAIL"

# Switch endpoints
sed -i 's|entrypoint-finney|archive.substrate|g' .env.miner
pm2 restart precog_miner

# Check firewall
ufw status
sudo ufw allow 8092
```

#### **Memory/Resource Issues**
```bash
# Monitor resources
htop
nvidia-smi

# Reduce model size
python3 scripts/quantize_model.py models/domination_model_trained.pth

# Optimize batch processing
sed -i 's/batch_size=32/batch_size=16/g' precog/miners/custom_model.py
pm2 restart precog_miner
```

---

## **9. ADVANCED OPTIMIZATION** <a name="advanced-optimization"></a>

### **Peak Hour Optimization**
```bash
# Configure peak hour settings
cat > peak_hour_config.py << 'EOF'
import datetime
import pytz

class PeakHourOptimizer:
    def __init__(self):
        self.peak_hours = [
            (9, 11),   # UTC 9-11
            (13, 15)   # UTC 13-15
        ]
        self.timezone = pytz.timezone('UTC')
    
    def is_peak_hour(self):
        now = datetime.datetime.now(self.timezone)
        current_hour = now.hour
        
        for start, end in self.peak_hours:
            if start <= current_hour < end:
                return True
        return False
    
    def get_dynamic_settings(self):
        if self.is_peak_hour():
            return {
                'prediction_frequency': 5,  # minutes
                'confidence_threshold': 0.75,
                'batch_size': 8
            }
        else:
            return {
                'prediction_frequency': 15,  # minutes
                'confidence_threshold': 0.85,
                'batch_size': 4
            }

# Usage in miner
optimizer = PeakHourOptimizer()
settings = optimizer.get_dynamic_settings()
EOF
```

### **Market Regime Detection**
```bash
# Implement market regime adaptation
cat > regime_detector.py << 'EOF'
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class MarketRegimeDetector:
    def __init__(self):
        self.regimes = {
            0: 'ranging',
            1: 'bull_trend', 
            2: 'bear_trend',
            3: 'volatile'
        }
        self.kmeans = None
        
    def fit(self, historical_data):
        # Extract features for regime detection
        features = []
        for i in range(len(historical_data) - 20):
            window = historical_data[i:i+20]
            features.append([
                np.std(window),  # volatility
                np.mean(window), # level
                window[-1] - window[0],  # trend
                np.max(window) - np.min(window)  # range
            ])
        
        # Fit clustering model
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        self.kmeans.fit(features)
    
    def detect_regime(self, recent_data):
        features = [
            np.std(recent_data),
            np.mean(recent_data),
            recent_data[-1] - recent_data[0],
            np.max(recent_data) - np.min(recent_data)
        ]
        
        regime_id = self.kmeans.predict([features])[0]
        return self.regimes[regime_id]
    
    def get_regime_settings(self, regime):
        settings = {
            'ranging': {
                'frequency': 60,  # 1 hour
                'threshold': 0.90,
                'model': 'conservative'
            },
            'bull_trend': {
                'frequency': 30,  # 30 min
                'threshold': 0.80,
                'model': 'aggressive'
            },
            'bear_trend': {
                'frequency': 45,  # 45 min
                'threshold': 0.85,
                'model': 'defensive'
            },
            'volatile': {
                'frequency': 15,  # 15 min
                'threshold': 0.75,
                'model': 'adaptive'
            }
        }
        return settings.get(regime, settings['ranging'])

# Usage
detector = MarketRegimeDetector()
detector.fit(historical_prices)
current_regime = detector.detect_regime(recent_prices)
settings = detector.get_regime_settings(current_regime)
EOF
```

### **Automated Scaling**
```bash
# Create auto-scaling script
cat > auto_scaler.sh << 'EOF'
#!/bin/bash

# Monitor system resources and scale miner accordingly
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')

# Scale based on resources
if (( $(echo "$CPU_USAGE < 70 && $MEMORY_USAGE < 80" | bc -l) )); then
    # Resources available, scale up
    pm2 scale precog_miner +1
    echo "Scaled up: CPU ${CPU_USAGE}%, Memory ${MEMORY_USAGE}%"
elif (( $(echo "$CPU_USAGE > 90 || $MEMORY_USAGE > 90" | bc -l) )); then
    # Resources constrained, scale down
    pm2 scale precog_miner -1
    echo "Scaled down: CPU ${CPU_USAGE}%, Memory ${MEMORY_USAGE}%"
fi

# Adjust model parameters based on load
if (( $(echo "$CPU_USAGE > 85" | bc -l) )); then
    # High CPU, use lighter model
    export MODEL_TYPE=light
else
    export MODEL_TYPE=full
fi
EOF

# Schedule auto-scaling
chmod +x auto_scaler.sh
# Add to crontab: */5 * * * * /home/ocean/SN55/auto_scaler.sh
```

### **Backup & Recovery**
```bash
# Create comprehensive backup script
cat > backup_and_recover.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/home/ocean/precog_backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

echo "=== CREATING PRECOG BACKUP ==="

# Backup models
cp -r models/ $BACKUP_DIR/

# Backup configuration
cp .env.miner $BACKUP_DIR/
cp -r ~/.bittensor/wallets/precog_coldkey $BACKUP_DIR/wallet_backup/

# Backup logs
pm2 logs precog_miner > $BACKUP_DIR/miner_logs.txt

# Backup PM2 configuration
pm2 save
cp ~/.pm2/dump.pm2 $BACKUP_DIR/

# Create recovery script
cat > $BACKUP_DIR/restore.sh << RESTORE_EOF
#!/bin/bash
echo "=== RESTORING PRECOG BACKUP ==="

# Restore models
cp -r models/* /home/ocean/SN55/models/

# Restore configuration
cp .env.miner /home/ocean/SN55/

# Restore PM2
pm2 kill
cp dump.pm2 ~/.pm2/
pm2 resurrect

echo "Restoration complete!"
RESTORE_EOF

chmod +x $BACKUP_DIR/restore.sh

echo "Backup created at: $BACKUP_DIR"
echo "To restore: cd $BACKUP_DIR && ./restore.sh"
EOF

chmod +x backup_and_recover.sh
# Schedule daily backups: 0 3 * * * /home/ocean/SN55/backup_and_recover.sh
```

---

## **FINAL NOTES**

### **Success Metrics**
- **Hour 1**: Miner online and processing requests
- **Hour 6**: First TAO rewards received
- **Day 1**: Positive daily earnings
- **Week 1**: Consistent top-50 performance
- **Month 1**: Top-20 ranking achieved

### **Maintenance Schedule**
- **Daily**: Monitor logs, check balances, verify uptime
- **Weekly**: Performance analysis, model retraining consideration
- **Monthly**: Full system backup, security audit

### **Emergency Contacts**
- **Bittensor Discord**: For network issues
- **Precog GitHub**: For miner-specific issues
- **Local Logs**: For debugging (`pm2 logs precog_miner`)

**Remember: Patience is key. Precog subnet rewards quality predictions over quantity. Focus on model accuracy and consistent performance rather than aggressive trading.**

**Good luck, and welcome to the Precog network! 🚀**
