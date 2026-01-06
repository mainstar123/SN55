#!/bin/bash
# AUTOMATED MODEL RETRAINING SCRIPT
# Run this daily to improve model performance
# Usage: ./deployment/automated_retraining.sh

cd /home/ocean/SN55
source venv/bin/activate

echo "=== AUTOMATED MODEL RETRAINING ==="
echo "Time: $(date)"
echo "Working directory: $(pwd)"

# Check if miner is running
MINER_RUNNING=$(pm2 list | grep precog_miner | grep online | wc -l)
if [ $MINER_RUNNING -eq 0 ]; then
    echo "❌ Miner not running - cannot check performance"
    exit 1
fi

# Check miner performance from logs
echo "Checking miner performance..."
PERFORMANCE=$(pm2 logs precog_miner --lines 1000 2>/dev/null | grep "MAPE" | tail -1 | awk '{print $2}' | sed 's/%//')

if [ -z "$PERFORMANCE" ]; then
    echo "ℹ️ No MAPE data found in recent logs - checking prediction count instead"
    PREDICTIONS=$(pm2 logs precog_miner --lines 1000 2>/dev/null | grep "Prediction made" | wc -l)
    echo "Predictions in last 1000 log lines: $PREDICTIONS"

    if [ $PREDICTIONS -lt 50 ]; then
        echo "⚠️ Low prediction activity - skipping retraining"
        exit 0
    fi
else
    echo "Current MAPE: ${PERFORMANCE}%"

    # Retrain if performance is poor (>0.15%)
    if (( $(echo "$PERFORMANCE > 0.15" | bc -l) )); then
        echo "❌ Performance degraded (MAPE: ${PERFORMANCE}%) - initiating retraining..."
    else
        echo "✅ Performance good (MAPE: ${PERFORMANCE}%) - no retraining needed"
        exit 0
    fi
fi

# Collect fresh training data
echo "📊 Collecting fresh training data..."
python3 scripts/fetch_training_data.py --days 7 --output fresh_training_data.csv

if [ $? -ne 0 ]; then
    echo "❌ Failed to fetch training data"
    exit 1
fi

# Backup current model
echo "💾 Backing up current model..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p models/backups
cp models/domination_model_trained.pth models/backups/model_backup_$TIMESTAMP.pth
cp models/feature_scaler.pkl models/backups/scaler_backup_$TIMESTAMP.pkl

echo "🔄 Retraining model with fresh data..."

# Run retraining
python3 -c "
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib

print('Loading data...')
data = pd.read_csv('fresh_training_data.csv')

# Prepare features and target
feature_cols = [col for col in data.columns if col != 'target']
X = data[feature_cols]
y = data['target']

print(f'Dataset shape: {X.shape}')

# Load existing scaler or create new one
try:
    scaler = joblib.load('models/feature_scaler.pkl')
    X_scaled = scaler.transform(X)
    print('Using existing scaler')
except:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    print('Created new scaler')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print('Loading existing model...')
try:
    model = torch.load('models/domination_model_trained.pth')
    print('Loaded existing model')
except:
    print('❌ Could not load existing model')
    exit(1)

# Fine-tune model with lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

print('Fine-tuning model...')
model.train()
best_loss = float('inf')

for epoch in range(25):  # Reduced epochs for safety
    optimizer.zero_grad()
    outputs = model(torch.FloatTensor(X_train))
    loss = criterion(outputs.squeeze(), torch.FloatTensor(y_train.values))
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(torch.FloatTensor(X_test))
        val_loss = criterion(val_outputs.squeeze(), torch.FloatTensor(y_test.values))
    model.train()

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model, 'models/retrained_model_temp.pth')

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch+1}/25 - Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

# Load best model
model = torch.load('models/retrained_model_temp.pth')

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_pred = model(torch.FloatTensor(X_test)).squeeze().numpy()
    mape = mean_absolute_percentage_error(y_test, test_pred) * 100

print(f'Final test MAPE: {mape:.4f}%')
print(f'Improvement: {float(PERFORMANCE) - mape:.4f}%' if 'PERFORMANCE' in locals() else 'First training')

# Save retrained model
torch.save(model, 'models/domination_model_trained.pth')
joblib.dump(scaler, 'models/feature_scaler.pkl')

print('✅ Model retrained successfully!')
"

if [ $? -eq 0 ]; then
    echo "✅ Retraining completed successfully"

    # Restart miner with new model
    echo "🔄 Restarting miner with updated model..."
    pm2 restart precog_miner

    # Wait and verify
    sleep 10
    if pm2 list | grep -q "precog_miner.*online"; then
        echo "✅ Miner restarted successfully with new model"
        echo "📊 Monitor performance with: pm2 logs precog_miner --follow"
    else
        echo "❌ Miner failed to restart - check logs"
        pm2 logs precog_miner --lines 20
    fi

else
    echo "❌ Retraining failed"
    # Restore backup
    cp models/backups/model_backup_$TIMESTAMP.pth models/domination_model_trained.pth
    cp models/backups/scaler_backup_$TIMESTAMP.pkl models/feature_scaler.pkl
    echo "Restored backup model"
fi

echo "=== RETRAINING COMPLETE ==="
