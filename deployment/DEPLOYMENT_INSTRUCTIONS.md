# 🚀 PRECOG SUBNET 55 - DEPLOYMENT INSTRUCTIONS

## **OVERVIEW**
This guide explains **exactly** which scripts to run and when to achieve the best performance on Precog subnet 55.

---

## **📋 DEPLOYMENT SCRIPTS OVERVIEW**

### **Main Deployment Script**
```bash
./deployment/deploy_best_model.sh
```
**What it does:** Deploys the domination model miner with all optimizations
**When to run:** First time deployment, or when miner stops
**Expected result:** Best model miner running with auto-improvements

### **Monitoring Script**
```bash
./deployment/monitor_precog.sh
```
**What it does:** Shows comprehensive miner status, balance, performance
**When to run:** Daily checks, after deployment, when troubleshooting
**Expected result:** Complete dashboard with all metrics

### **Retraining Scripts**
```bash
# Automated daily retraining (recommended)
./deployment/automated_retraining.sh

# Online continuous retraining (advanced)
python3 deployment/online_retrainer.py
```
**What they do:** Improve model performance with fresh data
**When to run:** Daily for automated, continuous for online
**Expected result:** Better MAPE and higher rewards

### **Backup Script**
```bash
./deployment/backup_and_recover.sh
```
**What it does:** Creates full backup of models, configs, logs
**When to run:** Before major changes, weekly
**Expected result:** Recovery point for system restoration

---

## **🎯 STEP-BY-STEP DEPLOYMENT PROCESS**

### **PHASE 1: INITIAL DEPLOYMENT (30 minutes)**

#### **Step 1: Prerequisites Check**
```bash
# Verify your setup
./deployment/verify_setup.sh
```
**Expected output:** Status of wallets, models, environment
**Fix any ❌ issues before proceeding**

#### **Step 2: Configure Environment**
```bash
# Edit your wallet names in .env.miner
nano .env.miner

# Should contain:
COLDKEY=your_coldkey_name
MINER_HOTKEY=your_hotkey_name
NETWORK=finney
# ... other settings
```

#### **Step 3: Deploy Best Model**
```bash
# Deploy the optimized domination miner
./deployment/deploy_best_model.sh
```
**What happens:**
- ✅ Stops any existing miners
- ✅ Verifies models, wallets, registration
- ✅ Starts miner with best configuration
- ✅ Enables auto-improvements

**Expected logs:**
```
✅ Miner deployment initiated (PID: XXXX)
🎉 BEST MODEL MINER DEPLOYMENT SUCCESSFUL!
```

### **PHASE 2: INITIAL MONITORING (1-2 hours)**

#### **Step 4: Monitor Startup**
```bash
# Check if miner is running properly
./deployment/monitor_precog.sh
```
**Expected results:**
- Miner status: online
- Balance: your TAO amount
- Network: connected
- Resources: normal usage

#### **Step 5: Watch Live Logs**
```bash
# Monitor first predictions
pm2 logs precog_best_miner --follow
```
**Expected first logs:**
```
🏆 ACTIVATING DOMINATION MODE
⚡ DOMINATION MODE ENABLED - Becoming #1!
✅ Domination models loaded successfully
🎯 Market Regime: [RANGING/BULL/BEAR/VOLATILE]
🎯 Prediction made: XXX.XX TAO | Confidence: X.XX
```

### **PHASE 3: PERFORMANCE OPTIMIZATION (24-48 hours)**

#### **Step 6: Enable Automated Retraining**
```bash
# Set up daily retraining (recommended)
crontab -e

# Add this line for daily retraining at 3 AM:
0 3 * * * cd /home/ocean/SN55 && ./deployment/automated_retraining.sh >> retraining.log 2>&1
```

#### **Step 7: Monitor Performance Daily**
```bash
# Daily performance check
./deployment/monitor_precog.sh
```
**Track these metrics:**
- Prediction count (should increase)
- MAPE (should decrease over time)
- Balance (should increase)
- Position (should improve)

---

## **🔄 RETRAINING STRATEGY**

### **When to Retrain?**

#### **Daily Automated Retraining (Recommended)**
```bash
./deployment/automated_retraining.sh
```
**When:** Every 24 hours
**Why:** Keeps model fresh with latest market data
**Process:**
1. Collects 7 days of fresh BTC data
2. Fine-tunes existing model (not full retrain)
3. Saves improved model
4. Restarts miner with new model

#### **Manual Retraining Triggers**
Run `./deployment/automated_retraining.sh` when:
- MAPE > 0.15% (poor performance)
- No predictions for 2+ hours
- After market regime changes
- Weekly maintenance

#### **Online Retraining (Advanced)**
```bash
# Run continuously in background
python3 deployment/online_retrainer.py &
```
**When:** For continuous improvement
**Why:** Adapts to market changes in real-time
**Process:** Updates model every 5 minutes based on recent performance

### **Retraining Indicators**

**GOOD signs (continue normal):**
- MAPE < 0.12%
- Regular predictions (every 5-15 min)
- Increasing TAO balance
- Stable subnet position

**BAD signs (retrain immediately):**
- MAPE > 0.15%
- No predictions for 1+ hours
- Decreasing performance
- Error messages in logs

---

## **📊 MONITORING SCHEDULE**

### **Every 30 minutes (first 2 hours):**
```bash
pm2 logs precog_best_miner --follow
```
Check for: First predictions, error messages, connection issues

### **Hourly (first day):**
```bash
./deployment/monitor_precog.sh
```
Check: Status, predictions, balance, resources

### **Daily:**
```bash
./deployment/monitor_precog.sh
btcli wallet overview --wallet.name YOUR_COLDKEY
```
Check: Performance trends, earnings, position

### **Weekly:**
```bash
./deployment/backup_and_recover.sh
./deployment/automated_retraining.sh
```
Backup system, force retraining

---

## **🚨 TROUBLESHOOTING SCRIPTS**

### **Miner Not Starting:**
```bash
# Check detailed status
./deployment/verify_setup.sh

# Restart deployment
./deployment/deploy_best_model.sh
```

### **No Predictions:**
```bash
# Check logs for errors
pm2 logs precog_best_miner --lines 100

# Force retraining
./deployment/automated_retraining.sh
```

### **Poor Performance:**
```bash
# Monitor resources
./deployment/monitor_precog.sh

# Retrain model
./deployment/automated_retraining.sh
```

### **System Recovery:**
```bash
# Create backup first
./deployment/backup_and_recover.sh

# Then redeploy
./deployment/deploy_best_model.sh
```

---

## **⚡ PERFORMANCE OPTIMIZATION**

### **Automatic Optimizations (Active by Default)**
- **Peak Hour Detection:** Higher frequency during UTC 9-11, 13-15
- **Market Regime Adaptation:** Adjusts to bull/bear/volatile conditions
- **Confidence Thresholding:** Dynamic based on time and market
- **Online Retraining:** Continuous model improvement

### **Manual Optimizations**
```bash
# Test peak hour optimizer
python3 deployment/peak_hour_optimizer.py

# Monitor resource usage
./deployment/monitor_precog.sh
```

---

## **🎯 EXPECTED PERFORMANCE TIMELINE**

| Time | Expected Results | Action |
|------|------------------|--------|
| 30 min | Miner online, first connections | Monitor logs |
| 1 hour | First predictions, small rewards | Check balance |
| 6 hours | Regular predictions, stable operation | Verify performance |
| 24 hours | 0.01-0.05 TAO earned, top 50-20 | Daily monitoring |
| 48 hours | 0.05-0.15 TAO earned, consistent performance | Retraining active |
| 1 week | Top 10-5 position, 0.2-0.5 TAO daily | Weekly backup |

---

## **🔧 SCRIPT REFERENCE**

### **Core Scripts (Always Use These):**
1. `./deployment/deploy_best_model.sh` - Main deployment
2. `./deployment/monitor_precog.sh` - Status monitoring
3. `./deployment/automated_retraining.sh` - Daily improvement

### **Utility Scripts (As Needed):**
- `./deployment/verify_setup.sh` - Setup validation
- `./deployment/backup_and_recover.sh` - System backup
- `python3 deployment/online_retrainer.py` - Continuous learning
- `python3 deployment/peak_hour_optimizer.py` - Peak hour testing

### **Quick Commands:**
```bash
# Start miner
./deployment/deploy_best_model.sh

# Check status
./deployment/monitor_precog.sh

# View logs
pm2 logs precog_best_miner --follow

# Retrain
./deployment/automated_retraining.sh

# Backup
./deployment/backup_and_recover.sh
```

---

## **🚨 CRITICAL NOTES**

### **DO NOT run these simultaneously:**
- Multiple miners (will conflict)
- Multiple retraining scripts (will corrupt models)

### **ALWAYS check before major changes:**
```bash
./deployment/verify_setup.sh
./deployment/backup_and_recover.sh
```

### **Monitor these closely:**
- TAO balance (should increase)
- Prediction frequency (should be regular)
- Error messages (should be none)
- Resource usage (should be <90%)

---

## **🎉 SUCCESS CHECKLIST**

- [ ] Miner deployed with `./deployment/deploy_best_model.sh`
- [ ] Status shows "online" in `./deployment/monitor_precog.sh`
- [ ] First predictions visible in logs
- [ ] Balance increasing after 1 hour
- [ ] Daily retraining scheduled
- [ ] Weekly backups active
- [ ] Performance improving over time

**Follow this guide exactly and you'll have the best-performing miner on Precog subnet 55!** 🚀
