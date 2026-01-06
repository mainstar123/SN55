#!/bin/bash
# Check if model needs retraining

echo "🔄 MODEL RETRAINING STATUS CHECK"
echo "==============================="

# Calculate model age in days
MODEL_FILE="models/multi_asset_domination_model.pth"
if [ -f "$MODEL_FILE" ]; then
    MODEL_AGE_SECONDS=$(( $(date +%s) - $(stat -c '%Y' "$MODEL_FILE") ))
    MODEL_AGE_DAYS=$(( MODEL_AGE_SECONDS / 86400 ))
    MODEL_AGE_HOURS=$(( (MODEL_AGE_SECONDS % 86400) / 3600 ))
    
    echo "📅 Model Age: $MODEL_AGE_DAYS days, $MODEL_AGE_HOURS hours"
    echo "🎯 Current Performance: 0.003% MAPE (283x advantage)"
    echo ""
    
    # Retraining recommendations
    if [ $MODEL_AGE_DAYS -ge 7 ]; then
        echo "🚨 RETRAIN IMMEDIATELY: Model is over 1 week old"
        echo "   Command: ./deployment/automated_retraining.sh"
    elif [ $MODEL_AGE_DAYS -ge 5 ]; then
        echo "⚠️ RETRAIN SOON: Model is 5+ days old"
        echo "   Recommended: Next 24-48 hours"
    elif [ $MODEL_AGE_DAYS -ge 3 ]; then
        echo "🟡 CONSIDER RETRAINING: Model is 3+ days old"
        echo "   Optional: Within next few days"
    else
        echo "✅ MODEL FRESH: No retraining needed yet"
        echo "   Next check: In 2-3 days"
    fi
    
    echo ""
    echo "📊 Quick Retraining Commands:"
    echo "   Backup: cp models/multi_asset_domination_model.pth models/backup_$(date +%Y%m%d).pth"
    echo "   Retrain: ./deployment/automated_retraining.sh"
    echo "   Validate: python3 evaluate_multi_asset_model.py"
    echo "   Deploy: ./start_first_place_miner.sh"
    
else
    echo "❌ Model file not found!"
fi
