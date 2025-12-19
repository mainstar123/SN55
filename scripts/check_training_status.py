#!/usr/bin/env python3
"""
Quick training status checker
"""

import os
import subprocess
from datetime import datetime

def get_training_status():
    """Get comprehensive training status"""

    print("🔍 PRECOG TRAINING STATUS CHECK")
    print("=" * 50)

    # Check if training is running
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
        training_running = "python scripts/train_models.py" in result.stdout
        monitor_running = "python scripts/monitor_training.py" in result.stdout

        print(f"📊 Training Process: {'✅ RUNNING' if training_running else '❌ NOT RUNNING'}")
        print(f"👀 Monitor Process: {'✅ RUNNING' if monitor_running else '❌ NOT RUNNING'}")
        print()

    except Exception as e:
        print(f"❌ Error checking processes: {e}")
        return

    # Check log files
    log_files = {
        "Training Log": "full_training.log",
        "Monitor Log": "monitor.log",
        "Training Alerts": "training_alerts.log"
    }

    for name, filename in log_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"📄 {name}: {size:,} bytes")
        else:
            print(f"📄 {name}: Not found")

    print()

    # Check training progress
    if os.path.exists("full_training.log"):
        try:
            with open("full_training.log", "r") as f:
                lines = f.readlines()

            # Find current fold and epoch
            current_fold = 1
            current_epoch = 0
            total_epochs = 150

            for line in reversed(lines[-100:]):  # Check last 100 lines
                if "Training fold" in line and "/5" in line:
                    try:
                        current_fold = int(line.split("fold ")[1].split("/")[0])
                    except:
                        pass
                elif "Epoch " in line and "/150" in line:
                    try:
                        current_epoch = int(line.split("Epoch ")[1].split("/")[0])
                    except:
                        pass

            # Calculate progress
            folds_completed = current_fold - 1
            current_fold_progress = current_epoch / total_epochs
            total_progress = (folds_completed + current_fold_progress) / 5

            print("📈 TRAINING PROGRESS:")
            print(".1f")
            print(".1f")
            print(".1f")
            print()

            # Show recent performance
            print("🎯 RECENT PERFORMANCE:")
            for line in reversed(lines[-20:]):
                if "MAE:" in line and "DirAcc:" in line:
                    print(f"   {line.strip()}")
                    break

        except Exception as e:
            print(f"❌ Error reading training log: {e}")

    # Check model files
    model_dir = "models"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        print(f"🤖 Saved Models: {len(model_files)} files")
        for model in sorted(model_files):
            mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(model_dir, model)))
            print(f"   📁 {model} (modified: {mtime.strftime('%H:%M')})")

    print()
    print("💡 COMMANDS:")
    print("   • View logs: tail -f full_training.log")
    print("   • Stop training: pkill -f 'python scripts/train_models.py'")
    print("   • Check alerts: tail -f training_alerts.log")

if __name__ == "__main__":
    get_training_status()
