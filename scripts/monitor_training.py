#!/usr/bin/env python3
"""
Training completion monitor with alerts
"""

import os
import time
import logging
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_training_status():
    """Check if training is complete and send alert"""
    log_file = "full_training.log"

    if not os.path.exists(log_file):
        return False, "Training not started yet"

    # Check if training is still running
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if "python scripts/train_models.py" in result.stdout:
            # Training still running - get progress
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Find current epoch and fold
            current_fold = 1
            current_epoch = 0

            for line in reversed(lines[-50:]):  # Check last 50 lines
                if "Training fold" in line:
                    try:
                        current_fold = int(line.split("fold ")[1].split("/")[0])
                    except:
                        pass
                elif "Epoch " in line and "/150" in line:
                    try:
                        current_epoch = int(line.split("Epoch ")[1].split("/")[0])
                    except:
                        pass

            progress = f"Fold {current_fold}/5, Epoch {current_epoch}/150"
            return False, f"Training in progress: {progress}"

    except Exception as e:
        logger.error(f"Error checking process: {e}")

    # Check if training completed
    with open(log_file, 'r') as f:
        content = f.read()

    if "Cross-validation results" in content:
        # Training completed!
        return True, "Training completed successfully!"

    if "ERROR" in content or "Exception" in content:
        return True, "Training encountered an error!"

    return False, "Training status unknown"

def send_alert(message):
    """Send alert notification"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create alert log
    with open("training_alerts.log", "a") as f:
        f.write(f"[{timestamp}] {message}\n")

    # Print to console with bells
    print(f"\n{'='*60}")
    print(f"🚨 TRAINING ALERT: {message}")
    print(f"{'='*60}\n")

    # Try to send system notification if available
    try:
        subprocess.run(['notify-send', 'Precog Training', message],
                      capture_output=True)
    except:
        pass  # notify-send not available

def main():
    """Monitor training completion"""
    logger.info("Starting training monitor...")

    check_interval = 300  # Check every 5 minutes
    last_status = None

    while True:
        completed, status = check_training_status()

        if completed:
            send_alert(status)
            logger.info(f"Training monitor exiting: {status}")
            break

        if status != last_status:
            logger.info(f"Training status: {status}")
            last_status = status

        time.sleep(check_interval)

if __name__ == "__main__":
    main()

