
# Add this to your miner.py to enable domination mode

# Check for domination flag
import os
domination_mode = os.getenv('DOMINATION_MODE', 'false').lower() == 'true'

if domination_mode:
    logger.info("🏆 ACTIVATING DOMINATION MODE")
    logger.info("🎯 Target: Surpass UID 31 and become #1")

    # Import domination forward function
    from precog.miners.domination_forward import forward as domination_forward

    # Replace forward function
    original_forward = forward
    forward = domination_forward

    logger.info("✅ Domination mode activated!")
