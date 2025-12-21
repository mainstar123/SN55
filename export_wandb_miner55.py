import wandb
import pandas as pd
import os

# Set your wandb API key
os.environ["WANDB_API_KEY"] = "28abf92e01954279d6c7016f62b5fe5cc7513885"

# Connect and get the run object
api = wandb.Api()
run = api.run("/yumaai/sn55-validators/runs/11t39nhr")  # Update if needed

# Download full history
df = run.history()

# Relevant columns for miner 55
columns_needed = [
    "miners_info.55.miner_tao_bittensor_prediction",
    "miners_info.55.miner_reward",
    "miners_info.55.miner_moving_average",
    "miners_info.55.miner_eth_prediction",
    "miners_info.55.miner_btc_prediction",
    "Step"
]
filtered_cols = [col for col in columns_needed if col in df.columns]
miner55_df = df[filtered_cols]

# Export
csv_path = "wandb_miner55.csv"
miner55_df.to_csv(csv_path, index=False)
print(f"Exported data to {csv_path}\n\nLatest entries:")
print(miner55_df.tail())

