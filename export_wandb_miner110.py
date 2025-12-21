import wandb
import pandas as pd
import os

# Set your API key (do not share publicly)
os.environ["WANDB_API_KEY"] = "28abf92e01954279d6c7016f62b5fe5cc7513885"

# Connect to wandb and fetch the run
target_run_path = "/yumaai/sn55-validators/runs/11t39nhr"  # Adjust if needed
api = wandb.Api()
run = api.run(target_run_path)

# Download full history as DataFrame - NO 'sample_rate' argument!
df = run.history()  # Get all available history by default

# Columns from UID 110
columns_needed = [
    "miners_info.110.miner_tao_bittensor_prediction",
    "miners_info.110.miner_reward",
    "miners_info.110.miner_moving_average",
    "miners_info.110.miner_eth_prediction",
    "miners_info.110.miner_btc_prediction",
    "Step"
]
filtered_cols = [col for col in columns_needed if col in df.columns]
miner110_df = df[filtered_cols]

# Export to CSV
csv_path = "wandb_miner110.csv"
miner110_df.to_csv(csv_path, index=False)
print(f"Exported data to {csv_path}\n\nLatest entries:")
print(miner110_df.tail())
