# Shadow Evaluation Harness for Bittensor Subnet 55

This system compares your forecasting model against the current top miner on Precog Subnet 55 before you register on-chain.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   export COINMETRICS_API_KEY="your_coinmetrics_api_key_here"
   ```

   Get your API key from: https://docs.coinmetrics.io/api/v4/

3. **Implement your model:**
   Edit `main.py` and replace the `example_model_predict` function with your actual forecasting model.

## Usage

### Continuous Evaluation
```bash
python main.py --mode continuous
```

This will:
- Collect data every 5 minutes
- Resolve ground truth hourly
- Print daily evaluation summaries

### Single Evaluation Cycle
```bash
python main.py --mode once
```

### Custom Intervals
```bash
# Collect every 10 minutes, evaluate every 12 hours
python main.py --mode continuous --collection-interval 10 --evaluation-interval 12
```

## Architecture

- **collector.py**: Handles periodic data collection from APIs and your model
- **ground_truth.py**: Resolves actual price outcomes after 1 hour
- **evaluator.py**: Computes validator-like metrics and rolling comparisons
- **main.py**: Orchestrates the evaluation system

## Data Format

The system maintains a CSV file (`forecast_data.csv`) with columns:
- `timestamp`: When forecast was made
- `spot_price`: BTC price at forecast time
- `top_point/low/high`: Top miner predictions
- `my_point/low/high`: Your model predictions
- `actual_price_1h`: Actual price 1 hour later
- `actual_low/high`: Min/max price during the hour
- `resolved_at`: When ground truth was resolved

## Metrics

The system evaluates using validator-like metrics:

- **MAE**: Mean Absolute Error on point forecasts
- **Interval Hit Rate**: % of times prediction interval contained actual range
- **Combined Score**: Weighted combination of accuracy, coverage, and width
- **Verdict**: "LIKELY BETTER", "UNCLEAR", or "WORSE" vs top miner

## Important Notes

- The Precog API endpoint format may need adjustment based on actual API response
- Ensure your model function returns the expected format: `{'point': float, 'low': float, 'high': float}`
- The system requires at least 10 resolved forecasts for confident verdicts
- All timestamps are in UTC
