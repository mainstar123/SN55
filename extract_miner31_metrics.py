#!/usr/bin/env python3
"""
Extract Miner 31 Key Metrics from W&B API
Manual extraction based on provided run IDs
"""

import requests
import json
import os

# W&B API key
WANDB_API_KEY = "28abf92e01954279d6c7016f62b5fe5cc7513885"

def get_run_data(run_id):
    """Get run data from W&B API"""
    url = f"https://api.wandb.ai/graphql"
    headers = {
        "Authorization": f"Bearer {WANDB_API_KEY}",
        "Content-Type": "application/json"
    }

    query = """
    query Run($project: String!, $entity: String!, $run: String!) {
      project(name: $project, entityName: $entity) {
        run(name: $run) {
          history
          summaryMetrics
          config
        }
      }
    }
    """

    variables = {
        "project": "sn55-validators",
        "entity": "yumaai",
        "run": run_id
    }

    try:
        response = requests.post(url, json={"query": query, "variables": variables}, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', {}).get('project', {}).get('run', {})
        else:
            print(f"❌ API Error for {run_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Request Error for {run_id}: {e}")
        return None

def extract_miner31_metrics(run_data, run_id):
    """Extract miner 31 specific metrics from run data"""
    if not run_data or 'history' not in run_data:
        return None

    try:
        history = json.loads(run_data['history'])
        if not isinstance(history, list):
            return None

        miner31_data = []
        for entry in history:
            miner_data = {}
            # Look for miner 31 specific keys
            for key, value in entry.items():
                if '31.' in key and value is not None:
                    miner_data[key] = value

            if miner_data:
                miner_data['_step'] = entry.get('Step', entry.get('_step'))
                miner31_data.append(miner_data)

        return {
            'run_id': run_id,
            'record_count': len(miner31_data),
            'data': miner31_data
        }

    except Exception as e:
        print(f"❌ Error processing {run_id}: {e}")
        return None

def analyze_metrics(all_metrics):
    """Analyze collected metrics"""
    print("\n📊 MINER 31 PERFORMANCE ANALYSIS"    print("=" * 60)

    total_records = 0
    all_rewards = []
    all_predictions = {'btc': [], 'eth': [], 'tao_bittensor': []}

    for metrics in all_metrics:
        if metrics and 'data' in metrics:
            run_data = metrics['data']
            total_records += len(run_data)

            for record in run_data:
                # Extract rewards
                reward_key = 'miners_info.31.miner_reward'
                if reward_key in record:
                    all_rewards.append(record[reward_key])

                # Extract predictions
                for asset in ['btc', 'eth', 'tao_bittensor']:
                    pred_key = f'miners_info.31.miner_{asset}_prediction'
                    if pred_key in record and record[pred_key] is not None:
                        all_predictions[asset].append(record[pred_key])

    print(f"📈 Total Records: {total_records}")
    print(f"📊 Reward Records: {len(all_rewards)}")

    if all_rewards:
        rewards_array = [r for r in all_rewards if r is not None and isinstance(r, (int, float))]
        if rewards_array:
            print("
💰 REWARD ANALYSIS:"            print(".6f"            print(".6f"            print(".6f"            print(f"      Count: {len(rewards_array)}")

    for asset, preds in all_predictions.items():
        if preds:
            preds_array = [p for p in preds if p is not None and isinstance(p, (int, float))]
            if preds_array:
                print("
🎯 {asset.upper()} PREDICTIONS:"                print(".4f"                print(".4f"                print(f"         Count: {len(preds_array)}")

    return {
        'total_records': total_records,
        'reward_stats': {
            'mean': np.mean(rewards_array) if rewards_array else None,
            'median': np.median(rewards_array) if rewards_array else None,
            'count': len(rewards_array) if rewards_array else 0
        } if 'rewards_array' in locals() else None
    }

def compare_with_your_model(miner31_stats):
    """Compare miner 31 with your model"""
    print("\n🏆 COMPARISON: MINER 31 vs YOUR MODEL"    print("=" * 60)

    # Load your model results
    if os.path.exists('elite_domination_results.json'):
        with open('elite_domination_results.json', 'r') as f:
            your_data = json.load(f)

        print("🎯 Your Elite Domination Model:")
        if 'model_performance' in your_data:
            perf = your_data['model_performance']
            if 'estimated_tao_per_prediction' in perf:
                your_reward = perf['estimated_tao_per_prediction']
                print(".6f")

                if miner31_stats and miner31_stats.get('reward_stats', {}).get('mean'):
                    miner31_reward = miner31_stats['reward_stats']['mean']
                    improvement = (your_reward - miner31_reward) / miner31_reward * 100
                    print(".6f"
                          f"   📈 Potential Improvement: +{improvement:.1f}%")

    print("
🎯 DOMINATION TARGETS:"    print("   • Surpass miner 31's average reward")
    print("   • Achieve MAPE < 1.0% and directional accuracy > 75%")
    print("   • Deploy elite_domination_model.pth to mainnet")
    print("   • Become #1 miner within 48 hours")

def main():
    # Miner 31's run IDs
    run_ids = ['mk3oqwr4', 'ij45zgor', 'pgqizltz', '2f04nm44']

    print("🔍 EXTRACTING MINER 31 DATA FROM W&B")
    print("=" * 60)

    all_metrics = []

    for run_id in run_ids:
        print(f"📥 Processing run: {run_id}")

        run_data = get_run_data(run_id)
        if run_data:
            metrics = extract_miner31_metrics(run_data, run_id)
            if metrics:
                all_metrics.append(metrics)
                print(f"   ✅ Extracted {metrics['record_count']} records")
            else:
                print("   ⚠️  No miner 31 data found"
        else:
            print(f"   ❌ Failed to get data")

    if all_metrics:
        stats = analyze_metrics(all_metrics)
        compare_with_your_model(stats)

        # Save raw data
        with open('miner31_raw_data.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print("
💾 Raw data saved to: miner31_raw_data.json"    else:
        print("❌ No data extracted from any runs")

if __name__ == "__main__":
    main()
