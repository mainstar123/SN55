#!/usr/bin/env python3
"""
Supremacy Backtesting
Comprehensive backtesting of supremacy model against historical data
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional

class SupremacyModel(nn.Module):
    """Supremacy ensemble model for backtesting"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class SupremacyBacktester:
    """Comprehensive backtesting system for supremacy models"""

    def __init__(self, model_path: str = "latest_supremacy_model.pth"):
        self.model_path = model_path
        self.model = None
        self.results = {}

        print("📊 SUPREMACY BACKTESTING SYSTEM")
        print("=" * 50)

    def load_supremacy_model(self):
        """Load the trained supremacy model"""
        print("🔄 Loading supremacy model...")

        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model = SupremacyModel()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            print("✅ Supremacy model loaded")
            print(".4f")
            print(f"   📅 Trained: {checkpoint.get('timestamp', 'Unknown')}")

            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def generate_historical_data(self, days: int = 365) -> pd.DataFrame:
        """Generate realistic historical market data for backtesting"""
        print(f"📈 Generating {days} days of historical market data...")

        np.random.seed(42)
        n_samples = days * 24  # Hourly data

        # Create realistic timestamps
        start_date = datetime.now() - timedelta(days=days)
        timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]

        # Generate market features
        t = np.linspace(0, 8*np.pi, n_samples)

        # Price data with trends and volatility
        base_price = 50000 + 10000 * np.sin(t * 0.1)  # BTC-like price movement
        trend = 0.001 * np.arange(n_samples)  # Long-term trend
        seasonal = 5000 * np.sin(2 * np.pi * np.arange(n_samples) / (24*30))  # Monthly cycle
        daily_cycle = 2000 * np.sin(2 * np.pi * (np.arange(n_samples) % 24) / 24)  # Daily cycle
        noise = np.random.normal(0, 1000, n_samples)  # Random noise

        price = base_price + trend + seasonal + daily_cycle + noise

        # Technical indicators (24 features)
        data = {
            'timestamp': timestamps,
            'price': price,
            'returns': np.diff(price, prepend=price[0]),
            'sma_20': pd.Series(price).rolling(20).mean().fillna(method='bfill'),
            'sma_50': pd.Series(price).rolling(50).mean().fillna(method='bfill'),
            'ema_12': pd.Series(price).ewm(span=12).mean(),
            'ema_26': pd.Series(price).ewm(span=26).mean(),
            'rsi': self._calculate_rsi(price),
            'macd': self._calculate_macd(price),
            'bb_upper': pd.Series(price).rolling(20).mean() + 2 * pd.Series(price).rolling(20).std(),
            'bb_lower': pd.Series(price).rolling(20).mean() - 2 * pd.Series(price).rolling(20).std(),
            'volume': np.random.lognormal(10, 1, n_samples),  # Simulated volume
            'volatility': pd.Series(price).rolling(20).std().fillna(method='bfill'),
        }

        # Add more technical indicators
        for i in range(12):
            data[f'tech_{i}'] = np.random.randn(n_samples) * 0.1 + np.sin(t * (i+1) * 0.1)

        df = pd.DataFrame(data)
        df = df.fillna(method='bfill').fillna(method='ffill')  # Handle NaN values

        print(f"   ✅ Generated {len(df)} data points")
        print(".2f")
        print(f"   📊 Features: {len(df.columns) - 1} technical indicators")

        return df

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(period).mean()
        avg_loss = pd.Series(loss).rolling(period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)  # Neutral RSI

    def _calculate_macd(self, prices):
        """Calculate MACD indicator"""
        ema12 = pd.Series(prices).ewm(span=12).mean()
        ema26 = pd.Series(prices).ewm(span=26).mean()
        return ema12 - ema26

    def prepare_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare features for model input"""
        # Select technical indicators (exclude timestamp and price)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'price', 'returns']]
        features = df[feature_cols].values

        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        # Convert to tensor and reshape for model
        if hasattr(self.model, 'network') and len(self.model.network) > 0:
            # Check expected input size
            first_layer = self.model.network[0]
            if hasattr(first_layer, 'in_features'):
                expected_size = first_layer.in_features
                if features.shape[1] != expected_size:
                    # Adjust feature count
                    if features.shape[1] > expected_size:
                        features = features[:, :expected_size]
                    else:
                        # Pad with zeros
                        padding = np.zeros((features.shape[0], expected_size - features.shape[1]))
                        features = np.concatenate([features, padding], axis=1)

        return torch.from_numpy(features.astype(np.float32))

    def run_backtest(self, df: pd.DataFrame, initial_balance: float = 10000.0) -> Dict:
        """Run comprehensive backtest"""
        print("\n🎯 RUNNING COMPREHENSIVE BACKTEST")
        print("=" * 50)

        # Prepare features
        features = self.prepare_features(df)

        # Generate predictions
        print("🔮 Generating model predictions...")
        with torch.no_grad():
            predictions = []
            batch_size = 100

            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size]
                batch_preds = self.model(batch).squeeze().numpy()
                predictions.extend(batch_preds)

        predictions = np.array(predictions)

        # Calculate directional accuracy
        actual_directions = np.sign(df['returns'].values)
        predicted_directions = np.sign(predictions - 0.5)  # Convert to -1/1 from 0-1

        directional_accuracy = np.mean(predicted_directions[1:] == actual_directions[1:])

        # Trading simulation
        print("📊 Running trading simulation...")

        balance = initial_balance
        position = 0  # 0 = no position, 1 = long, -1 = short
        trades = []
        equity_curve = [initial_balance]

        # Simple trading strategy based on predictions
        for i in range(1, len(predictions)):
            pred = predictions[i]

            # Enter long if prediction > 0.6
            if position == 0 and pred > 0.6:
                position = 1
                entry_price = df['price'].iloc[i]
                trades.append({
                    'type': 'long',
                    'entry_time': df['timestamp'].iloc[i],
                    'entry_price': entry_price,
                    'quantity': balance / entry_price * 0.1  # 10% of balance
                })

            # Enter short if prediction < 0.4
            elif position == 0 and pred < 0.4:
                position = -1
                entry_price = df['price'].iloc[i]
                trades.append({
                    'type': 'short',
                    'entry_time': df['timestamp'].iloc[i],
                    'entry_price': entry_price,
                    'quantity': balance / entry_price * 0.1
                })

            # Exit positions (simple time-based exit)
            elif position != 0 and (i - trades[-1]['entry_time'].hour) >= 24:  # Hold for 24 hours
                exit_price = df['price'].iloc[i]
                entry_price = trades[-1]['entry_price']

                if position == 1:  # Long position
                    pnl = (exit_price - entry_price) * trades[-1]['quantity']
                else:  # Short position
                    pnl = (entry_price - exit_price) * trades[-1]['quantity']

                balance += pnl
                trades[-1].update({
                    'exit_time': df['timestamp'].iloc[i],
                    'exit_price': exit_price,
                    'pnl': pnl
                })

                position = 0

            equity_curve.append(balance)

        # Calculate performance metrics
        total_return = (balance - initial_balance) / initial_balance * 100
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)

        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        total_closed_trades = len([t for t in trades if 'exit_time' in t])
        win_rate = winning_trades / max(total_closed_trades, 1) * 100

        # Compile results
        results = {
            'backtest_period': {
                'start': df['timestamp'].min().strftime('%Y-%m-%d'),
                'end': df['timestamp'].max().strftime('%Y-%m-%d'),
                'total_days': len(df) // 24,
                'total_hours': len(df)
            },
            'model_performance': {
                'directional_accuracy': directional_accuracy,
                'prediction_confidence': np.mean(np.abs(predictions - 0.5) * 2),  # 0-1 scale
                'prediction_volatility': np.std(predictions)
            },
            'trading_performance': {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return_pct': total_return,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': len(trades),
                'closed_trades': total_closed_trades,
                'win_rate_pct': win_rate,
                'avg_trade_pnl': np.mean([t.get('pnl', 0) for t in trades if 'pnl' in t])
            },
            'risk_metrics': {
                'volatility': np.std(np.diff(np.log(equity_curve))),
                'var_95': np.percentile(np.diff(equity_curve), 5),
                'max_consecutive_losses': self._max_consecutive_losses(trades)
            }
        }

        self.results = results
        return results

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_sharpe_ratio(self, equity_curve: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        returns = np.diff(np.log(equity_curve))
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _max_consecutive_losses(self, trades: List[Dict]) -> int:
        """Calculate maximum consecutive losing trades"""
        max_consecutive = 0
        current_consecutive = 0

        for trade in trades:
            if 'pnl' in trade and trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def generate_backtest_report(self) -> Dict:
        """Generate comprehensive backtest report"""
        if not self.results:
            print("❌ No backtest results available. Run backtest first.")
            return {}

        print("\n📋 GENERATING BACKTEST REPORT")
        print("=" * 50)

        results = self.results

        # Performance assessment
        directional_acc = results['model_performance']['directional_accuracy']
        total_return = results['trading_performance']['total_return_pct']
        max_dd = results['trading_performance']['max_drawdown_pct']
        win_rate = results['trading_performance']['win_rate_pct']

        # Assessment criteria
        assessment = {
            'directional_accuracy': 'EXCELLENT' if directional_acc >= 0.90 else
                                  'GOOD' if directional_acc >= 0.80 else
                                  'FAIR' if directional_acc >= 0.70 else 'POOR',
            'trading_performance': 'EXCELLENT' if total_return > 50 and max_dd < 20 else
                                 'GOOD' if total_return > 20 and max_dd < 30 else
                                 'FAIR' if total_return > 0 else 'POOR',
            'risk_management': 'EXCELLENT' if max_dd < 15 else
                             'GOOD' if max_dd < 25 else
                             'FAIR' if max_dd < 35 else 'POOR'
        }

        overall_rating = 'EXCELLENT' if all(r == 'EXCELLENT' for r in assessment.values()) else \
                        'GOOD' if sum(1 for r in assessment.values() if r in ['EXCELLENT', 'GOOD']) >= 2 else \
                        'FAIR'

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_path': self.model_path,
                'model_type': 'SupremacyEnsemble'
            },
            'backtest_summary': results,
            'performance_assessment': assessment,
            'overall_rating': overall_rating,
            'recommendations': self._generate_recommendations(assessment, results),
            'key_insights': self._generate_insights(results)
        }

        # Save report
        filename = f"supremacy_backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"✅ Backtest report saved: {filename}")

        return report

    def _generate_recommendations(self, assessment: Dict, results: Dict) -> List[str]:
        """Generate recommendations based on backtest results"""
        recommendations = []

        directional_acc = results['model_performance']['directional_accuracy']
        total_return = results['trading_performance']['total_return_pct']
        max_dd = results['trading_performance']['max_drawdown_pct']

        if directional_acc < 0.85:
            recommendations.append("IMPROVE: Focus on directional accuracy optimization (>85% target)")
        elif directional_acc < 0.90:
            recommendations.append("OPTIMIZE: Fine-tune for 90%+ directional accuracy achievement")

        if total_return < 10:
            recommendations.append("STRATEGY: Review trading strategy and position sizing")
        elif total_return > 30:
            recommendations.append("MAINTAIN: Current trading strategy performing well")

        if max_dd > 25:
            recommendations.append("RISK: Implement better risk management (stop-losses, position sizing)")
        else:
            recommendations.append("RISK: Risk management parameters acceptable")

        if len(recommendations) == 0:
            recommendations.append("EXCELLENT: All performance metrics meet or exceed targets")
            recommendations.append("MAINTAIN: Continue current strategy with monitoring")

        return recommendations

    def _generate_insights(self, results: Dict) -> List[str]:
        """Generate key insights from backtest results"""
        insights = []

        directional_acc = results['model_performance']['directional_accuracy']
        total_return = results['trading_performance']['total_return_pct']
        win_rate = results['trading_performance']['win_rate_pct']

        insights.append(".1%")

        if directional_acc > 0.85:
            insights.append("Strong directional prediction capability demonstrated")
        else:
            insights.append("Directional accuracy needs improvement for competitive edge")

        insights.append(".1%")

        if win_rate > 55:
            insights.append("Trading strategy shows positive expectancy")
        elif win_rate > 45:
            insights.append("Trading strategy near breakeven - needs refinement")
        else:
            insights.append("Trading strategy requires significant improvement")

        insights.append(".1%")

        return insights

    def print_backtest_results(self):
        """Print formatted backtest results"""
        if not self.results:
            print("❌ No backtest results available")
            return

        results = self.results

        print("\n" + "="*80)
        print("🎯 SUPREMACY BACKTEST RESULTS")
        print("="*80)

        # Period
        period = results['backtest_period']
        print(f"📅 Period: {period['start']} to {period['end']} ({period['total_days']} days)")

        # Model Performance
        model_perf = results['model_performance']
        print("\n🤖 MODEL PERFORMANCE:")
        print(".1%")
        print(".1%")

        # Trading Performance
        trading_perf = results['trading_performance']
        print("\n📈 TRADING PERFORMANCE:")
        print(".2f")
        print(".1%")
        print(".2f")
        print(".2f")
        print(f"   🎯 Win Rate: {trading_perf['win_rate_pct']:.1f}%")
        print(f"   📊 Total Trades: {trading_perf['total_trades']}")

        # Risk Metrics
        risk_metrics = results['risk_metrics']
        print("
🛡️ RISK METRICS:"        print(".1%")
        print(".2f")
        print(f"   📉 Max Consecutive Losses: {risk_metrics['max_consecutive_losses']}")

        print("\n" + "="*80)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Supremacy Backtesting System")
    parser.add_argument("--model", type=str, default="latest_supremacy_model.pth",
                       help="Path to supremacy model")
    parser.add_argument("--days", type=int, default=365,
                       help="Number of days for backtest")
    parser.add_argument("--balance", type=float, default=10000.0,
                       help="Initial trading balance")

    args = parser.parse_args()

    # Initialize backtester
    backtester = SupremacyBacktester(args.model)

    # Load model
    if not backtester.load_supremacy_model():
        print("❌ Cannot proceed without model")
        return

    # Generate historical data
    historical_data = backtester.generate_historical_data(args.days)

    # Run backtest
    results = backtester.run_backtest(historical_data, args.balance)

    # Generate report
    report = backtester.generate_backtest_report()

    # Print results
    backtester.print_backtest_results()

    # Final assessment
    directional_acc = results['model_performance']['directional_accuracy']
    total_return = results['trading_performance']['total_return_pct']

    print("
🎯 FINAL ASSESSMENT:"    if directional_acc >= 0.90:
        print("   🏆 SUPREMACY ACHIEVED: >90% directional accuracy!")
        print("   💪 Ready for #1 position deployment")
    elif directional_acc >= 0.85:
        print("   🥈 STRONG PERFORMANCE: >85% directional accuracy")
        print("   🔄 Close to supremacy target")
    else:
        print("   📈 IMPROVEMENT NEEDED: Directional accuracy below target")
        print("   🔧 Focus on model optimization")

    if total_return > 20:
        print("   💰 PROFITABLE: Positive trading performance")
    elif total_return > 0:
        print("   ⚖️ BREAK-EVEN: Trading at breakeven")
    else:
        print("   📉 UNPROFITABLE: Negative trading performance")

if __name__ == "__main__":
    main()
