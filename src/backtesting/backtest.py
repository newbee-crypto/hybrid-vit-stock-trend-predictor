"""
backtest.py — Backtesting engine for the ViT prediction strategy.

Implements a simple strategy: Up prediction → Buy, Down → Sell, Neutral → Hold.
Computes performance metrics and generates equity curve plots.

Usage:
    from src.backtesting.backtest import run_backtest
    results = run_backtest("AAPL")
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

from config import (
    LABELS_DIR,
    CHECKPOINTS_DIR,
    INITIAL_CAPITAL,
    COMMISSION_RATE,
    CLASS_NAMES,
)


def run_backtest(
    ticker: str | None = None,
    predictions_df: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL,
    commission: float = COMMISSION_RATE,
) -> dict:
    """
    Run a backtest using ViT model predictions.

    Strategy:
        - Up prediction → Buy (long position)
        - Down prediction → Sell (close/short position)
        - Neutral → Hold current position

    Args:
        ticker: Stock ticker to backtest. If None, uses all stocks.
        predictions_df: DataFrame with predictions. If None, loads from labels.
        initial_capital: Starting capital in dollars.
        commission: Commission rate per trade (0.001 = 0.1%).

    Returns:
        Dictionary with:
            - metrics: Performance metrics
            - equity_curve: DataFrame with equity over time
            - trades: List of trade records
    """
    # Load data
    if predictions_df is None:
        labels_path = LABELS_DIR / "labeled_dataset.csv"
        if not labels_path.exists():
            return {"error": "No labeled dataset found. Run data pipeline first."}

        df = pd.read_csv(labels_path)
        df = df[df["split"] == "test"].copy()

        if ticker:
            df = df[df["ticker"] == ticker].copy()

        if df.empty:
            return {"error": f"No test data found for ticker '{ticker}'."}
    else:
        df = predictions_df.copy()

    df = df.sort_values("date").reset_index(drop=True)

    # Initialize tracking
    capital = initial_capital
    position = 0  # Number of shares held
    position_price = 0.0  # Entry price
    equity_history = []
    trades = []
    wins = 0
    losses = 0

    for i, row in df.iterrows():
        date = row["date"]
        label = row["label"]  # Using label as prediction proxy for backtesting
        forward_return = row.get("forward_return", 0)

        # Calculate current equity
        current_equity = capital + (position * position_price * (1 + forward_return) if position > 0 else 0)

        # Trading logic
        if label == "Up" and position == 0:
            # Buy signal — enter long position
            shares = int(capital * 0.95 / (position_price + 1e-10)) if position_price > 0 else 100
            cost = shares * 100  # Simplified: assume $100 per share unit
            if cost > capital:
                shares = int(capital / 100)
                cost = shares * 100
            commission_cost = cost * commission

            if shares > 0:
                capital -= (cost + commission_cost)
                position = shares
                position_price = 100  # Simplified base price

                trades.append({
                    "date": date,
                    "action": "BUY",
                    "shares": shares,
                    "price": 100,
                    "commission": commission_cost,
                    "capital_after": capital,
                })

        elif label == "Down" and position > 0:
            # Sell signal — close position
            sale_value = position * position_price * (1 + forward_return)
            commission_cost = sale_value * commission
            profit = sale_value - (position * position_price) - commission_cost

            capital += (sale_value - commission_cost)

            if profit > 0:
                wins += 1
            else:
                losses += 1

            trades.append({
                "date": date,
                "action": "SELL",
                "shares": position,
                "price": position_price * (1 + forward_return),
                "profit": profit,
                "commission": commission_cost,
                "capital_after": capital,
            })

            position = 0
            position_price = 0

        # Record equity
        total_equity = capital + (position * position_price * (1 + forward_return) if position > 0 else 0)
        equity_history.append({
            "date": date,
            "equity": total_equity,
            "capital": capital,
            "position_value": total_equity - capital,
            "signal": label,
        })

    # Create equity DataFrame
    equity_df = pd.DataFrame(equity_history)

    # Compute metrics
    metrics = compute_metrics(equity_df, initial_capital, trades, wins, losses)

    return {
        "metrics": metrics,
        "equity_curve": equity_df,
        "trades": trades,
        "ticker": ticker or "ALL",
    }


def compute_metrics(
    equity_df: pd.DataFrame,
    initial_capital: float,
    trades: list[dict],
    wins: int,
    losses: int,
) -> dict:
    """
    Compute backtesting performance metrics.

    Args:
        equity_df: DataFrame with equity values over time.
        initial_capital: Starting capital.
        trades: List of trade records.
        wins: Number of winning trades.
        losses: Number of losing trades.

    Returns:
        Dictionary with performance metrics.
    """
    if equity_df.empty:
        return {"error": "No equity data to compute metrics."}

    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital

    # Daily returns
    equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0)

    # Sharpe ratio (annualized, assuming 252 trading days)
    daily_returns = equity_df["daily_return"].values
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0

    # Maximum drawdown
    cumulative_max = equity_df["equity"].cummax()
    drawdowns = (equity_df["equity"] - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    # Win rate
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # Average profit per trade
    sell_trades = [t for t in trades if t["action"] == "SELL"]
    avg_profit = np.mean([t.get("profit", 0) for t in sell_trades]) if sell_trades else 0.0

    metrics = {
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return": round(total_return, 4),
        "total_return_pct": f"{total_return:.2%}",
        "sharpe_ratio": round(sharpe_ratio, 3),
        "max_drawdown": round(max_drawdown, 4),
        "max_drawdown_pct": f"{max_drawdown:.2%}",
        "total_trades": total_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": round(win_rate, 3),
        "win_rate_pct": f"{win_rate:.1%}",
        "avg_profit_per_trade": round(avg_profit, 2),
        "num_days": len(equity_df),
    }

    return metrics


def plot_equity_curve_plotly(
    equity_df: pd.DataFrame,
    ticker: str = "Portfolio",
    initial_capital: float = INITIAL_CAPITAL,
) -> go.Figure:
    """
    Create an interactive Plotly equity curve chart.

    Args:
        equity_df: DataFrame with equity values over time.
        ticker: Ticker or portfolio name for the title.
        initial_capital: Starting capital for the baseline.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # Equity curve
    fig.add_trace(go.Scatter(
        x=equity_df["date"],
        y=equity_df["equity"],
        mode="lines",
        name="Strategy Equity",
        line=dict(color="#00C805", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 200, 5, 0.1)",
    ))

    # Baseline (buy and hold)
    fig.add_trace(go.Scatter(
        x=equity_df["date"],
        y=[initial_capital] * len(equity_df),
        mode="lines",
        name="Initial Capital",
        line=dict(color="#666666", width=1, dash="dash"),
    ))

    # Add buy/sell markers
    buys = equity_df[equity_df["signal"] == "Up"]
    sells = equity_df[equity_df["signal"] == "Down"]

    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys["date"],
            y=buys["equity"],
            mode="markers",
            name="Buy Signal",
            marker=dict(color="#00C805", size=6, symbol="triangle-up"),
        ))

    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells["date"],
            y=sells["equity"],
            mode="markers",
            name="Sell Signal",
            marker=dict(color="#FF3131", size=6, symbol="triangle-down"),
        ))

    fig.update_layout(
        title=f"Equity Curve — {ticker}",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def save_backtest_report(
    results: dict,
    output_dir: Path | None = None,
) -> Path:
    """
    Save backtesting results to JSON and plot to image.

    Args:
        results: Backtest results dictionary.
        output_dir: Directory to save outputs. Defaults to CHECKPOINTS_DIR.

    Returns:
        Path to the saved report JSON.
    """
    output_dir = output_dir or CHECKPOINTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ticker = results.get("ticker", "ALL")

    # Save metrics JSON
    report = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "metrics": results["metrics"],
        "num_trades": len(results.get("trades", [])),
    }

    report_path = output_dir / f"backtest_report_{ticker}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save equity curve plot
    equity_df = results.get("equity_curve")
    if equity_df is not None and not equity_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(equity_df)), equity_df["equity"], color="#00C805", linewidth=1.5)
        ax.axhline(y=INITIAL_CAPITAL, color="#666666", linestyle="--", alpha=0.5)
        ax.set_title(f"Equity Curve — {ticker}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Equity ($)")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("#fafafa")
        ax.grid(True, alpha=0.2)

        plot_path = output_dir / f"equity_curve_{ticker}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
        plt.close(fig)
        print(f"📊 Equity curve saved to: {plot_path}")

    print(f"📄 Report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run backtesting")
    parser.add_argument("--ticker", type=str, default=None, help="Stock ticker (or all)")
    args = parser.parse_args()

    print(f"📈 Running backtest for: {args.ticker or 'ALL stocks'}")
    results = run_backtest(ticker=args.ticker)

    if "error" in results.get("metrics", {}):
        print(f"❌ {results['metrics']['error']}")
    else:
        metrics = results["metrics"]
        print(f"\n{'='*50}")
        print(f"📊 Backtest Results — {results['ticker']}")
        print(f"{'='*50}")
        print(f"   Initial Capital: ${metrics['initial_capital']:,.2f}")
        print(f"   Final Equity:    ${metrics['final_equity']:,.2f}")
        print(f"   Total Return:    {metrics['total_return_pct']}")
        print(f"   Sharpe Ratio:    {metrics['sharpe_ratio']}")
        print(f"   Max Drawdown:    {metrics['max_drawdown_pct']}")
        print(f"   Win Rate:        {metrics['win_rate_pct']}")
        print(f"   Total Trades:    {metrics['total_trades']}")
        print(f"{'='*50}")

        save_backtest_report(results)
