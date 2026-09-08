"""
3_Backtest.py — Backtesting Page.

Allows users to select a stock, run backtesting, view performance metrics,
and see interactive equity curve charts.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import streamlit as st

from config import STOCK_TICKERS, INITIAL_CAPITAL
from utils.ui_helpers import (
    inject_custom_css,
    metric_card,
    section_header,
    custom_divider,
)

inject_custom_css()

st.markdown("# 📊 Backtesting")
st.markdown("Test the hybrid model prediction strategy against historical data.")

st.markdown(custom_divider(), unsafe_allow_html=True)

# ━━━ Configuration ━━━
col1, col2 = st.columns([1, 1])

with col1:
    ticker = st.selectbox(
        "Select Stock",
        ["ALL"] + STOCK_TICKERS,
        index=0,
        key="backtest_ticker",
    )

with col2:
    capital = st.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=10_000_000,
        value=int(INITIAL_CAPITAL),
        step=10000,
        key="backtest_capital",
    )

run_backtest_btn = st.button(
    "🚀 Run Backtest",
    type="primary",
    use_container_width=True,
    key="run_backtest",
)

if run_backtest_btn:
    st.markdown(custom_divider(), unsafe_allow_html=True)

    with st.spinner("📈 Running backtest..."):
        try:
            from src.backtesting.backtest import run_backtest, plot_equity_curve_plotly

            ticker_arg = None if ticker == "ALL" else ticker
            results = run_backtest(
                ticker=ticker_arg,
                initial_capital=float(capital),
            )

            if "error" in results.get("metrics", {}):
                st.error(f"❌ {results['metrics']['error']}")
                st.info("Make sure you've run the data pipeline first:\n"
                       "1. `python src/data/collect_data.py`\n"
                       "2. `python src/data/generate_charts.py`\n"
                       "3. `python src/data/label_data.py`")
                st.stop()

            metrics = results["metrics"]

            # ━━━ Performance Metrics ━━━
            st.markdown(section_header("📊 Performance Metrics"), unsafe_allow_html=True)

            m_cols = st.columns(4)
            with m_cols[0]:
                return_type = "positive" if metrics["total_return"] > 0 else "negative"
                st.markdown(
                    metric_card(
                        "Total Return",
                        metrics["total_return_pct"],
                        f"${metrics['final_equity'] - metrics['initial_capital']:+,.2f}",
                        return_type,
                    ),
                    unsafe_allow_html=True,
                )

            with m_cols[1]:
                wr_type = "positive" if metrics["win_rate"] > 0.5 else "negative" if metrics["win_rate"] < 0.4 else "neutral"
                st.markdown(
                    metric_card(
                        "Win Rate",
                        metrics["win_rate_pct"],
                        f"{metrics['winning_trades']}W / {metrics['losing_trades']}L",
                        wr_type,
                    ),
                    unsafe_allow_html=True,
                )

            with m_cols[2]:
                sharpe_type = "positive" if metrics["sharpe_ratio"] > 1 else "negative" if metrics["sharpe_ratio"] < 0 else "neutral"
                st.markdown(
                    metric_card(
                        "Sharpe Ratio",
                        f"{metrics['sharpe_ratio']:.3f}",
                        "Risk-adjusted return",
                        sharpe_type,
                    ),
                    unsafe_allow_html=True,
                )

            with m_cols[3]:
                dd_type = "positive" if metrics["max_drawdown"] > -0.1 else "negative"
                st.markdown(
                    metric_card(
                        "Max Drawdown",
                        metrics["max_drawdown_pct"],
                        "Peak-to-trough decline",
                        dd_type,
                    ),
                    unsafe_allow_html=True,
                )

            # Second row of metrics
            m_cols2 = st.columns(4)
            with m_cols2[0]:
                st.markdown(
                    metric_card("Initial Capital", f"${metrics['initial_capital']:,.2f}"),
                    unsafe_allow_html=True,
                )
            with m_cols2[1]:
                eq_type = "positive" if metrics["final_equity"] > metrics["initial_capital"] else "negative"
                st.markdown(
                    metric_card("Final Equity", f"${metrics['final_equity']:,.2f}", delta_type=eq_type),
                    unsafe_allow_html=True,
                )
            with m_cols2[2]:
                st.markdown(
                    metric_card("Total Trades", str(metrics["total_trades"])),
                    unsafe_allow_html=True,
                )
            with m_cols2[3]:
                st.markdown(
                    metric_card("Avg Profit/Trade", f"${metrics['avg_profit_per_trade']:,.2f}"),
                    unsafe_allow_html=True,
                )

            # ━━━ Equity Curve ━━━
            st.markdown(custom_divider(), unsafe_allow_html=True)
            st.markdown(section_header("📈 Equity Curve"), unsafe_allow_html=True)

            equity_df = results.get("equity_curve")
            if equity_df is not None and not equity_df.empty:
                fig = plot_equity_curve_plotly(
                    equity_df,
                    ticker=results["ticker"],
                    initial_capital=float(capital),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No equity curve data available.")

            # ━━━ Trade Log ━━━
            st.markdown(custom_divider(), unsafe_allow_html=True)
            st.markdown(section_header("📋 Trade Log"), unsafe_allow_html=True)

            trades = results.get("trades", [])
            if trades:
                import pandas as pd
                trades_df = pd.DataFrame(trades)
                trades_display = trades_df[["date", "action", "shares", "capital_after"]].copy()
                trades_display.columns = ["Date", "Action", "Shares", "Capital After"]
                trades_display["Capital After"] = trades_display["Capital After"].apply(lambda x: f"${x:,.2f}")

                st.dataframe(
                    trades_display,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No trades were executed during the backtest period.")

        except Exception as e:
            st.error(f"❌ Backtest error: {str(e)}")
            st.exception(e)

else:
    # Show placeholder
    st.markdown("""
    <div class="prediction-banner">
        <h2>Configure and Run Backtest</h2>
        <div style="color: #8b8fa3; font-size: 1rem; max-width: 500px; margin: 0 auto;">
            Select a stock (or ALL), set your initial capital, and click
            <strong>Run Backtest</strong> to see how the ViT prediction
            strategy would have performed.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Strategy Rules:**
    - 📈 **Up prediction** → Buy (enter long position)
    - 📉 **Down prediction** → Sell (close position)
    - ➡️ **Neutral** → Hold (no action)

    **Metrics Computed:**
    - Total Return & Final Equity
    - Win Rate (winning vs losing trades)
    - Sharpe Ratio (risk-adjusted return)
    - Maximum Drawdown (worst peak-to-trough decline)
    """)
