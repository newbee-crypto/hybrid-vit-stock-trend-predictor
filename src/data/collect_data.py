"""
collect_data.py — Download OHLCV stock data from Yahoo Finance.

Downloads historical daily data for all configured stock tickers
and saves each as a CSV file in data/raw/.

Usage:
    python -m src.data.collect_data
    python src/data/collect_data.py
"""

import sys
from pathlib import Path

# Allow running as standalone script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yfinance as yf
import pandas as pd
from tqdm import tqdm

from config import (
    STOCK_TICKERS,
    DATA_START_DATE,
    DATA_END_DATE,
    RAW_DATA_DIR,
)


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download OHLCV data for a single stock ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume.

    Raises:
        ValueError: If no data is returned for the ticker.
    """
    print(f"📥 Downloading {ticker} from {start_date} to {end_date}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # Clean up the DataFrame
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)

    # Remove any rows with NaN values
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        print(f"   ⚠️  Dropped {initial_len - len(df)} rows with NaN values.")

    # Validate data integrity
    assert (df["High"] >= df["Low"]).all(), f"Data integrity error for {ticker}: High < Low found."
    assert (df["Volume"] >= 0).all(), f"Data integrity error for {ticker}: Negative volume found."

    print(f"   ✅ {ticker}: {len(df)} trading days downloaded.")
    return df


def save_stock_data(df: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    """
    Save stock DataFrame to CSV.

    Args:
        df: Stock OHLCV DataFrame.
        ticker: Stock ticker symbol.
        output_dir: Directory to save the CSV file.

    Returns:
        Path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{ticker}.csv"
    df.to_csv(filepath, index=False)
    return filepath


def collect_all_data(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download and save OHLCV data for all configured stock tickers.

    Args:
        tickers: List of stock tickers. Defaults to config STOCK_TICKERS.
        start_date: Start date. Defaults to config DATA_START_DATE.
        end_date: End date. Defaults to config DATA_END_DATE.
        output_dir: Output directory. Defaults to config RAW_DATA_DIR.

    Returns:
        Dictionary mapping ticker to its DataFrame.
    """
    tickers = tickers or STOCK_TICKERS
    start_date = start_date or DATA_START_DATE
    end_date = end_date or DATA_END_DATE
    output_dir = output_dir or RAW_DATA_DIR

    results = {}
    failed = []

    for ticker in tqdm(tickers, desc="Downloading stocks"):
        try:
            df = download_stock_data(ticker, start_date, end_date)
            filepath = save_stock_data(df, ticker, output_dir)
            results[ticker] = df
            print(f"   💾 Saved to {filepath}")
        except Exception as e:
            print(f"   ❌ Failed to download {ticker}: {e}")
            failed.append(ticker)

    print(f"\n{'='*50}")
    print(f"📊 Download Summary:")
    print(f"   Success: {len(results)}/{len(tickers)}")
    if failed:
        print(f"   Failed: {failed}")
    print(f"   Output: {output_dir}")
    print(f"{'='*50}")

    return results


if __name__ == "__main__":
    collect_all_data()
