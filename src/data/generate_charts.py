"""
generate_charts.py — Generate candlestick chart images from OHLCV data.

Creates 224x224 candlestick chart images using a sliding window approach.
Charts are clean (no axes labels/titles) so the ViT model learns from
pure visual patterns.

Usage:
    python -m src.data.generate_charts
    python src/data/generate_charts.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from config import (
    STOCK_TICKERS,
    RAW_DATA_DIR,
    CHARTS_DIR,
    WINDOW_SIZE,
    IMAGE_SIZE,
    CHART_DPI,
)


# Custom dark style for candlestick charts (better for model learning)
CUSTOM_STYLE = mpf.make_mpf_style(
    base_mpf_style="charles",
    marketcolors=mpf.make_marketcolors(
        up="#00C805",       # Green for up candles
        down="#FF3131",     # Red for down candles
        edge="inherit",
        wick="inherit",
        volume="in",
    ),
    figcolor="#000000",
    facecolor="#000000",
    gridcolor="#1a1a2e",
    gridstyle="--",
    gridaxis="both",
    y_on_right=False,
)


def generate_chart_image(
    df_window: pd.DataFrame,
    output_path: Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
    dpi: int = CHART_DPI,
) -> bool:
    """
    Generate a single candlestick chart image from a DataFrame window.

    Args:
        df_window: DataFrame with OHLCV data (must have DatetimeIndex).
        output_path: Path to save the chart image.
        image_size: Target image size (width, height) in pixels.
        dpi: Dots per inch for render quality.

    Returns:
        True if chart was generated successfully, False otherwise.
    """
    try:
        # Calculate figure size from pixel dimensions
        fig_width = image_size[0] / dpi
        fig_height = image_size[1] / dpi

        # Create the candlestick chart — clean, no axes labels
        fig, axes = mpf.plot(
            df_window,
            type="candle",
            style=CUSTOM_STYLE,
            volume=True,
            figsize=(fig_width, fig_height),
            returnfig=True,
            tight_layout=True,
            axisoff=True,
        )

        # Remove all text, ticks, and labels for clean chart
        for ax in axes:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)

        # Save the figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="#000000",
        )
        plt.close(fig)

        # Resize to exact target size
        img = Image.open(output_path)
        img = img.resize(image_size, Image.LANCZOS)
        img.save(output_path)

        return True

    except Exception as e:
        print(f"   ❌ Error generating chart: {e}")
        plt.close("all")
        return False


def generate_charts_for_ticker(
    ticker: str,
    raw_data_dir: Path = RAW_DATA_DIR,
    charts_dir: Path = CHARTS_DIR,
    window_size: int = WINDOW_SIZE,
) -> list[dict]:
    """
    Generate all candlestick chart images for a single stock ticker.

    Uses a sliding window to create overlapping chart windows.

    Args:
        ticker: Stock ticker symbol.
        raw_data_dir: Directory containing raw CSV files.
        charts_dir: Directory to save chart images.
        window_size: Number of trading days per chart.

    Returns:
        List of dictionaries with chart metadata (ticker, date, image_path).
    """
    csv_path = raw_data_dir / f"{ticker}.csv"
    if not csv_path.exists():
        print(f"   ⚠️  No data file found for {ticker}, skipping.")
        return []

    # Load and prepare data
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.set_index("Date")
    df = df.sort_index()

    # Output directory for this ticker
    ticker_dir = charts_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    chart_metadata = []
    total_windows = len(df) - window_size

    for i in range(total_windows):
        window = df.iloc[i : i + window_size]
        end_date = window.index[-1].strftime("%Y-%m-%d")
        filename = f"{end_date}.png"
        output_path = ticker_dir / filename

        # Skip if already generated
        if output_path.exists():
            chart_metadata.append({
                "ticker": ticker,
                "date": end_date,
                "image_path": str(output_path),
                "window_start": window.index[0].strftime("%Y-%m-%d"),
                "window_end": end_date,
            })
            continue

        success = generate_chart_image(window, output_path)
        if success:
            chart_metadata.append({
                "ticker": ticker,
                "date": end_date,
                "image_path": str(output_path),
                "window_start": window.index[0].strftime("%Y-%m-%d"),
                "window_end": end_date,
            })

    return chart_metadata


def generate_all_charts(
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate candlestick chart images for all configured stock tickers.

    Args:
        tickers: List of stock tickers. Defaults to config STOCK_TICKERS.

    Returns:
        DataFrame with all chart metadata.
    """
    tickers = tickers or STOCK_TICKERS
    all_metadata = []

    for ticker in tqdm(tickers, desc="Generating charts"):
        print(f"\n📊 Generating charts for {ticker}...")
        metadata = generate_charts_for_ticker(ticker)
        all_metadata.extend(metadata)
        print(f"   ✅ {ticker}: {len(metadata)} charts generated.")

    # Save metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = CHARTS_DIR / "charts_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"\n{'='*50}")
    print(f"📊 Chart Generation Summary:")
    print(f"   Total charts: {len(all_metadata)}")
    print(f"   Tickers: {len(tickers)}")
    print(f"   Metadata saved to: {metadata_path}")
    print(f"{'='*50}")

    return metadata_df


if __name__ == "__main__":
    generate_all_charts()
