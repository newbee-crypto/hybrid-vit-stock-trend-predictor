"""
label_data.py — Label candlestick chart images with trend direction.

Computes forward returns, technical indicators (RSI, MACD), and assigns
Up/Down/Neutral labels. Performs time-based train/val/test split.

Usage:
    python -m src.data.label_data
    python src/data/label_data.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import (
    STOCK_TICKERS,
    RAW_DATA_DIR,
    CHARTS_DIR,
    LABELS_DIR,
    WINDOW_SIZE,
    FORWARD_WINDOW,
    TREND_UP_THRESHOLD,
    TREND_DOWN_THRESHOLD,
    VOLATILITY_WINDOW,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    TRAIN_RATIO,
    VAL_RATIO,
    CLASS_NAMES,
)


def assign_time_based_splits(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Assign train/val/test splits using unique chart dates.

    Keeping an entire date in a single split avoids leakage where some
    tickers from the same market day land in train and others in val/test.
    """
    dataset = dataset.copy()
    unique_dates = np.array(sorted(dataset["date"].unique()))

    if len(unique_dates) < 3:
        dataset["split"] = "train"
        return dataset

    train_cut_idx = min(max(int(len(unique_dates) * TRAIN_RATIO) - 1, 0), len(unique_dates) - 1)
    val_cut_idx = min(
        max(int(len(unique_dates) * (TRAIN_RATIO + VAL_RATIO)) - 1, train_cut_idx),
        len(unique_dates) - 1,
    )

    train_cut_date = unique_dates[train_cut_idx]
    val_cut_date = unique_dates[val_cut_idx]

    dataset["split"] = "test"
    dataset.loc[dataset["date"] <= train_cut_date, "split"] = "train"
    dataset.loc[
        (dataset["date"] > train_cut_date) & (dataset["date"] <= val_cut_date),
        "split",
    ] = "val"
    return dataset


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI and MACD technical indicators for a stock DataFrame.

    Args:
        df: DataFrame with OHLCV data (must have 'Close' column).

    Returns:
        DataFrame with added RSI, MACD, MACD_signal, MACD_hist columns.
    """
    df = df.copy()

    close = df["Close"].astype(float)

    # RSI using Wilder-style exponential smoothing.
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = losses.ewm(alpha=1 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50.0)

    # MACD using exponential moving averages.
    ema_fast = close.ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(
        span=MACD_SIGNAL,
        adjust=False,
        min_periods=MACD_SIGNAL,
    ).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    df[["MACD", "MACD_signal", "MACD_hist"]] = df[
        ["MACD", "MACD_signal", "MACD_hist"]
    ].fillna(0.0)

    return df


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trend labels based on forward returns.

    The label is determined by the percentage change in closing price
    over the next FORWARD_WINDOW trading days:
      - Up: return > TREND_UP_THRESHOLD (+.5%)
      - Down: return < TREND_DOWN_THRESHOLD (-.5%)
      - Neutral: otherwise

    Args:
        df: DataFrame with 'Close' column.

    Returns:
        DataFrame with added 'forward_return' and 'label' columns.
    """
    df = df.copy()

    # Compute forward return
    close = df["Close"].astype(float)
    df["forward_return"] = close.shift(-FORWARD_WINDOW) / close - 1.0

    # Normalize the target by recent realized volatility so the same
    # threshold works across low-volatility and high-volatility tickers.
    daily_return = close.pct_change()
    df["rolling_volatility"] = (
        daily_return
        .rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW)
        .std()
        * np.sqrt(FORWARD_WINDOW)
    )
    df["trend_score"] = df["forward_return"] / df["rolling_volatility"].replace(0, np.nan)

    # Assign labels
    conditions = [
        df["trend_score"] > TREND_UP_THRESHOLD,
        df["trend_score"] < TREND_DOWN_THRESHOLD,
    ]
    choices = ["Up", "Down"]
    df["label"] = np.select(conditions, choices, default="Neutral")

    # Convert to integer label
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    df["label_id"] = df["label"].map(label_map)

    return df


def create_labeled_dataset(
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Create the complete labeled dataset by combining chart metadata with
    technical indicators and trend labels.

    Args:
        tickers: List of stock tickers. Defaults to config STOCK_TICKERS.

    Returns:
        DataFrame with columns: image_path, ticker, date, label, label_id,
        RSI, MACD, MACD_signal, forward_return, trend_score, split.
    """
    tickers = tickers or STOCK_TICKERS
    all_records = []

    for ticker in tqdm(tickers, desc="Labeling data"):
        # Load raw data
        csv_path = RAW_DATA_DIR / f"{ticker}.csv"
        if not csv_path.exists():
            print(f"   ⚠️  No raw data for {ticker}, skipping.")
            continue

        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # Compute indicators and labels
        df = compute_technical_indicators(df)
        df = compute_labels(df)

        # Match with chart images
        chart_dir = CHARTS_DIR / ticker
        if not chart_dir.exists():
            print(f"   ⚠️  No charts directory for {ticker}, skipping.")
            continue

        # A chart ending on day i should use the label computed from the
        # same day, not the next trading day. The previous version shifted
        # labels forward by one day and blurred the training target.
        last_labeled_idx = len(df) - FORWARD_WINDOW
        for i in range(WINDOW_SIZE - 1, last_labeled_idx):
            chart_date = df.iloc[i]["Date"]
            date_str = pd.Timestamp(chart_date).strftime("%Y-%m-%d")
            image_path = chart_dir / f"{date_str}.png"

            row = df.iloc[i]

            if (
                pd.isna(row.get("forward_return"))
                or pd.isna(row.get("trend_score"))
                or pd.isna(row.get("label_id"))
            ):
                continue

            if not image_path.exists():
                continue

            all_records.append({
                "image_path": str(image_path),
                "ticker": ticker,
                "date": date_str,
                "label": row["label"],
                "label_id": int(row["label_id"]),
                "RSI": round(row.get("RSI", 0), 2) if not pd.isna(row.get("RSI")) else 50.0,
                "MACD": round(row.get("MACD", 0), 4) if not pd.isna(row.get("MACD")) else 0.0,
                "MACD_signal": round(row.get("MACD_signal", 0), 4) if not pd.isna(row.get("MACD_signal")) else 0.0,
                "forward_return": round(row["forward_return"], 4),
                "trend_score": round(row["trend_score"], 4),
            })

    dataset = pd.DataFrame(all_records)

    if dataset.empty:
        print("❌ No labeled data was generated. Check that charts exist.")
        return dataset

    # Sort by date for time-based split
    dataset = dataset.sort_values("date").reset_index(drop=True)
    dataset = assign_time_based_splits(dataset)

    return dataset


def save_labeled_dataset(dataset: pd.DataFrame) -> Path:
    """
    Save the labeled dataset to CSV and print summary statistics.

    Args:
        dataset: The complete labeled DataFrame.

    Returns:
        Path to the saved CSV file.
    """
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LABELS_DIR / "labeled_dataset.csv"
    dataset.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"🏷️  Labeling Summary:")
    print(f"   Total samples: {len(dataset)}")
    print(f"\n   Label Distribution:")
    for label in CLASS_NAMES:
        count = (dataset["label"] == label).sum()
        pct = count / len(dataset) * 100
        print(f"     {label}: {count} ({pct:.1f}%)")
    print(f"\n   Split Distribution:")
    for split in ["train", "val", "test"]:
        count = (dataset["split"] == split).sum()
        pct = count / len(dataset) * 100
        print(f"     {split}: {count} ({pct:.1f}%)")
    print(f"\n   Saved to: {output_path}")
    print(f"{'='*50}")

    return output_path


if __name__ == "__main__":
    dataset = create_labeled_dataset()
    if not dataset.empty:
        save_labeled_dataset(dataset)
