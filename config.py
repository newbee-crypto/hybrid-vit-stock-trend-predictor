"""
Global configuration for the Hybrid ViT Stock Trend Predictor.

Contains all tunable parameters: stock universe, model hyperparameters,
data pipeline settings, signal fusion weights, and path definitions.
"""

import os
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROJECT PATHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHARTS_DIR = DATA_DIR / "charts"
LABELS_DIR = DATA_DIR / "labels"
PATTERNS_DIR = DATA_DIR / "patterns"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# STOCK UNIVERSE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
STOCK_TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]

# Data collection period (3 years of history)
DATA_START_DATE = "2022-01-01"
DATA_END_DATE = "2025-12-31"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
WINDOW_SIZE = 20          # Number of trading days per candlestick chart
IMAGE_SIZE = (224, 224)   # ViT input resolution
CHART_STYLE = "charles"   # mplfinance chart style
CHART_DPI = 100           # DPI for chart rendering

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# LABELING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
FORWARD_WINDOW = 10       # Days ahead to compute return
TREND_UP_THRESHOLD = 0.5      # Volatility-adjusted score threshold → Up
TREND_DOWN_THRESHOLD = -0.5   # Volatility-adjusted score threshold → Down
VOLATILITY_WINDOW = 20        # Rolling window for volatility-normalized labels
# Between thresholds → Neutral

CLASS_NAMES = ["Down", "Neutral", "Up"]
NUM_CLASSES = len(CLASS_NAMES)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNICAL INDICATORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA SPLIT (time-based)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
VIT_MODEL_NAME = "vit_small_patch16_224"
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 6
SCHEDULER_T_MAX = 10      # Legacy scheduler setting

# Best checkpoint filename
BEST_CHECKPOINT = "best_model.pth"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIGNAL FUSION WEIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
FUSION_WEIGHTS = {
    "vit": 0.40,         # ViT model prediction
    "rag": 0.20,         # RAG candlestick pattern match
    "news": 0.25,        # News sentiment (Tavily)
    "technical": 0.15,   # RSI + MACD indicators
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENAI / LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "candlestick_patterns"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# MEMORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
MEMORY_WINDOW = 5         # Number of chat turns to remember
INTEREST_THRESHOLD = 3    # Queries on same stock before personalization

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKTESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
INITIAL_CAPITAL = 100_000.0
COMMISSION_RATE = 0.001   # 0.1% per trade

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# STREAMLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
STREAMLIT_PAGE_TITLE = "Hybrid ViT Stock Trend Predictor"
STREAMLIT_PAGE_ICON = "📊"
STREAMLIT_LAYOUT = "wide"


def ensure_dirs():
    """Create all required project directories if they don't exist."""
    dirs = [
        RAW_DATA_DIR,
        CHARTS_DIR,
        LABELS_DIR,
        PATTERNS_DIR,
        CHECKPOINTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("✅ All directories created successfully.")
    print(f"   Project root: {PROJECT_ROOT}")
    print(f"   Stocks: {STOCK_TICKERS}")
    print(f"   Window: {WINDOW_SIZE} days")
    print(f"   Image size: {IMAGE_SIZE}")
    print(f"   Model: {VIT_MODEL_NAME}")
