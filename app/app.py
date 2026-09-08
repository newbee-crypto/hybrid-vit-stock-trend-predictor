"""
app.py — Main Streamlit application for the Candlestick ViT Predictor.

Configures the multi-page Streamlit app with sidebar navigation,
theme settings, and session state initialization.

Usage:
    streamlit run app/app.py
"""

import sys
from pathlib import Path
from prometheus_client import start_http_server
from src.monitoring.metrics import APP_REQUESTS

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import streamlit as st
from dotenv import load_dotenv

from config import (
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_LAYOUT,
    STOCK_TICKERS,
    ensure_dirs,
)
from utils.ui_helpers import inject_custom_css

# Load environment variables
load_dotenv()

# Page config — MUST be first Streamlit command
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded",
)

# Inject custom CSS
inject_custom_css()

# Ensure directories exist
ensure_dirs()


def initialize_session_state():
    """Initialize all session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "memory_manager" not in st.session_state:
        from src.memory.chat_memory import ChatMemoryManager
        st.session_state.memory_manager = ChatMemoryManager()

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "model" not in st.session_state:
        st.session_state.model = None

    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = STOCK_TICKERS[0]

@st.cache_resource
def start_prometheus_metrics_server():
    start_http_server(8000)
    return True

def main():
    start_prometheus_metrics_server()
    APP_REQUESTS.labels(page="home").inc()
    """Main application entry point."""
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.markdown("# 📊 ViT Predictor")
        st.markdown("---")

        st.markdown("""
        **Candlestick Chart Analysis** powered by:
        - 🤖 Vision Transformer (ViT)
        - 🧠 Google Gemini AI
        - 📰 Real-time News Sentiment
        - 📏 Technical Indicators

        ---

        ### Navigation
        Use the sidebar pages to access:
        1. **Analyze Chart** — Get predictions
        2. **Chat** — Talk to the analyst
        3. **Backtest** — Test strategies
        4. **Performance** — View metrics
        """)

        st.markdown("---")

        # Quick stock selector
        st.session_state.selected_ticker = st.selectbox(
            "🎯 Quick Stock Select",
            STOCK_TICKERS,
            index=STOCK_TICKERS.index(st.session_state.selected_ticker),
        )

        st.markdown("---")
        st.caption("⚠️ For educational purposes only. Not financial advice.")

    # Main page content
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-size: 3rem; font-weight: 800;">
            Candlestick ViT Predictor
        </h1>
        <p style="color: #8b8fa3; font-size: 1.2rem; max-width: 600px; margin: 0 auto;">
            AI-powered stock trend prediction using Vision Transformers,
            explainable AI, and multi-signal fusion.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2.5rem;">🔮</div>
            <div class="metric-label">Analyze</div>
            <div style="color: #b0b0c0; font-size: 0.85rem;">
                Upload charts or select stocks for AI prediction
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2.5rem;">💬</div>
            <div class="metric-label">Chat</div>
            <div style="color: #b0b0c0; font-size: 0.85rem;">
                Talk with the AI analyst about any stock
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2.5rem;">📊</div>
            <div class="metric-label">Backtest</div>
            <div style="color: #b0b0c0; font-size: 0.85rem;">
                Test prediction strategies with historical data
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2.5rem;">🎯</div>
            <div class="metric-label">Performance</div>
            <div style="color: #b0b0c0; font-size: 0.85rem;">
                View model metrics and evaluation scores
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Quick start guide
    st.markdown("""
    <div class="section-header">🚀 Quick Start</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    1. **Prepare Data**: Run `python src/data/collect_data.py` to download stock data
    2. **Generate Charts**: Run `python src/data/generate_charts.py` to create chart images
    3. **Label Data**: Run `python src/data/label_data.py` to create labeled dataset
    4. **Train Model**: Use the Colab notebook or run `python src/model/train.py`
    5. **Start Analyzing**: Navigate to the **Analyze Chart** page

    > 💡 The app works in **fallback mode** even without a trained model —
    > predictions will use the pretrained ViT backbone.
    """)


if __name__ == "__main__":
    main()
