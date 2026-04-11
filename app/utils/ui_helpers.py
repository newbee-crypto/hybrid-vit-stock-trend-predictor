"""
ui_helpers.py — Streamlit UI utilities and custom styling.

Provides custom CSS injection, metric card components, and chart
rendering utilities for the Streamlit frontend.

Usage:
    from app.utils.ui_helpers import inject_custom_css, metric_card
"""

import streamlit as st
import plotly.graph_objects as go


def inject_custom_css():
    """Inject custom CSS for a premium dark-themed Streamlit UI."""
    st.markdown("""
    <style>
    /* ━━━ Global Styles ━━━ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ━━━ Sidebar ━━━ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    [data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* ━━━ Metric Cards ━━━ */
    .metric-card {
        background: linear-gradient(135deg, rgba(26, 26, 62, 0.8) 0%, rgba(10, 10, 26, 0.9) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    }

    .metric-card .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #8b8fa3;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fafafa;
    }

    .metric-card .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 4px;
    }

    .delta-positive { color: #00C805; }
    .delta-negative { color: #FF3131; }
    .delta-neutral { color: #FFD700; }

    /* ━━━ Signal Badge ━━━ */
    .signal-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .signal-buy {
        background: linear-gradient(135deg, #00C805 0%, #00a804 100%);
        color: #000;
        box-shadow: 0 4px 15px rgba(0, 200, 5, 0.3);
    }

    .signal-sell {
        background: linear-gradient(135deg, #FF3131 0%, #cc2727 100%);
        color: #fff;
        box-shadow: 0 4px 15px rgba(255, 49, 49, 0.3);
    }

    .signal-hold {
        background: linear-gradient(135deg, #FFD700 0%, #ccac00 100%);
        color: #000;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
    }

    /* ━━━ Prediction Banner ━━━ */
    .prediction-banner {
        background: linear-gradient(135deg, rgba(26, 26, 62, 0.9) 0%, rgba(10, 10, 40, 0.95) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .prediction-banner h2 {
        color: #fafafa;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 16px;
    }

    .prediction-banner .trend-text {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 8px 0;
    }

    .trend-up { color: #00C805; }
    .trend-down { color: #FF3131; }
    .trend-neutral { color: #FFD700; }

    /* ━━━ Chat Interface ━━━ */
    .chat-message {
        padding: 16px 20px;
        border-radius: 16px;
        margin: 8px 0;
        max-width: 85%;
    }

    .chat-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        text-align: right;
    }

    .chat-ai {
        background: rgba(26, 26, 62, 0.8);
        border: 1px solid rgba(102, 126, 234, 0.2);
        color: #e0e0e0;
    }

    /* ━━━ Section Headers ━━━ */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #667eea;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    /* ━━━ Progress Bar Custom ━━━ */
    .confidence-bar {
        background: rgba(26, 26, 62, 0.5);
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin: 8px 0;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }

    /* ━━━ Pattern Card ━━━ */
    .pattern-card {
        background: rgba(26, 26, 62, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }

    .pattern-card h4 {
        color: #667eea;
        margin-bottom: 8px;
    }

    .pattern-card p {
        color: #b0b0c0;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* ━━━ Divider ━━━ */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 24px 0;
    }

    /* ━━━ Responsive tweaks ━━━ */
    @media (max-width: 768px) {
        .metric-card .metric-value { font-size: 1.4rem; }
        .prediction-banner .trend-text { font-size: 1.8rem; }
    }
    </style>
    """, unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str,
    delta: str = "",
    delta_type: str = "neutral",
) -> str:
    """
    Create an HTML metric card component.

    Args:
        label: Card label/title.
        value: Main metric value.
        delta: Optional delta/change value.
        delta_type: 'positive', 'negative', or 'neutral'.

    Returns:
        HTML string for the metric card.
    """
    delta_class = f"delta-{delta_type}"
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""

    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def signal_badge(signal: str) -> str:
    """
    Create an HTML signal badge (Buy/Sell/Hold).

    Args:
        signal: Signal type ('Buy', 'Sell', 'Hold').

    Returns:
        HTML string for the badge.
    """
    signal_class = f"signal-{signal.lower()}"
    return f'<span class="signal-badge {signal_class}">{signal}</span>'


def prediction_banner(
    prediction: str,
    confidence: float,
    ticker: str = "",
) -> str:
    """
    Create an HTML prediction result banner.

    Args:
        prediction: Predicted trend ('Up', 'Down', 'Neutral').
        confidence: Prediction confidence (0-1).
        ticker: Stock ticker symbol.

    Returns:
        HTML string for the banner.
    """
    trend_class = f"trend-{prediction.lower()}"
    trend_emoji = {"Up": "📈", "Down": "📉", "Neutral": "➡️"}.get(prediction, "❓")

    return f"""
    <div class="prediction-banner">
        <h2>{'Trend Prediction for ' + ticker if ticker else 'Trend Prediction'}</h2>
        <div class="trend-text {trend_class}">{trend_emoji} {prediction.upper()}</div>
        <div style="color: #8b8fa3; font-size: 1.1rem;">
            Confidence: <strong style="color: #fafafa;">{confidence:.1%}</strong>
        </div>
    </div>
    """


def pattern_card(
    name: str,
    description: str,
    signal: str,
    accuracy: float,
    similarity: float,
) -> str:
    """
    Create an HTML pattern match card.

    Args:
        name: Pattern name.
        description: Pattern description.
        signal: Expected signal direction.
        accuracy: Historical accuracy (0-1).
        similarity: Similarity score (0-1).

    Returns:
        HTML string for the pattern card.
    """
    signal_color = {
        "Bullish": "#00C805",
        "Bearish": "#FF3131",
        "Neutral": "#FFD700",
    }.get(signal, "#8b8fa3")

    return f"""
    <div class="pattern-card">
        <h4>{name}</h4>
        <p>{description}</p>
        <div style="display: flex; gap: 16px; margin-top: 12px; font-size: 0.85rem;">
            <span style="color: {signal_color};">Signal: {signal}</span>
            <span style="color: #8b8fa3;">Accuracy: {accuracy:.0%}</span>
            <span style="color: #8b8fa3;">Match: {similarity:.0%}</span>
        </div>
    </div>
    """


def confidence_bar(
    confidence: float,
    color: str = "#667eea",
) -> str:
    """
    Create an HTML confidence progress bar.

    Args:
        confidence: Confidence value (0-1).
        color: Bar fill color.

    Returns:
        HTML string for the confidence bar.
    """
    width = confidence * 100
    return f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {width}%; background: linear-gradient(90deg, {color}, {color}dd);"></div>
    </div>
    """


def section_header(text: str) -> str:
    """
    Create a styled section header.

    Args:
        text: Header text.

    Returns:
        HTML string.
    """
    return f'<div class="section-header">{text}</div>'


def custom_divider() -> str:
    """Create a custom horizontal divider."""
    return '<div class="custom-divider"></div>'
