"""
1_Analyze_Chart.py — Chart Analysis Page.

Allows users to upload a chart image or select a stock + date range,
then displays: prediction, Grad-CAM heatmap, explanation, patterns,
and signal fusion results.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta

from config import STOCK_TICKERS, CHARTS_DIR, LABELS_DIR, CHECKPOINTS_DIR
from src.monitoring.metrics import (
    INFERENCE_LATENCY,
    INFERENCE_REQUESTS,
    MODEL_LOADED,
    PREDICTION_CONFIDENCE,
)
from utils.ui_helpers import (
    inject_custom_css,
    metric_card,
    signal_badge,
    prediction_banner,
    pattern_card,
    confidence_bar,
    section_header,
    custom_divider,
)

inject_custom_css()

st.markdown("# 🔮 Analyze Chart")
st.markdown("Upload a candlestick chart image or select a stock to get AI-powered trend predictions.")

st.markdown(custom_divider(), unsafe_allow_html=True)

# ━━━ Input Mode ━━━
tab1, tab2 = st.tabs(["📈 Select Stock", "📤 Upload Image"])

image_to_analyze = None
selected_ticker = None
rsi_val = None
macd_val = None
macd_signal_val = None

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        selected_ticker = st.selectbox(
            "Select Stock",
            STOCK_TICKERS,
            index=0,
            key="analyze_ticker",
        )

    with col2:
        # Get available dates for selected ticker
        chart_dir = CHARTS_DIR / selected_ticker
        available_charts = []
        if chart_dir.exists():
            available_charts = sorted(chart_dir.glob("*.png"))

        if available_charts:
            chart_dates = [c.stem for c in available_charts]
            selected_date = st.selectbox(
                "Select Date",
                chart_dates[-20:],  # Show last 20 dates
                index=len(chart_dates[-20:]) - 1,
                key="analyze_date",
            )
            image_path = chart_dir / f"{selected_date}.png"
            if image_path.exists():
                image_to_analyze = image_path
        else:
            st.warning(f"No charts found for {selected_ticker}. Run the data pipeline first.")

    # Load technical indicators
    labels_path = LABELS_DIR / "labeled_dataset.csv"
    if labels_path.exists() and selected_ticker:
        try:
            df = pd.read_csv(labels_path)
            ticker_data = df[df["ticker"] == selected_ticker].sort_values("date")
            if not ticker_data.empty:
                last_row = ticker_data.iloc[-1]
                rsi_val = last_row.get("RSI")
                macd_val = last_row.get("MACD")
                macd_signal_val = last_row.get("MACD_signal")
        except Exception:
            pass

    if image_to_analyze and st.button("🔍 Analyze", type="primary", use_container_width=True, key="analyze_btn_stock"):
        pass  # Analysis runs below

with tab2:
    st.caption("Demo mode only. For strongest results, use the stock selector flow.")
    uploaded_file = st.file_uploader(
        "Upload a candlestick chart image (PNG, JPG)",
        type=["png", "jpg", "jpeg"],
        key="chart_upload",
    )

    if uploaded_file:
        image_to_analyze = Image.open(uploaded_file)
        selected_ticker = st.text_input("Ticker symbol (optional)", "STOCK", key="upload_ticker")

    if uploaded_file and st.button("🔍 Analyze", type="primary", use_container_width=True, key="analyze_btn_upload"):
        pass  # Analysis runs below

# ━━━ Run Analysis ━━━
if image_to_analyze is not None:
    st.markdown(custom_divider(), unsafe_allow_html=True)

    with st.spinner("🧠 Running AI analysis..."):
        try:
            # 1. ViT Prediction
            from src.model.inference import predict_image, build_live_aux_features
            aux_features = build_live_aux_features(rsi=rsi_val, macd=macd_val)
            inference_start = time.perf_counter()
            if isinstance(image_to_analyze, (str, Path)):
                prediction_result = predict_image(
                    str(image_to_analyze),
                    aux_features=aux_features,
                )
                display_image = Image.open(image_to_analyze)
            else:
                prediction_result = predict_image(
                    image_to_analyze,
                    aux_features=aux_features,
                )
                display_image = image_to_analyze

            prediction = prediction_result["prediction"]
            confidence = prediction_result["confidence"]
            probabilities = prediction_result["probabilities"]

            # Record the complete model-serving operation, including model loading.
            INFERENCE_LATENCY.observe(time.perf_counter() - inference_start)
            INFERENCE_REQUESTS.labels(prediction=prediction).inc()
            PREDICTION_CONFIDENCE.observe(confidence)
            MODEL_LOADED.set(1)

            # Display prediction banner
            st.markdown(
                prediction_banner(prediction, confidence, selected_ticker or ""),
                unsafe_allow_html=True,
            )

            if prediction_result.get("is_fallback"):
                st.warning("⚠️ Using untrained model — predictions may not be reliable. Train the model first.")

            # Layout: Image + Grad-CAM | Details
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown(section_header("📊 Chart & Heatmap"), unsafe_allow_html=True)

                # Show original chart
                st.image(display_image, caption="Candlestick Chart", use_container_width=True)

                # 2. Grad-CAM
                try:
                    from src.xai.gradcam import generate_gradcam
                    gradcam_result = generate_gradcam(
                        image_to_analyze,
                        device_str="cpu",
                    )
                    st.image(
                        gradcam_result["overlay"],
                        caption="Grad-CAM Heatmap — Model Focus Areas",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.info(f"Grad-CAM visualization unavailable: {str(e)[:100]}")

            with col_right:
                st.markdown(section_header("📈 Probability Distribution"), unsafe_allow_html=True)

                # Probability bars
                for class_name in ["Up", "Neutral", "Down"]:
                    prob = probabilities.get(class_name, 0)
                    color = {"Up": "#00C805", "Down": "#FF3131", "Neutral": "#FFD700"}[class_name]
                    st.markdown(f"**{class_name}**: {prob:.1%}")
                    st.markdown(confidence_bar(prob, color), unsafe_allow_html=True)

                # Technical indicators
                if rsi_val is not None or macd_val is not None:
                    st.markdown(section_header("📏 Technical Indicators"), unsafe_allow_html=True)

                    t_col1, t_col2 = st.columns(2)
                    with t_col1:
                        rsi_display = f"{rsi_val:.1f}" if rsi_val is not None else "N/A"
                        rsi_delta = "Overbought" if rsi_val and rsi_val > 70 else "Oversold" if rsi_val and rsi_val < 30 else "Neutral"
                        rsi_type = "negative" if rsi_val and rsi_val > 70 else "positive" if rsi_val and rsi_val < 30 else "neutral"
                        st.markdown(metric_card("RSI (14)", rsi_display, rsi_delta, rsi_type), unsafe_allow_html=True)

                    with t_col2:
                        macd_display = f"{macd_val:.4f}" if macd_val is not None else "N/A"
                        macd_delta = "Bullish" if macd_val and macd_signal_val and macd_val > macd_signal_val else "Bearish"
                        macd_type = "positive" if macd_delta == "Bullish" else "negative"
                        st.markdown(metric_card("MACD", macd_display, macd_delta, macd_type), unsafe_allow_html=True)

            # 3. Attention Analysis
            st.markdown(custom_divider(), unsafe_allow_html=True)
            st.markdown(section_header("🔍 Attention Analysis"), unsafe_allow_html=True)

            attention_description = ""
            try:
                from src.xai.attention_viz import describe_attention
                attention_result = describe_attention(
                    image_to_analyze,
                    device_str="cpu",
                )
                attention_description = attention_result.get("description", "")
                st.info(f"🔍 {attention_description}")

                # Show top regions
                top_regions = attention_result.get("top_regions", [])
                if top_regions:
                    r_cols = st.columns(len(top_regions))
                    for idx, region in enumerate(top_regions):
                        with r_cols[idx]:
                            st.markdown(
                                metric_card(
                                    region["description"],
                                    f"{region['attention_score']:.1%}",
                                ),
                                unsafe_allow_html=True,
                            )
            except Exception as e:
                st.info(f"Attention analysis unavailable: {str(e)[:100]}")

            # 4. GenAI Explanation
            st.markdown(custom_divider(), unsafe_allow_html=True)
            st.markdown(section_header("🧠 AI Explanation"), unsafe_allow_html=True)

            try:
                from src.genai.explainer import explain_prediction
                explanation = explain_prediction(
                    prediction=prediction,
                    confidence=confidence,
                    probabilities=probabilities,
                    attention_description=attention_description,
                    rsi=rsi_val,
                    macd=macd_val,
                    ticker=selected_ticker or "the stock",
                )
                st.markdown(f"> {explanation}")
            except Exception as e:
                st.info(f"AI explanation unavailable: {str(e)[:100]}")

            # 5. RAG Pattern Matching
            st.markdown(custom_divider(), unsafe_allow_html=True)
            st.markdown(section_header("🔄 Similar Candlestick Patterns"), unsafe_allow_html=True)

            similar_patterns = []
            try:
                from src.genai.rag_patterns import find_similar_patterns, initialize_pattern_db
                initialize_pattern_db()
                similar_patterns = find_similar_patterns(
                    prediction=prediction,
                    confidence=confidence,
                    rsi=rsi_val,
                    macd=macd_val,
                    attention_description=attention_description,
                )

                if similar_patterns:
                    p_cols = st.columns(len(similar_patterns))
                    for idx, pat in enumerate(similar_patterns):
                        with p_cols[idx]:
                            st.markdown(
                                pattern_card(
                                    name=pat["name"],
                                    description=pat["description"],
                                    signal=pat["signal"],
                                    accuracy=pat["historical_accuracy"],
                                    similarity=pat["similarity_score"],
                                ),
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("No similar patterns found.")
            except Exception as e:
                st.info(f"Pattern matching unavailable: {str(e)[:100]}")

            # 6. Signal Fusion
            st.markdown(custom_divider(), unsafe_allow_html=True)
            st.markdown(section_header("⚡ Signal Fusion"), unsafe_allow_html=True)

            try:
                from src.genai.signal_fusion import fuse_signals
                fusion_result = fuse_signals(
                    vit_result=prediction_result,
                    rag_patterns=similar_patterns,
                    ticker=selected_ticker or "AAPL",
                    rsi=rsi_val,
                    macd=macd_val,
                    macd_signal_val=macd_signal_val,
                )

                # Final signal badge
                f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
                with f_col1:
                    st.markdown(
                        f"<div style='text-align:center;'>{signal_badge(fusion_result['signal'])}</div>",
                        unsafe_allow_html=True,
                    )
                with f_col2:
                    st.markdown(
                        metric_card("Combined Confidence", f"{fusion_result['confidence']:.1%}"),
                        unsafe_allow_html=True,
                    )
                with f_col3:
                    st.markdown(
                        metric_card("Signal Agreement", f"{fusion_result['agreement']:.0%}"),
                        unsafe_allow_html=True,
                    )

                # Component breakdown
                st.markdown("**Component Breakdown:**")
                comp_cols = st.columns(4)
                component_names = {"vit": "🤖 ViT Model", "rag": "🔄 RAG Patterns", "news": "📰 News", "technical": "📏 Technicals"}
                for idx, (name, sig) in enumerate(fusion_result["components"].items()):
                    weight = fusion_result["weights"][name]
                    with comp_cols[idx]:
                        delta_type = "positive" if sig["signal"] == "Buy" else "negative" if sig["signal"] == "Sell" else "neutral"
                        st.markdown(
                            metric_card(
                                f"{component_names.get(name, name)} ({weight:.0%})",
                                sig["signal"],
                                f"Score: {sig['score']:+.3f}",
                                delta_type,
                            ),
                            unsafe_allow_html=True,
                        )

                # Fusion explanation
                try:
                    from src.genai.explainer import explain_signal_fusion
                    fusion_explanation = explain_signal_fusion(fusion_result, selected_ticker or "the stock")
                    st.markdown(f"> {fusion_explanation}")
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Signal fusion error: {str(e)[:200]}")

            # Track in memory
            if "memory_manager" in st.session_state and selected_ticker:
                st.session_state.memory_manager.add_turn(
                    f"Analyzed {selected_ticker} chart",
                    f"Prediction: {prediction} ({confidence:.1%})",
                )

        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
            st.exception(e)

else:
    st.info("👆 Select a stock and date above, or upload a chart image to begin analysis.")
