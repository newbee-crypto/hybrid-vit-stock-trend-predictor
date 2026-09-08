"""
4_Performance.py -- Model performance dashboard.

Displays model evaluation metrics, confusion matrix, per-class scores,
and training history.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import json
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from config import CHECKPOINTS_DIR, CLASS_NAMES
from utils.ui_helpers import (
    inject_custom_css,
    metric_card,
    section_header,
    custom_divider,
    confidence_bar,
)

inject_custom_css()

st.markdown("# Model Performance")
st.markdown("View model evaluation metrics and training history.")

st.markdown(custom_divider(), unsafe_allow_html=True)

metrics_path = CHECKPOINTS_DIR / "evaluation_metrics.json"
history_path = CHECKPOINTS_DIR / "training_history.json"

if metrics_path.exists():
    with open(metrics_path) as f:
        eval_metrics = json.load(f)

    st.markdown(section_header("Test Set Metrics"), unsafe_allow_html=True)

    m_cols = st.columns(4)
    with m_cols[0]:
        acc = eval_metrics.get("accuracy", 0)
        acc_type = "positive" if acc > 0.6 else "negative" if acc < 0.4 else "neutral"
        st.markdown(metric_card("Accuracy", f"{acc:.1%}", delta_type=acc_type), unsafe_allow_html=True)
    with m_cols[1]:
        f1 = eval_metrics.get("f1_macro", 0)
        f1_type = "positive" if f1 > 0.5 else "negative" if f1 < 0.3 else "neutral"
        st.markdown(metric_card("F1 Score (Macro)", f"{f1:.4f}", delta_type=f1_type), unsafe_allow_html=True)
    with m_cols[2]:
        st.markdown(metric_card("Test Samples", str(eval_metrics.get("num_samples", 0))), unsafe_allow_html=True)
    with m_cols[3]:
        st.markdown(metric_card("Classes", str(len(CLASS_NAMES))), unsafe_allow_html=True)

    st.markdown(custom_divider(), unsafe_allow_html=True)
    st.markdown(section_header("Confusion Matrix"), unsafe_allow_html=True)

    cm = eval_metrics.get("confusion_matrix")
    if cm:
        cm_array = np.array(cm)
        fig = go.Figure(data=go.Heatmap(
            z=cm_array,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            text=cm_array,
            texttemplate="%{text}",
            hoverongaps=False,
        ))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            height=420,
        )

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.plotly_chart(fig, use_container_width=True)
        with col_right:
            st.markdown(section_header("Per-Class Metrics"), unsafe_allow_html=True)
            report = eval_metrics.get("classification_report", {})
            for class_name in CLASS_NAMES:
                class_metrics = report.get(class_name, {})
                precision = class_metrics.get("precision", 0)
                recall = class_metrics.get("recall", 0)
                f1_score = class_metrics.get("f1-score", 0)
                support = class_metrics.get("support", 0)

                with st.expander(f"{class_name} - F1: {f1_score:.3f}", expanded=True):
                    pc_cols = st.columns(3)
                    with pc_cols[0]:
                        st.metric("Precision", f"{precision:.3f}")
                    with pc_cols[1]:
                        st.metric("Recall", f"{recall:.3f}")
                    with pc_cols[2]:
                        st.metric("Support", f"{support}")

                    color = {"Up": "#00C805", "Down": "#FF3131", "Neutral": "#FFD700"}[class_name]
                    st.markdown(f"F1 Score: {f1_score:.1%}")
                    st.markdown(confidence_bar(f1_score, color), unsafe_allow_html=True)
else:
    st.info("No evaluation metrics found. Run `python -m src.model.evaluate` first.")

st.markdown(custom_divider(), unsafe_allow_html=True)
st.markdown(section_header("Training History"), unsafe_allow_html=True)

if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines+markers", name="Train Loss"))
    fig_loss.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers", name="Val Loss"))
    fig_loss.update_layout(
        title="Training & Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        hovermode="x unified",
    )

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=epochs, y=history["train_acc"], mode="lines+markers", name="Train Accuracy"))
    fig_acc.add_trace(go.Scatter(x=epochs, y=history["val_acc"], mode="lines+markers", name="Val Accuracy"))
    fig_acc.update_layout(
        title="Training & Validation Accuracy",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        hovermode="x unified",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_loss, use_container_width=True)
    with col2:
        st.plotly_chart(fig_acc, use_container_width=True)
else:
    st.info("No training history found. Train the model first.")
