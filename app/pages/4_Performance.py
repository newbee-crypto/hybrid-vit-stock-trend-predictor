"""
4_Performance.py — Model Performance Dashboard.

Displays model evaluation metrics, confusion matrix, per-class scores,
and LLM judge evaluation results.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config import CHECKPOINTS_DIR, CLASS_NAMES
from app.utils.ui_helpers import (
    inject_custom_css,
    metric_card,
    section_header,
    custom_divider,
    confidence_bar,
)

inject_custom_css()

st.markdown("# 🎯 Model Performance")
st.markdown("View detailed model evaluation metrics and LLM judge scores.")

st.markdown(custom_divider(), unsafe_allow_html=True)

# ━━━ Load Evaluation Metrics ━━━
metrics_path = CHECKPOINTS_DIR / "evaluation_metrics.json"
history_path = CHECKPOINTS_DIR / "training_history.json"
judge_path = CHECKPOINTS_DIR / "llm_judge_report.json"

if metrics_path.exists():
    with open(metrics_path) as f:
        eval_metrics = json.load(f)

    # ━━━ Overall Metrics ━━━
    st.markdown(section_header("📊 Model Evaluation — Test Set"), unsafe_allow_html=True)

    m_cols = st.columns(4)
    with m_cols[0]:
        acc = eval_metrics.get("accuracy", 0)
        acc_type = "positive" if acc > 0.6 else "negative" if acc < 0.4 else "neutral"
        st.markdown(
            metric_card("Accuracy", f"{acc:.1%}", delta_type=acc_type),
            unsafe_allow_html=True,
        )

    with m_cols[1]:
        f1 = eval_metrics.get("f1_macro", 0)
        f1_type = "positive" if f1 > 0.5 else "negative" if f1 < 0.3 else "neutral"
        st.markdown(
            metric_card("F1 Score (Macro)", f"{f1:.4f}", delta_type=f1_type),
            unsafe_allow_html=True,
        )

    with m_cols[2]:
        st.markdown(
            metric_card("Test Samples", str(eval_metrics.get("num_samples", 0))),
            unsafe_allow_html=True,
        )

    with m_cols[3]:
        st.markdown(
            metric_card("Classes", str(len(CLASS_NAMES))),
            unsafe_allow_html=True,
        )

    # ━━━ Confusion Matrix ━━━
    st.markdown(custom_divider(), unsafe_allow_html=True)
    st.markdown(section_header("🔢 Confusion Matrix"), unsafe_allow_html=True)

    cm = eval_metrics.get("confusion_matrix")
    if cm:
        cm_array = np.array(cm)

        # Plot with Plotly
        fig = go.Figure(data=go.Heatmap(
            z=cm_array,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            text=cm_array,
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"},
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
            width=500,
            height=400,
        )

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.plotly_chart(fig, use_container_width=True)

        # ━━━ Per-Class Metrics ━━━
        with col_right:
            st.markdown(section_header("📋 Per-Class Metrics"), unsafe_allow_html=True)

            report = eval_metrics.get("classification_report", {})
            for class_name in CLASS_NAMES:
                class_metrics = report.get(class_name, {})
                precision = class_metrics.get("precision", 0)
                recall = class_metrics.get("recall", 0)
                f1_score = class_metrics.get("f1-score", 0)
                support = class_metrics.get("support", 0)

                with st.expander(f"**{class_name}** — F1: {f1_score:.3f}", expanded=True):
                    pc_cols = st.columns(3)
                    with pc_cols[0]:
                        st.metric("Precision", f"{precision:.3f}")
                    with pc_cols[1]:
                        st.metric("Recall", f"{recall:.3f}")
                    with pc_cols[2]:
                        st.metric("Support", f"{support}")

                    # Visual bar
                    color = {"Up": "#00C805", "Down": "#FF3131", "Neutral": "#FFD700"}[class_name]
                    st.markdown(f"F1 Score: {f1_score:.1%}")
                    st.markdown(confidence_bar(f1_score, color), unsafe_allow_html=True)

    # ━━━ Confusion matrix image fallback ━━━
    cm_image_path = CHECKPOINTS_DIR / "confusion_matrix.png"
    if cm_image_path.exists() and not cm:
        st.image(str(cm_image_path), caption="Confusion Matrix", use_container_width=True)

else:
    st.info("📦 No evaluation metrics found. Run the evaluation script first:\n```\npython -m src.model.evaluate\n```")

# ━━━ Training History ━━━
st.markdown(custom_divider(), unsafe_allow_html=True)
st.markdown(section_header("📈 Training History"), unsafe_allow_html=True)

if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss chart
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=history["train_loss"],
        mode="lines+markers", name="Train Loss",
        line=dict(color="#667eea", width=2),
        marker=dict(size=4),
    ))
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=history["val_loss"],
        mode="lines+markers", name="Val Loss",
        line=dict(color="#FF3131", width=2),
        marker=dict(size=4),
    ))
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

    # Accuracy chart
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=history["train_acc"],
        mode="lines+markers", name="Train Accuracy",
        line=dict(color="#00C805", width=2),
        marker=dict(size=4),
    ))
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=history["val_acc"],
        mode="lines+markers", name="Val Accuracy",
        line=dict(color="#FFD700", width=2),
        marker=dict(size=4),
    ))
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

    # Learning rate schedule
    if "lr" in history:
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(
            x=epochs, y=history["lr"],
            mode="lines", name="Learning Rate",
            line=dict(color="#764ba2", width=2),
        ))
        fig_lr.update_layout(
            title="Learning Rate Schedule",
            xaxis_title="Epoch",
            yaxis_title="Learning Rate",
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
        )
        st.plotly_chart(fig_lr, use_container_width=True)

else:
    st.info("📦 No training history found. Train the model first.")

# ━━━ LLM Judge Scores ━━━
st.markdown(custom_divider(), unsafe_allow_html=True)
st.markdown(section_header("🧑‍⚖️ LLM Judge Evaluation"), unsafe_allow_html=True)

if judge_path.exists():
    with open(judge_path) as f:
        judge_report = json.load(f)

    aggregate = judge_report.get("aggregate", {})

    j_cols = st.columns(4)
    with j_cols[0]:
        st.markdown(
            metric_card("Accuracy Score", f"{aggregate.get('accuracy', 0):.1f}/5"),
            unsafe_allow_html=True,
        )
    with j_cols[1]:
        st.markdown(
            metric_card("Clarity Score", f"{aggregate.get('clarity', 0):.1f}/5"),
            unsafe_allow_html=True,
        )
    with j_cols[2]:
        st.markdown(
            metric_card("Alignment Score", f"{aggregate.get('alignment', 0):.1f}/5"),
            unsafe_allow_html=True,
        )
    with j_cols[3]:
        overall = aggregate.get("overall", 0)
        overall_type = "positive" if overall >= 4 else "negative" if overall < 3 else "neutral"
        st.markdown(
            metric_card("Overall Score", f"{overall:.1f}/5", delta_type=overall_type),
            unsafe_allow_html=True,
        )

    # Radar chart
    categories = ["Accuracy", "Clarity", "Alignment"]
    values = [
        aggregate.get("accuracy", 0),
        aggregate.get("clarity", 0),
        aggregate.get("alignment", 0),
    ]
    values.append(values[0])  # Close the polygon
    categories.append(categories[0])

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(102, 126, 234, 0.2)",
        line=dict(color="#667eea", width=2),
        name="LLM Judge Scores",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], color="#8b8fa3"),
            bgcolor="#0e1117",
        ),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        showlegend=False,
        title="Explanation Quality Radar",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Individual evaluations
    individual = judge_report.get("individual_scores", [])
    if individual:
        with st.expander(f"📋 Individual Evaluations ({len(individual)} samples)"):
            for i, score in enumerate(individual):
                st.markdown(f"**Sample {i+1}:** Accuracy={score.get('accuracy')}/5, "
                           f"Clarity={score.get('clarity')}/5, "
                           f"Alignment={score.get('alignment')}/5")
                st.caption(score.get("feedback", ""))

else:
    st.info("📦 No LLM judge evaluation found. Run the evaluation:\n```\npython -m src.evaluation.llm_judge\n```")

    # Offer to run evaluation
    if st.button("🧑‍⚖️ Run LLM Judge Evaluation", type="primary"):
        with st.spinner("Running LLM judge..."):
            try:
                from src.evaluation.llm_judge import evaluate_explanation, save_evaluation_report

                test_explanation = (
                    "Based on the chart analysis, the model predicts a moderate "
                    "chance of upward movement with reasonable confidence."
                )

                scores = evaluate_explanation(
                    prediction="Up",
                    confidence=0.65,
                    explanation=test_explanation,
                    rsi=55.0,
                    macd=0.5,
                    ticker="AAPL",
                )

                result = {
                    "individual_scores": [scores],
                    "aggregate": {
                        "accuracy": scores["accuracy"],
                        "clarity": scores["clarity"],
                        "alignment": scores["alignment"],
                        "overall": scores["average"],
                        "num_evaluated": 1,
                    },
                }

                save_evaluation_report(result)
                st.success("✅ Evaluation complete! Refreshing...")
                st.rerun()

            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")
