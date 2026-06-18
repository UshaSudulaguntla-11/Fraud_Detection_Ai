"""
FraudLens AI – Explainable Credit Card Fraud Detection System
=============================================================
A professional, end-to-end AI application for detecting fraudulent
credit card transactions using Machine Learning and Explainable AI.

Author: FraudLens AI Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.prediction import (
    load_model,
    load_feature_names,
    predict_single,
    predict_batch,
    validate_upload,
    get_model_metrics,
    get_risk_level,
)
from utils.explainability import (
    get_shap_explainer,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_waterfall,
    get_top_features,
    explain_single_transaction,
    plot_shap_waterfall_from_explanation,
)
from utils.visualization import (
    plot_prediction_distribution,
    plot_amount_distribution,
    plot_probability_distribution,
    plot_model_comparison,
    plot_fraud_by_amount_range,
    COLORS,
)

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FraudLens AI – Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for Professional Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Styles ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar Styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }

    section[data-testid="stSidebar"] .stRadio > label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }

    /* ── Header Gradient Text ── */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: -1px;
        line-height: 1.2;
    }

    .subtitle-text {
        color: #a0aec0;
        font-size: 1.15rem;
        font-weight: 400;
        margin-top: -8px;
        letter-spacing: 0.5px;
    }

    /* ── KPI / Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(26, 32, 44, 0.95) 0%, rgba(45, 55, 72, 0.85) 100%);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 8px 0;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    .metric-icon {
        font-size: 1.6rem;
        margin-bottom: 4px;
    }

    /* ── Feature Cards (Home page) ── */
    .feature-card {
        background: linear-gradient(145deg, rgba(26, 32, 44, 0.9) 0%, rgba(45, 55, 72, 0.7) 100%);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 16px;
        padding: 28px 22px;
        text-align: center;
        transition: all 0.35s ease;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .feature-card:hover {
        transform: translateY(-6px);
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.12);
    }

    .feature-icon {
        font-size: 2.4rem;
        margin-bottom: 12px;
    }

    .feature-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 8px;
    }

    .feature-desc {
        font-size: 0.82rem;
        color: #a0aec0;
        line-height: 1.5;
    }

    /* ── System Architecture Steps ── */
    .arch-step {
        background: linear-gradient(145deg, rgba(26, 32, 44, 0.9) 0%, rgba(45, 55, 72, 0.7) 100%);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 14px;
        padding: 18px 12px;
        text-align: center;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .arch-icon {
        font-size: 1.8rem;
        margin-bottom: 8px;
    }

    .arch-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 4px;
        line-height: 1.3;
    }

    .arch-desc {
        font-size: 0.72rem;
        color: #a0aec0;
        line-height: 1.4;
    }

    .arch-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 130px;
        font-size: 1.4rem;
        color: #667eea;
    }

    /* ── Result Cards ── */
    .result-fraud {
        background: linear-gradient(135deg, rgba(229, 62, 62, 0.15) 0%, rgba(197, 48, 48, 0.1) 100%);
        border: 2px solid rgba(245, 101, 101, 0.5);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .result-legit {
        background: linear-gradient(135deg, rgba(56, 161, 105, 0.15) 0%, rgba(39, 103, 73, 0.1) 100%);
        border: 2px solid rgba(72, 187, 120, 0.5);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    /* ── Section Divider ── */
    .section-divider {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 32px 0;
    }

    /* ── Info Box ── */
    .info-box {
        background: rgba(66, 153, 225, 0.08);
        border-left: 4px solid #4299e1;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #bee3f8;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    .warning-box {
        background: rgba(237, 137, 54, 0.08);
        border-left: 4px solid #ed8936;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #fefcbf;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    /* ── Badge ── */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .badge-fraud {
        background: rgba(245, 101, 101, 0.2);
        color: #fc8181;
        border: 1px solid rgba(245, 101, 101, 0.3);
    }

    .badge-legit {
        background: rgba(72, 187, 120, 0.2);
        color: #68d391;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }

    /* ── Streamlit element overrides ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(26, 32, 44, 0.9) 0%, rgba(45, 55, 72, 0.7) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 16px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #1a202c;
    }
    ::-webkit-scrollbar-thumb {
        background: #4a5568;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #667eea;
    }


</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load Model & Feature Names (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_model():
    """Load the trained Random Forest model."""
    return load_model()


@st.cache_resource(show_spinner=False)
def _load_features():
    """Load feature names list."""
    return load_feature_names()


model = _load_model()
feature_names = _load_features()


# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <span style="font-size: 3rem;">🛡️</span>
        <h2 style="background: linear-gradient(135deg, #667eea, #764ba2);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin: 4px 0 0 0; font-size: 1.5rem;">FraudLens AI</h2>
        <p style="color: #718096; font-size: 0.78rem; margin-top: 2px;">
            v1.0.0 &nbsp;|&nbsp; Powered by ML & SHAP
        </p>
    </div>
    <hr style="border: 0; height: 1px;
               background: linear-gradient(90deg, transparent, rgba(102,126,234,0.4), transparent);
               margin: 8px 0 20px 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "📍 Navigation",
        [
            "🏠 Home",
            "🔍 Single Prediction",
            "📊 Batch Fraud Scanner",
            "🧠 Explainable AI",
            "📈 Analytics Dashboard",
            "ℹ️ About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 10px 0; color: #4a5568; font-size: 0.75rem;">
        Built with ❤️ using<br>
        Streamlit • scikit-learn • SHAP
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 1 – HOME
# ══════════════════════════════════════════════
if page == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 30px 0 10px 0;">
        <span style="font-size: 4rem;">🛡️</span>
        <h1 class="gradient-text">FraudLens AI</h1>
        <p class="subtitle-text">Explainable Credit Card Fraud Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Feature Cards
    st.markdown("### ✨ Key Features")
    cols = st.columns(5)

    features_data = [
        ("🎯", "Fraud Detection", "ML-powered prediction with Random Forest achieving 0.83 F1 Score"),
        ("🧠", "Explainable AI", "SHAP-based explanations for every prediction"),
        ("📊", "Batch Analysis", "Upload CSV files for bulk fraud scanning"),
        ("📈", "Analytics Dashboard", "Interactive visualizations & real-time insights"),
        ("📥", "Downloadable Reports", "Export predictions & analysis as CSV"),
    ]

    for col, (icon, title, desc) in zip(cols, features_data):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Model Performance Metrics
    st.markdown("### 🏆 Model Performance Comparison")

    metrics = get_model_metrics()
    fig = plot_model_comparison(metrics)
    st.plotly_chart(fig, use_container_width=True)

    # Model comparison table
    model_cols = st.columns(3)
    for col, (model_name, model_metrics) in zip(model_cols, metrics.items()):
        with col:
            is_selected = model_name == "Random Forest"
            border_color = "rgba(72,187,120,0.6)" if is_selected else "rgba(102,126,234,0.25)"
            badge_html = '<span class="badge badge-legit" style="margin-left: 8px;">SELECTED</span>' if is_selected else ""

            st.markdown(f"""
            <div class="metric-card" style="border-color: {border_color};">
                <div class="metric-label">{model_name}{badge_html}</div>
                <div style="margin-top: 12px;">
                    <span style="color: #667eea;">Precision:</span>
                    <strong style="color: #e2e8f0;">{model_metrics['Precision']:.2f}</strong> &nbsp;|&nbsp;
                    <span style="color: #667eea;">Recall:</span>
                    <strong style="color: #e2e8f0;">{model_metrics['Recall']:.2f}</strong> &nbsp;|&nbsp;
                    <span style="color: #667eea;">F1:</span>
                    <strong style="color: #e2e8f0;">{model_metrics['F1 Score']:.2f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Dataset Overview
    st.markdown("### 📦 Dataset Overview")
    ds_cols = st.columns(4)
    ds_data = [
        ("📋", "284,807", "Total Transactions", "#667eea"),
        ("✅", "284,315", "Legitimate", "#48bb78"),
        ("🚨", "492", "Fraudulent", "#f56565"),
        ("⚖️", "0.1727%", "Fraud Rate", "#ed8936"),
    ]
    for col, (icon, value, label, color) in zip(ds_cols, ds_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value" style="color: {color};">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # System Architecture / Workflow
    st.markdown("### 🏗️ System Architecture")
    st.markdown(
        '<p class="subtitle-text" style="font-size:0.95rem; margin-top:-4px;">'
        "How a transaction flows through FraudLens AI, end to end</p>",
        unsafe_allow_html=True,
    )

    arch_steps = [
        ("📥", "Transaction Data", "Raw input — Time, Amount, V1–V28"),
        ("🌲", "Random Forest Model", "Trained classifier scores the transaction"),
        ("🚦", "Fraud Prediction", "Label + fraud probability + confidence"),
        ("🧠", "SHAP Explainability", "Per-feature contribution breakdown"),
        ("📈", "Analytics Dashboard", "Aggregated insights & monitoring"),
    ]

    arch_cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])
    step_idx = 0
    for i, col in enumerate(arch_cols):
        with col:
            if i % 2 == 0:
                icon, title, desc = arch_steps[step_idx]
                st.markdown(f"""
                <div class="arch-step">
                    <div class="arch-icon">{icon}</div>
                    <div class="arch-title">{title}</div>
                    <div class="arch-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
                step_idx += 1
            else:
                st.markdown(
                    '<div class="arch-arrow">➜</div>', unsafe_allow_html=True
                )


# ══════════════════════════════════════════════
# PAGE 2 – SINGLE TRANSACTION PREDICTION
# ══════════════════════════════════════════════
elif page == "🔍 Single Prediction":
    st.markdown("""
    <h1 class="gradient-text" style="font-size: 2rem;">🔍 Single Transaction Prediction</h1>
    <p class="subtitle-text">Enter transaction details to predict fraud probability</p>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Info box
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ How to use:</strong> Enter values for each feature below.
        Features V1–V28 are PCA-transformed and do not have direct semantic meaning.
        The <strong>Time</strong> feature represents seconds elapsed since the first transaction,
        and <strong>Amount</strong> is the transaction value.
    </div>
    """, unsafe_allow_html=True)

    # Input form
    user_input = {}

    # Sample fraud transaction values, used by the "Load Sample" button below.
    SAMPLE_FRAUD_TXN = {
        "Time": 406.0, "V1": -2.3122, "V2": 1.9519, "V3": -1.6085,
        "V4": 3.9979, "V5": -0.5228, "V6": -1.4265, "V7": -2.5372,
        "V8": 1.3916, "V9": -2.7709, "V10": -2.7724, "V11": 3.2020,
        "V12": -2.8999, "V13": -0.5952, "V14": -4.2896, "V15": 0.3899,
        "V16": -1.1408, "V17": -2.8300, "V18": -0.0168, "V19": 0.4163,
        "V20": 0.1264, "V21": 0.5170, "V22": -0.0351, "V23": -0.4652,
        "V24": 0.3200, "V25": 0.0445, "V26": 0.1778, "V27": 0.2612,
        "V28": -0.1430, "Amount": 239.93,
    }

    # Streamlit forbids writing to st.session_state[key] once a widget with
    # that key has been instantiated *in the current run*. So instead of
    # setting session_state right before the widgets below are created
    # (too late — Python has already executed the button click handling
    # code, but it hasn't executed the widget-creation code yet... except
    # it has, because this block was originally placed AFTER the widgets).
    #
    # The fix: check for a "pending load" flag and populate session_state
    # here, at the very top of the page, before any number_input widget
    # below is instantiated. The flag itself is set by the button further
    # down and triggers a rerun, so this block runs again on the next pass
    # — this time early enough to safely seed the values.
    if st.session_state.get("_load_sample_pending", False):
        st.session_state["time_input"] = SAMPLE_FRAUD_TXN["Time"]
        st.session_state["amount_input"] = SAMPLE_FRAUD_TXN["Amount"]
        for feat in [f"V{i}" for i in range(1, 29)]:
            st.session_state[f"v_input_{feat}"] = SAMPLE_FRAUD_TXN[feat]
        st.session_state["_load_sample_pending"] = False
        st.info("✅ Sample fraud transaction loaded! Click **Predict Transaction** to analyze.")

    with st.expander("⏱️ Time & Amount", expanded=True):
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            user_input["Time"] = st.number_input(
                "Time (seconds since first transaction)",
                value=0.0, format="%.2f", key="time_input",
            )
        with t_col2:
            user_input["Amount"] = st.number_input(
                "Amount ($)",
                value=0.0, format="%.2f", key="amount_input",
            )

    # V-features grouped into three sections per the design spec, instead
    # of one long flat block — easier to scan and fill in.
    v_feature_groups = [
        ("🔢 Transaction Features – V1 to V10", list(range(1, 11))),
        ("🔢 Transaction Features – V11 to V20", list(range(11, 21))),
        ("🔢 Transaction Features – V21 to V28", list(range(21, 29))),
    ]

    for group_idx, (group_title, v_range) in enumerate(v_feature_groups):
        with st.expander(group_title, expanded=False):
            if group_idx == 0:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Note:</strong> V1–V28 are PCA-transformed features from the original dataset.
                    They cannot be interpreted semantically but are essential for fraud prediction.
                </div>
                """, unsafe_allow_html=True)

            v_features = [f"V{i}" for i in v_range]
            rows = [v_features[i:i + 5] for i in range(0, len(v_features), 5)]
            for row in rows:
                cols = st.columns(len(row))
                for col, feat in zip(cols, row):
                    with col:
                        user_input[feat] = st.number_input(
                            feat, value=0.0, format="%.6f", key=f"v_input_{feat}",
                        )

    st.markdown("")

    # Predict button
    col_btn, col_sample = st.columns([1, 1])
    with col_btn:
        predict_clicked = st.button("🚀 Predict Transaction", use_container_width=True)
    with col_sample:
        sample_clicked = st.button("📋 Load Sample Fraud Transaction", use_container_width=True)

    # Don't touch session_state here — the widgets above already exist for
    # this run. Just set the flag and rerun; the block at the top of this
    # page will pick it up on the next run, before widgets are recreated.
    if sample_clicked:
        st.session_state["_load_sample_pending"] = True
        st.rerun()

    if predict_clicked:
        with st.spinner("🔄 Analyzing transaction..."):
            prediction, fraud_prob, confidence = predict_single(model, user_input)
            risk_level, risk_color = get_risk_level(fraud_prob)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 📋 Prediction Results")

        # Result card
        if prediction == 1:
            st.markdown(f"""
            <div class="result-fraud">
                <span style="font-size: 3.5rem;">🚨</span>
                <h2 style="color: #fc8181; margin: 10px 0 5px 0;">FRAUD DETECTED</h2>
                <p style="color: #feb2b2; font-size: 1.1rem;">This transaction has been flagged as potentially fraudulent</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legit">
                <span style="font-size: 3.5rem;">✅</span>
                <h2 style="color: #68d391; margin: 10px 0 5px 0;">LEGITIMATE TRANSACTION</h2>
                <p style="color: #9ae6b4; font-size: 1.1rem;">This transaction appears to be legitimate</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Metrics row
        res_cols = st.columns(3)
        with res_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-value" style="color: {'#f56565' if fraud_prob > 0.5 else '#48bb78'};">
                    {fraud_prob:.2%}
                </div>
                <div class="metric-label">Fraud Probability</div>
            </div>
            """, unsafe_allow_html=True)
        with res_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value" style="color: #667eea;">{confidence:.2%}</div>
                <div class="metric-label">Model Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with res_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">⚡</div>
                <div class="metric-value" style="color: {risk_color};">{risk_level}</div>
                <div class="metric-label">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Flagship Feature: SHAP explanation for THIS prediction ──
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 🧠 Why Was This Prediction Made?")
        st.markdown("""
        <div class="info-box">
            Fresh SHAP values computed for this exact transaction — not a cached
            sample. Each feature below pushed the prediction toward
            <strong>Fraud</strong> or toward <strong>Legitimate</strong>.
        </div>
        """, unsafe_allow_html=True)

        try:
            with st.spinner("🔄 Computing SHAP explanation for this transaction..."):
                input_df = pd.DataFrame([user_input])
                expected_order = getattr(model, "feature_names_in_", None)
                expected_order = (
                    list(expected_order) if expected_order is not None else feature_names
                )
                input_df = input_df[expected_order]
                contrib_df, explanation = explain_single_transaction(
                    model, input_df, top_n=5
                )

            exp_col1, exp_col2 = st.columns([1, 1])

            with exp_col1:
                st.markdown("#### 🏆 Top 5 Contributing Features")
                for _, row in contrib_df.iterrows():
                    is_fraud_push = row["Direction"] == "Toward Fraud"
                    arrow_color = "#fc8181" if is_fraud_push else "#68d391"
                    sign = "+" if is_fraud_push else "−"
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                padding:10px 14px; margin-bottom:8px; border-radius:10px;
                                background: rgba(26,32,44,0.6); border-left: 3px solid {arrow_color};">
                        <span style="color:#e2e8f0; font-weight:600;">{row['Feature']}</span>
                        <span style="color:{arrow_color}; font-weight:700;">
                            {sign}{row['Contribution_Pct']:.1f}% &nbsp;→&nbsp; {row['Direction']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            with exp_col2:
                fig_single_waterfall = plot_shap_waterfall_from_explanation(explanation)
                st.pyplot(fig_single_waterfall)

        except Exception as exc:
            st.warning(f"⚠️ Could not compute SHAP explanation for this transaction: {exc}")


# ══════════════════════════════════════════════
# PAGE 3 – BATCH FRAUD SCANNER
# ══════════════════════════════════════════════
elif page == "📊 Batch Fraud Scanner":
    st.markdown("""
    <h1 class="gradient-text" style="font-size: 2rem;">📊 Batch Fraud Scanner</h1>
    <p class="subtitle-text">Upload a CSV file to scan multiple transactions at once</p>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Instructions:</strong> Upload a CSV file containing transaction data.
        The file must include columns: <code>Time</code>, <code>V1–V28</code>, and <code>Amount</code>.
        The <code>Class</code> column is optional.
        <br><br>
        📥 A sample file is available in the <code>sample_files/</code> directory.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"],
        help="Upload a CSV file with columns: Time, V1-V28, Amount",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! **{len(df):,}** transactions found.")

            # Validate schema
            is_valid, error_msg = validate_upload(df, feature_names)

            if not is_valid:
                st.error(f"❌ Validation Error: {error_msg}")
            else:
                st.markdown("### 📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True, height=300)

                if st.button("🚀 Scan for Fraud", use_container_width=True):
                    with st.spinner("🔄 Scanning transactions... This may take a moment."):
                        results_df = predict_batch(model, df, feature_names)

                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                    # Summary KPIs
                    fraud_count = int((results_df["Prediction"] == 1).sum())
                    legit_count = int((results_df["Prediction"] == 0).sum())
                    total = len(results_df)
                    fraud_pct = (fraud_count / total) * 100 if total > 0 else 0

                    st.markdown("### 🎯 Scan Results Summary")
                    kpi_cols = st.columns(4)

                    kpi_data = [
                        ("📋", f"{total:,}", "Total Scanned", "#667eea"),
                        ("✅", f"{legit_count:,}", "Legitimate", "#48bb78"),
                        ("🚨", f"{fraud_count:,}", "Fraudulent", "#f56565"),
                        ("📊", f"{fraud_pct:.2f}%", "Fraud Rate", "#ed8936"),
                    ]

                    for col, (icon, value, label, color) in zip(kpi_cols, kpi_data):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-icon">{icon}</div>
                                <div class="metric-value" style="color: {color};">{value}</div>
                                <div class="metric-label">{label}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("")

                    # Results table
                    st.markdown("### 📊 Detailed Results")

                    # Add color-coded labels
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "Fraud_Probability": st.column_config.ProgressColumn(
                                "Fraud Probability",
                                format="%.4f",
                                min_value=0,
                                max_value=1,
                            ),
                            "Label": st.column_config.TextColumn("Label"),
                        },
                    )

                    # Download button
                    csv_output = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_output,
                        file_name="fraud_scan_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")


# ══════════════════════════════════════════════
# PAGE 4 – EXPLAINABLE AI
# ══════════════════════════════════════════════
elif page == "🧠 Explainable AI":
    st.markdown("""
    <h1 class="gradient-text" style="font-size: 2rem;">🧠 Explainable AI with SHAP</h1>
    <p class="subtitle-text">Understand how the model makes fraud predictions</p>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # SHAP explanation text
    st.markdown("""
    <div class="info-box">
        <strong>🧠 What is SHAP?</strong><br><br>
        <strong>SHAP (SHapley Additive exPlanations)</strong> is a game-theoretic approach to explain
        the output of any machine learning model. It connects optimal credit allocation with local
        explanations using Shapley values from cooperative game theory.
        <br><br>
        <strong>Key concepts:</strong><br>
        • Each feature gets a SHAP value indicating its contribution to pushing the prediction
        away from the baseline (average prediction).<br>
        • Positive SHAP values push the prediction toward fraud; negative values push toward legitimate.<br>
        • We use <strong>TreeExplainer</strong>, which is optimized for tree-based models like Random Forest.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Note on Features:</strong> V1–V28 are PCA-transformed features from the original
        confidential data. While SHAP reveals their importance, these features
        <strong>cannot be interpreted semantically</strong> (e.g., V14 doesn't correspond to a specific
        transaction attribute like "merchant category"). Only <strong>Time</strong> and
        <strong>Amount</strong> retain their original meaning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Load sample data for SHAP analysis
    @st.cache_data(show_spinner=False)
    def load_sample_data(n_samples: int = 200) -> pd.DataFrame:
        """Load a stratified sample from the dataset for SHAP computation."""
        try:
            df = pd.read_csv("creditcard.csv")
            fraud = df[df["Class"] == 1]
            legit = df[df["Class"] == 0].sample(
                n=min(n_samples, len(df[df["Class"] == 0])),
                random_state=42,
            )
            sample = pd.concat([fraud.head(min(50, len(fraud))), legit], ignore_index=True)
            return sample
        except FileNotFoundError:
            return None

    sample_data = load_sample_data()

    if sample_data is not None:
        X_sample = sample_data[feature_names]

        # Cache SHAP computation
        @st.cache_data(show_spinner=False)
        def cached_shap_values(_model, _X_sample_values, _feature_names):
            """Compute and cache SHAP values."""
            explainer = get_shap_explainer(_model)
            X_df = pd.DataFrame(_X_sample_values, columns=_feature_names)
            shap_vals = compute_shap_values(explainer, X_df)
            return shap_vals, explainer

        with st.spinner("🔄 Computing SHAP values... This may take a moment."):
            shap_values, explainer = cached_shap_values(
                model, X_sample.values.tolist(), feature_names
            )

        # Tabs for different SHAP views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Summary Plot",
            "📈 Feature Importance",
            "🔬 Waterfall (Local)",
            "🏆 Top Features",
        ])

        with tab1:
            st.markdown("#### Global SHAP Summary Plot")
            st.markdown("""
            Each dot represents a feature's SHAP value for a single prediction.
            **Color** indicates the feature value (red = high, blue = low).
            **Horizontal position** shows the impact on prediction.
            """)
            fig_summary = plot_shap_summary(shap_values, X_sample)
            st.pyplot(fig_summary)

        with tab2:
            st.markdown("#### Mean Absolute SHAP Feature Importance")
            st.markdown("""
            This bar chart ranks features by their average absolute SHAP value across
            all predictions — revealing which features have the most impact overall.
            """)
            fig_bar = plot_shap_bar(shap_values, X_sample)
            st.pyplot(fig_bar)

        with tab3:
            st.markdown("#### Transaction-Level Waterfall Explanation")
            st.markdown("""
            Select a specific transaction to see how each feature contributed to
            pushing the prediction from the base value toward fraud or legitimate.
            """)

            max_idx = len(X_sample) - 1
            selected_idx = st.slider(
                "Select transaction index",
                min_value=0,
                max_value=max_idx,
                value=0,
            )

            # Show if fraud or legit
            if "Class" in sample_data.columns:
                actual = sample_data.iloc[selected_idx]["Class"]
                label_text = "🚨 FRAUD" if actual == 1 else "✅ LEGITIMATE"
                st.markdown(f"**Actual Label:** {label_text}")

            fig_waterfall = plot_shap_waterfall(
                explainer, shap_values, X_sample, index=selected_idx,
            )
            st.pyplot(fig_waterfall)

        with tab4:
            st.markdown("#### 🏆 Top Fraud-Driving Features")
            top_df = get_top_features(shap_values, feature_names, top_n=15)
            st.dataframe(top_df, use_container_width=True, hide_index=True)

            # Plotly bar chart of top features
            fig = go.Figure(go.Bar(
                x=top_df["Importance"].values[::-1],
                y=top_df["Feature"].values[::-1],
                orientation="h",
                marker=dict(
                    color=top_df["Importance"].values[::-1],
                    colorscale=[[0, "#667eea"], [1, "#e53e3e"]],
                ),
            ))
            fig.update_layout(
                title="Top Features by Mean |SHAP Value|",
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Feature",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500,
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Dataset file (`creditcard.csv`) not found. SHAP analysis requires the dataset.")


# ══════════════════════════════════════════════
# PAGE 5 – ANALYTICS DASHBOARD
# ══════════════════════════════════════════════
elif page == "📈 Analytics Dashboard":
    st.markdown("""
    <h1 class="gradient-text" style="font-size: 2rem;">📈 Analytics Dashboard</h1>
    <p class="subtitle-text">Interactive analytics and visualization of fraud detection insights</p>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Load dataset for analytics
    @st.cache_data(show_spinner=False)
    def load_analytics_data() -> pd.DataFrame:
        """Load dataset for analytics."""
        try:
            return pd.read_csv("creditcard.csv")
        except FileNotFoundError:
            return None

    analytics_df = load_analytics_data()

    if analytics_df is not None:
        # KPI Row
        total = len(analytics_df)
        fraud_count = int(analytics_df["Class"].sum())
        legit_count = total - fraud_count
        fraud_pct = (fraud_count / total) * 100
        avg_fraud_amount = analytics_df[analytics_df["Class"] == 1]["Amount"].mean()
        avg_legit_amount = analytics_df[analytics_df["Class"] == 0]["Amount"].mean()

        st.markdown("### 📊 Key Performance Indicators")
        kpi_cols = st.columns(5)

        kpi_data = [
            ("📋", f"{total:,}", "Total Transactions", "#667eea"),
            ("✅", f"{legit_count:,}", "Legitimate", "#48bb78"),
            ("🚨", f"{fraud_count:,}", "Fraudulent", "#f56565"),
            ("📊", f"{fraud_pct:.4f}%", "Fraud Rate", "#ed8936"),
            ("💰", f"${avg_fraud_amount:.2f}", "Avg Fraud Amount", "#e53e3e"),
        ]

        for col, (icon, value, label, color) in zip(kpi_cols, kpi_data):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value" style="color: {color};">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Charts row 1
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### 🎯 Class Distribution")
            fig_dist = plot_prediction_distribution(analytics_df["Class"])
            st.plotly_chart(fig_dist, use_container_width=True)

        with chart_col2:
            st.markdown("#### 💰 Transaction Amount Distribution")
            fig_amount = plot_amount_distribution(analytics_df)
            st.plotly_chart(fig_amount, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Charts row 2
        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.markdown("#### 📊 Fraud by Amount Range")
            fig_range = plot_fraud_by_amount_range(analytics_df)
            st.plotly_chart(fig_range, use_container_width=True)

        with chart_col4:
            st.markdown("#### 🏆 Model Performance Comparison")
            metrics = get_model_metrics()
            fig_model = plot_model_comparison(metrics)
            st.plotly_chart(fig_model, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Amount statistics comparison
        st.markdown("#### 📈 Amount Statistics: Fraud vs Legitimate")

        stats_fraud = analytics_df[analytics_df["Class"] == 1]["Amount"].describe()
        stats_legit = analytics_df[analytics_df["Class"] == 0]["Amount"].describe()

        stats_cols = st.columns(2)
        with stats_cols[0]:
            st.markdown("""
            <div class="metric-card" style="border-color: rgba(245,101,101,0.4);">
                <div class="metric-label" style="color: #fc8181;">🚨 Fraudulent Transactions</div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(stats_fraud).T.style.format("{:.2f}"),
                use_container_width=True,
            )

        with stats_cols[1]:
            st.markdown("""
            <div class="metric-card" style="border-color: rgba(72,187,120,0.4);">
                <div class="metric-label" style="color: #68d391;">✅ Legitimate Transactions</div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(stats_legit).T.style.format("{:.2f}"),
                use_container_width=True,
            )

        # Time series visualization
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### ⏱️ Transactions Over Time")

        # Create time bins
        analytics_df["Time_hours"] = analytics_df["Time"] / 3600
        time_bins = pd.cut(analytics_df["Time_hours"], bins=48)
        time_grouped = analytics_df.groupby([time_bins, "Class"]).size().unstack(fill_value=0)

        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=list(range(len(time_grouped))),
            y=time_grouped[0].values if 0 in time_grouped.columns else [],
            mode="lines",
            name="Legitimate",
            line=dict(color="#48bb78", width=2),
            fill="tozeroy",
            fillcolor="rgba(72, 187, 120, 0.1)",
        ))
        if 1 in time_grouped.columns:
            fig_time.add_trace(go.Scatter(
                x=list(range(len(time_grouped))),
                y=time_grouped[1].values,
                mode="lines+markers",
                name="Fraud",
                line=dict(color="#f56565", width=2),
                marker=dict(size=4),
                yaxis="y2",
            ))

        fig_time.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Time Bin (hours)",
            yaxis_title="Legitimate Count",
            yaxis2=dict(
                title="Fraud Count",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            height=400,
            font=dict(family="Inter, sans-serif"),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_time, use_container_width=True)

    else:
        st.warning("⚠️ Dataset file (`creditcard.csv`) not found. Analytics requires the dataset.")


# ══════════════════════════════════════════════
# PAGE 6 – ABOUT PROJECT
# ══════════════════════════════════════════════
elif page == "ℹ️ About Project":
    st.markdown("""
    <h1 class="gradient-text" style="font-size: 2rem;">ℹ️ About FraudLens AI</h1>
    <p class="subtitle-text">Learn about the project, methodology, and technical details</p>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Problem Statement
    st.markdown("### 🎯 Problem Statement")
    st.markdown("""
    Credit card fraud is a significant financial threat, with global losses exceeding
    **$32 billion annually**. As digital transactions continue to grow, so does the
    sophistication of fraudulent activities. Traditional rule-based systems fail to
    keep pace with evolving fraud patterns.

    **FraudLens AI** addresses this challenge by leveraging machine learning to automatically
    detect fraudulent transactions with high accuracy while providing transparent, explainable
    predictions through SHAP (SHapley Additive exPlanations).
    """)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Dataset
    st.markdown("### 📦 Dataset Information")
    st.markdown("""
    The dataset used is the **Credit Card Fraud Detection** dataset containing
    transactions made by European cardholders in September 2013.
    """)

    dataset_info = {
        "Property": [
            "Total Transactions",
            "Features",
            "PCA Features",
            "Non-PCA Features",
            "Target Variable",
            "Normal Transactions",
            "Fraud Transactions",
            "Fraud Percentage",
        ],
        "Value": [
            "284,807",
            "31 columns",
            "V1 – V28 (PCA-transformed)",
            "Time, Amount",
            "Class (0 = Normal, 1 = Fraud)",
            "284,315 (99.83%)",
            "492 (0.17%)",
            "0.1727%",
        ],
    }
    st.table(pd.DataFrame(dataset_info))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ML Pipeline
    st.markdown("### 🔧 Machine Learning Pipeline")

    pipeline_steps = [
        ("1️⃣", "Data Loading & Exploration",
         "Load the dataset and perform EDA including class distribution analysis, "
         "amount distribution visualization, and correlation analysis."),
        ("2️⃣", "Train-Test Split",
         "Split data into 80% training and 20% testing sets with stratification "
         "to maintain class ratios. `random_state=42` for reproducibility."),
        ("3️⃣", "Handle Class Imbalance",
         "Apply SMOTE (Synthetic Minority Over-sampling Technique) **only on training data** "
         "to balance the classes without data leakage."),
        ("4️⃣", "Model Training",
         "Train three models: **Logistic Regression**, **Random Forest**, and **XGBoost**. "
         "Each model is evaluated and compared."),
        ("5️⃣", "Model Evaluation",
         "Evaluate using Precision, Recall, F1 Score, Confusion Matrix, and "
         "Classification Report. **Random Forest** selected as the final model."),
        ("6️⃣", "Model Persistence",
         "Save the best model (`fraud_detection_model.pkl`) and feature names "
         "(`feature_names.pkl`) using joblib for deployment."),
    ]

    for icon, title, desc in pipeline_steps:
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; margin-bottom: 16px;
                    padding: 16px; background: rgba(26,32,44,0.5);
                    border-radius: 12px; border-left: 3px solid #667eea;">
            <span style="font-size: 1.5rem; margin-right: 14px;">{icon}</span>
            <div>
                <strong style="color: #e2e8f0; font-size: 1.05rem;">{title}</strong>
                <p style="color: #a0aec0; margin: 4px 0 0 0; font-size: 0.92rem; line-height: 1.6;">
                    {desc}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Model Comparison
    st.markdown("### 🏆 Model Comparison")
    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Precision": [0.07, 0.83, 0.53],
        "Recall": [0.91, 0.83, 0.89],
        "F1 Score": [0.12, 0.83, 0.66],
        "Selected": ["❌", "✅", "❌"],
    })
    st.table(comparison_df.set_index("Model"))

    st.markdown("""
    <div class="info-box">
        <strong>Why Random Forest?</strong><br>
        Random Forest was selected as the final model due to its balanced performance
        across all metrics — achieving the highest F1 Score (0.83) with strong Precision
        (0.83) and Recall (0.83). While Logistic Regression has high recall, its very
        low precision (0.07) would result in too many false alarms. XGBoost offers a
        good recall (0.89) but lower precision (0.53) compared to Random Forest.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # XAI
    st.markdown("### 🧠 Explainable AI Integration")
    st.markdown("""
    We integrate **SHAP (SHapley Additive exPlanations)** using `TreeExplainer`
    to provide both global and local model explanations:

    - **Global Explanations**: Summary plots and feature importance bar charts
      reveal which features are most influential across all predictions.
    - **Local Explanations**: Waterfall plots show how each feature contributed to
      a specific transaction's prediction.

    > **Important Note**: Features V1–V28 are PCA-transformed from the original
    > confidential data. While SHAP reveals their mathematical importance, these
    > features **cannot be mapped back** to specific real-world transaction attributes.
    > Only **Time** and **Amount** retain their original semantic meaning.
    """)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Future Improvements
    st.markdown("### 🚀 Future Improvements")
    improvements = [
        "🔄 **Real-time Streaming**: Integrate with Apache Kafka for live transaction monitoring",
        "🧪 **Deep Learning**: Experiment with autoencoders and LSTMs for anomaly detection",
        "🌐 **REST API**: Deploy as a FastAPI microservice for system integration",
        "🐳 **Docker**: Containerize the application for consistent deployment",
        "📊 **A/B Testing**: Framework for comparing model versions in production",
        "🔐 **Authentication**: Add user authentication and role-based access control",
        "📱 **Mobile Dashboard**: Responsive design for mobile fraud monitoring",
        "🔔 **Alert System**: Real-time email/SMS alerts for detected fraud",
    ]
    for item in improvements:
        st.markdown(f"- {item}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Tech Stack
    st.markdown("### 🛠️ Tech Stack")
    tech_data = {
        "Category": [
            "Language", "ML Framework", "Boosting", "Imbalanced Data",
            "Explainability", "Web Framework", "Visualization", "Deployment",
        ],
        "Technology": [
            "Python 3.9+", "scikit-learn", "XGBoost",
            "imbalanced-learn (SMOTE)", "SHAP", "Streamlit",
            "Plotly, Matplotlib", "Render / Streamlit Cloud",
        ],
    }
    st.table(pd.DataFrame(tech_data))