"""
app.py
------
Streamlit web app for the Investment Risk Profiling model.

Run with:
    streamlit run app.py

Loads the trained model bundle (model.pkl) and lets a user enter an
investor's profile to get a real-time risk category prediction
(Conservative / Moderate / Aggressive) with class probabilities.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "model.pkl"

st.set_page_config(
    page_title="Investment Risk Profiler",
    page_icon="📊",
    layout="centered",
)


@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "model.pkl not found. Run `python train_model.py` first to train "
            "and save the model."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


bundle = load_model_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
label_encoder = bundle["label_encoder"]
features = bundle["features"]
test_accuracy = bundle["test_accuracy"]

st.title("📊 Investment Risk Profiler")
st.write(
    "Enter an investor's profile below to predict their **risk tolerance "
    "category** — Conservative, Moderate, or Aggressive — in real time."
)
st.caption(f"Model test accuracy: **{test_accuracy * 100:.1f}%** (RandomForestClassifier)")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 35)
    annual_income = st.number_input(
        "Annual Income ($)", min_value=20_000, max_value=200_000, value=60_000, step=1_000
    )
    investment_amount = st.number_input(
        "Investment Amount ($)", min_value=1_000, max_value=100_000, value=15_000, step=500
    )
    years_experience = st.slider("Years of Investing Experience", 0, 30, 5)

with col2:
    financial_knowledge = st.slider("Financial Knowledge (1 = low, 10 = expert)", 1, 10, 5)
    portfolio_diversity = st.slider("Portfolio Diversity (1 = low, 10 = high)", 1, 10, 5)
    debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.30, step=0.01)

st.divider()

if st.button("Predict Risk Profile", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [[
            age,
            annual_income,
            investment_amount,
            years_experience,
            financial_knowledge,
            portfolio_diversity,
            debt_to_income,
        ]],
        columns=features,
    )

    input_scaled = scaler.transform(input_df)
    pred_encoded = model.predict(input_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    probabilities = model.predict_proba(input_scaled)[0]

    icon_map = {"Conservative": "🛡️", "Moderate": "⚖️", "Aggressive": "🚀"}
    st.success(f"### Predicted Risk Profile: {icon_map.get(pred_label, '')} **{pred_label}**")

    prob_df = pd.DataFrame(
        {"Risk Category": label_encoder.classes_, "Probability": probabilities}
    ).sort_values("Probability", ascending=False)

    st.write("**Prediction Confidence:**")
    st.bar_chart(prob_df.set_index("Risk Category"))

    with st.expander("See raw probabilities"):
        st.dataframe(
            prob_df.assign(Probability=lambda d: (d["Probability"] * 100).round(1).astype(str) + "%"),
            hide_index=True,
            use_container_width=True,
        )

st.divider()
with st.expander("ℹ️ About this model"):
    st.write(
        """
        - **Algorithm:** RandomForestClassifier (Scikit-learn)
        - **Features used:** Age, Annual Income, Investment Amount, Years of
          Experience, Financial Knowledge Score, Portfolio Diversity Score,
          Debt-to-Income Ratio
        - **Preprocessing:** StandardScaler feature scaling, label encoding
          of the target class
        - **Evaluation:** Train/test split (80/20) plus 5-fold cross-validation
        """
    )
