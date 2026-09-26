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

import auth
import billing

MODEL_PATH = "model.pkl"

st.set_page_config(
    page_title="Investment Risk Profiler",
    page_icon="📊",
    layout="centered",
)

st.markdown(
    """
    <style>
    .dev-watermark {
        position: fixed;
        top: 8px;
        right: 12px;
        z-index: 9999;
        font-size: 0.75rem;
        background: rgba(255, 255, 255, 0.6);
        padding: 2px 8px;
        border-radius: 6px;
    }
    .dev-watermark a {
        color: rgba(90, 90, 90, 0.85);
        text-decoration: none;
    }
    .dev-watermark a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="dev-watermark">developer&gt;: <a href="mailto:fungaisamanga09@gmail.com">fungaisamanga09@gmail.com</a></div>
    """,
    unsafe_allow_html=True,
)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "subscription_active" not in st.session_state:
    st.session_state.subscription_active = None  # None = not checked yet


def show_login_page():
    st.title("📊 Investment Risk Profiler")
    st.caption("Please log in or create an account to continue.")

    login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)

        if submitted:
            ok, msg = auth.authenticate(username, password)
            if ok:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error(msg)

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", key="signup_username")
            new_password = st.text_input(
                "Choose a password", type="password", key="signup_password"
            )
            confirm_password = st.text_input(
                "Confirm password", type="password", key="signup_confirm"
            )
            signup_submitted = st.form_submit_button(
                "Create Account", type="primary", use_container_width=True
            )

        if signup_submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, msg = auth.register_user(new_username, new_password)
                if ok:
                    st.success(msg + " Switch to the Log In tab to sign in.")
                else:
                    st.error(msg)


if not st.session_state.authenticated:
    show_login_page()
    st.stop()

# --- Handle returning from Stripe Checkout -------------------------------
# When Stripe redirects back after a successful payment, the URL contains
# ?session_id=... . We verify it and attach the resulting Stripe customer
# to this user's account.
query_session_id = st.query_params.get("session_id")
if query_session_id:
    returned_username, customer_id = billing.verify_checkout_session(query_session_id)
    if returned_username and customer_id:
        auth.set_stripe_customer_id(returned_username, customer_id)
        st.session_state.subscription_active = True
        st.query_params.clear()
        st.rerun()

# --- Check subscription status --------------------------------------------
if st.session_state.subscription_active is None:
    customer_id = auth.get_stripe_customer_id(st.session_state.username)
    st.session_state.subscription_active = billing.has_active_subscription(customer_id)


def show_subscribe_page():
    st.title("📊 Investment Risk Profiler")
    st.subheader("Subscribe to continue")
    st.write(
        "This app requires an active subscription. Subscribe below to get "
        "unlimited real-time risk profile predictions."
    )
    if st.button("Subscribe Now", type="primary", use_container_width=True):
        checkout_url = billing.create_checkout_session(st.session_state.username)
        st.link_button(
            "Continue to secure checkout →", checkout_url, use_container_width=True
        )
    if st.button("Log Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.subscription_active = None
        st.rerun()


if not st.session_state.subscription_active:
    show_subscribe_page()
    st.stop()

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.username}**")
    customer_id = auth.get_stripe_customer_id(st.session_state.username)
    if customer_id and st.button("Manage Billing", use_container_width=True):
        portal_url = billing.create_billing_portal_session(customer_id)
        st.link_button("Open Billing Portal →", portal_url, use_container_width=True)
    if st.button("Log Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.subscription_active = None
        st.rerun()


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
