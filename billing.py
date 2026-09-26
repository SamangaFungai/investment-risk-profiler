"""
billing.py
----------
Stripe integration for recurring subscriptions.

How it works (no webhook server needed — works fine on Streamlit Cloud):
  1. A logged-in user without an active subscription sees a "Subscribe"
     screen with a Stripe Checkout button.
  2. Checkout redirects back to the app with a `session_id` in the URL.
     We verify that session directly with Stripe's API and store the
     resulting `stripe_customer_id` on the user's account (in users.json,
     via auth.py).
  3. On future visits, we ask Stripe directly "does this customer have an
     active subscription right now?" — so renewals, failed payments, and
     cancellations (even ones made through Stripe's own billing portal)
     are always reflected without needing a webhook endpoint.

Required secrets (see .streamlit/secrets.toml.example):
  STRIPE_SECRET_KEY  - your Stripe secret key (sk_test_... or sk_live_...)
  STRIPE_PRICE_ID    - the recurring Price ID to charge (price_...)
  APP_URL            - the public URL of this deployed app, e.g.
                        https://your-app-name.streamlit.app
"""

import streamlit as st
import stripe

stripe.api_key = st.secrets.get("STRIPE_SECRET_KEY", "")
PRICE_ID = st.secrets.get("STRIPE_PRICE_ID", "")
APP_URL = st.secrets.get("APP_URL", "http://localhost:8501").rstrip("/")


def create_checkout_session(username: str, customer_email: str | None = None) -> str:
    """Creates a Stripe Checkout Session for a recurring subscription and
    returns the URL to redirect the user to."""
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": PRICE_ID, "quantity": 1}],
        client_reference_id=username,  # lets us map the payment back to our user
        customer_email=customer_email or None,
        success_url=f"{APP_URL}/?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=APP_URL,
    )
    return session.url


def verify_checkout_session(session_id: str):
    """Confirms a completed Checkout Session and returns
    (username, stripe_customer_id) so the caller can save it against the
    right account. Returns (None, None) if the session isn't valid/paid."""
    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except stripe.error.StripeError:
        return None, None

    if session.payment_status != "paid" and session.status != "complete":
        return None, None

    username = session.client_reference_id
    customer_id = session.customer
    return username, customer_id


def has_active_subscription(customer_id: str) -> bool:
    """Asks Stripe directly whether this customer currently has an active
    (or trialing) subscription."""
    if not customer_id:
        return False
    try:
        subs = stripe.Subscription.list(
            customer=customer_id, status="all", limit=10
        )
    except stripe.error.StripeError:
        return False

    for sub in subs.auto_paging_iter():
        if sub.status in ("active", "trialing"):
            return True
    return False


def create_billing_portal_session(customer_id: str) -> str:
    """Returns a URL to Stripe's hosted billing portal, where the user can
    update their card, view invoices, or cancel their subscription."""
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=APP_URL,
    )
    return session.url
