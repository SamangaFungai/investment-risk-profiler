"""
auth.py
-------
Simple username/password authentication for the Streamlit app.

Users are stored in a local JSON file (users.json) with bcrypt-hashed
passwords. This is intentionally lightweight — good enough for a personal
project or portfolio demo, but NOT a substitute for a real auth provider
(e.g. Auth0, Firebase Auth, or a proper database + session management) in
a production app with real users or sensitive data.

Note on Streamlit Community Cloud: the filesystem is ephemeral. Accounts
created will persist while the app stays "awake", but a full reboot/redeploy
can reset users.json. For anything long-lived, swap load_users/save_users
to read/write from a real database instead of a local file.
"""

import json
import os
import re

import bcrypt

USERS_FILE = "users.json"


def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def validate_username(username: str):
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r"^[A-Za-z0-9_]+$", username):
        return False, "Username can only contain letters, numbers, and underscores."
    return True, ""


def validate_password(password: str):
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."
    return True, ""


def register_user(username: str, password: str):
    ok, msg = validate_username(username)
    if not ok:
        return False, msg
    ok, msg = validate_password(password)
    if not ok:
        return False, msg

    users = load_users()
    if username in users:
        return False, "That username is already taken."

    users[username] = {"password_hash": hash_password(password)}
    save_users(users)
    return True, "Account created! You can now log in."


def authenticate(username: str, password: str):
    users = load_users()
    if username not in users:
        return False, "Incorrect username or password."
    if not verify_password(password, users[username]["password_hash"]):
        return False, "Incorrect username or password."
    return True, ""


def get_stripe_customer_id(username: str):
    users = load_users()
    return users.get(username, {}).get("stripe_customer_id")


def set_stripe_customer_id(username: str, customer_id: str) -> None:
    users = load_users()
    if username in users:
        users[username]["stripe_customer_id"] = customer_id
        save_users(users)
