

import json
import hashlib
import secrets
import sys
from pathlib import Path

import streamlit as st

USERS_FILE = Path(__file__).parent / "users.json"


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def add_or_update_user(username: str, password: str, role: str = "operator") -> None:
    """Create or overwrite a user's credentials with a freshly salted hash."""
    users = _load_users()
    salt = secrets.token_hex(16)
    users[username] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
        "role": role,
        "display_name": username.capitalize(),
    }
    _save_users(users)


def _verify_credentials(username: str, password: str) -> dict | None:
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    if _hash_password(password, user["salt"]) == user["password_hash"]:
        return user
    return None


def is_authenticated() -> bool:
    return bool(st.session_state.get("authenticated"))


def current_user() -> dict:
    return {
        "username": st.session_state.get("username"),
        "display_name": st.session_state.get("display_name"),
        "role": st.session_state.get("role"),
    }


def logout() -> None:
    for key in ("authenticated", "username", "display_name", "role"):
        st.session_state.pop(key, None)
    st.rerun()


def require_login() -> None:
    """
    Call this once, at the very top of app.py (right after st.set_page_config).
    If the user isn't logged in yet, it renders a login form and stops the
    script — nothing below this call runs until they log in successfully.
    """
    if is_authenticated():
        return

    st.markdown(
        "<h1 style='text-align:center;'>🚦 AI Traffic Accident Monitoring System</h1>"
        "<h4 style='text-align:center;'>Please sign in to continue</h4>",
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            user = _verify_credentials(username.strip(), password)
            if user:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username.strip()
                st.session_state["display_name"] = user["display_name"]
                st.session_state["role"] = user["role"]
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    st.stop()


if __name__ == "__main__":
    # CLI helper: python auth.py add <username> <password> [role]
    if len(sys.argv) >= 4 and sys.argv[1] == "add":
        uname, pw = sys.argv[2], sys.argv[3]
        role = sys.argv[4] if len(sys.argv) > 4 else "operator"
        add_or_update_user(uname, pw, role)
        print(f"Saved/updated user '{uname}' with role '{role}'.")
    else:
        print("Usage: python auth.py add <username> <password> [role]")
