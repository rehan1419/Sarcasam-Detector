"""
Auth for the "newsroom" admin area, where the data scientist logs in to
feed the model more data and trigger retraining.

Security measures implemented here (all necessary because /admin guards
a route that can add training data and retrain the live model):

1. Credentials from environment variables — nothing secret in the repo.
2. A persistent Flask secret key (data/.flask_secret_key) so sessions
   don't behave unpredictably across restarts / multiple gunicorn
   workers when FLASK_SECRET_KEY isn't set explicitly.
3. Session cookies expire after a short sliding window of inactivity
   (see SESSION_LIFETIME_MINUTES, wired up in app.py) — being logged in
   once does not mean being logged in forever.
4. The session is cleared before a successful login is written, so a
   session fixed by an attacker before login can't carry over.
5. A CSRF token is required on every state-changing admin form (login,
   add data, upload data, retrain, logout) so another site can't submit
   these on the data scientist's behalf.
6. Failed login attempts are rate-limited per source IP with a lockout
   window, to blunt password guessing.

Env vars:
    DS_USERNAME        Data scientist username (default: "datascientist")
    DS_PASSWORD         Data scientist password — set this explicitly in
                        production. If unset, a random one is generated
                        at startup and printed to the console once.
    FLASK_SECRET_KEY    Signs session cookies — set this explicitly in
                        production, or sessions won't survive a restart.
"""

import os
import time
import secrets
from functools import wraps

from flask import session, redirect, url_for, request, abort
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------

DS_USERNAME = os.environ.get("DS_USERNAME", "datascientist")

_env_password = os.environ.get("DS_PASSWORD")
GENERATED_PASSWORD = None
if not _env_password:
    GENERATED_PASSWORD = secrets.token_urlsafe(9)
    _env_password = GENERATED_PASSWORD

DS_PASSWORD_HASH = generate_password_hash(_env_password)

# ---------------------------------------------------------------------
# Persistent secret key (so sessions survive restarts / multiple
# workers when FLASK_SECRET_KEY isn't provided via the environment)
# ---------------------------------------------------------------------

_SECRET_KEY_PATH = os.path.join("data", ".flask_secret_key")


def get_secret_key() -> str:
    env_key = os.environ.get("FLASK_SECRET_KEY")
    if env_key:
        return env_key

    os.makedirs("data", exist_ok=True)
    if os.path.exists(_SECRET_KEY_PATH):
        with open(_SECRET_KEY_PATH) as f:
            key = f.read().strip()
            if key:
                return key

    key = secrets.token_hex(32)
    with open(_SECRET_KEY_PATH, "w") as f:
        f.write(key)
    return key


# ---------------------------------------------------------------------
# Login rate limiting (per source IP)
# ---------------------------------------------------------------------

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_WINDOW_SECONDS = 300  # 5 minutes

_failed_attempts = {}  # ip -> [timestamps of recent failures]


def _client_ip() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()


def is_locked_out() -> bool:
    ip = _client_ip()
    now = time.time()
    attempts = [t for t in _failed_attempts.get(ip, []) if now - t < LOCKOUT_WINDOW_SECONDS]
    _failed_attempts[ip] = attempts
    return len(attempts) >= MAX_FAILED_ATTEMPTS


def record_failed_attempt():
    ip = _client_ip()
    _failed_attempts.setdefault(ip, []).append(time.time())


def clear_failed_attempts():
    _failed_attempts.pop(_client_ip(), None)


def verify_login(username: str, password: str) -> bool:
    return username == DS_USERNAME and check_password_hash(DS_PASSWORD_HASH, password)


# ---------------------------------------------------------------------
# Session / CSRF
# ---------------------------------------------------------------------

def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapped


def get_csrf_token() -> str:
    """Return the current session's CSRF token, creating one if needed."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(16)
    return session["csrf_token"]


def csrf_protect():
    """Call at the top of any state-changing POST route. Aborts with 400
    if the submitted csrf_token doesn't match the session's token."""
    submitted = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    if not expected or not secrets.compare_digest(submitted, expected):
        abort(400, description="Invalid or missing CSRF token. Please reload the page and try again.")


def safe_next_url(candidate: str, default_endpoint: str = "admin") -> str:
    """Only allow redirecting to a local path after login.

    Without this, `next` is attacker-controlled input (?next=https://evil.example)
    and a crafted login link could send an authenticated data scientist
    somewhere else right after they type their password in. A safe value
    must be a single relative path — no scheme, no host, and not a
    protocol-relative '//host' path.
    """
    if (
        candidate
        and candidate.startswith("/")
        and not candidate.startswith("//")
        and "://" not in candidate
    ):
        return candidate
    return url_for(default_endpoint)
