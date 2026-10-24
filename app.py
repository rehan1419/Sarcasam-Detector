"""
Flask deployment for the Sarcasm Detector.

Loads the pre-trained pipeline from model.pkl (created by train_model.py)
and serves a simple HTML/CSS form where a user can paste a headline and
get a sarcastic / not-sarcastic prediction, plus a JSON API endpoint.
"""

import os
from datetime import timedelta

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash

from features import add_features
import auth
import data_store
import train_model

MODEL_PATH = "model.pkl"

app = Flask(__name__)
app.secret_key = auth.get_secret_key()

# Session hardening: sessions expire after a sliding window of
# inactivity, cookies aren't readable by JS, and (in production, over
# HTTPS) aren't sent over plain HTTP.
app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=20),
    SESSION_REFRESH_EACH_REQUEST=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FORCE_SECURE_COOKIES", "0") == "1",
)


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": auth.get_csrf_token}


if auth.GENERATED_PASSWORD:
    print("=" * 60)
    print("No DS_PASSWORD set — generated a temporary newsroom login:")
    print(f"  username: {auth.DS_USERNAME}")
    print(f"  password: {auth.GENERATED_PASSWORD}")
    print("Set DS_USERNAME / DS_PASSWORD env vars for a real deployment.")
    print("=" * 60)

model = None
model_load_error = None


def load_model():
    """(Re)load model.pkl from disk into the module-level `model`.

    Called at startup, and again by the admin retrain route so a fresh
    model is served immediately without restarting the process.
    """
    global model, model_load_error
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            model_load_error = None
        except Exception as exc:  # noqa: BLE001
            model_load_error = str(exc)
    else:
        model_load_error = (
            f"{MODEL_PATH} not found. Run 'python train_model.py' first to train "
            "and save the model."
        )


load_model()


def predict_headline(headline: str):
    """Run the full feature pipeline + model on a single headline string."""
    df = pd.DataFrame({"headline": [headline]})
    df = add_features(df)
    prediction = model.predict(df)[0]
    # LinearSVC has no predict_proba, but decision_function gives a
    # signed distance from the boundary we can show as a rough confidence.
    score = float(model.decision_function(df)[0])
    return int(prediction), score


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_error=model_load_error)


@app.route("/predict", methods=["POST"])
def predict():
    headline = (request.form.get("headline") or "").strip()

    if model_load_error:
        return render_template(
            "index.html", model_error=model_load_error, headline=headline
        )

    if not headline:
        return render_template(
            "index.html",
            error="Please enter a headline to analyze.",
            headline=headline,
        )

    prediction, score = predict_headline(headline)
    label = "Sarcastic 😏" if prediction == 1 else "Not Sarcastic 🙂"

    return render_template(
        "index.html",
        headline=headline,
        label=label,
        is_sarcastic=bool(prediction),
        score=round(score, 3),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: POST {"headline": "..."} -> {"is_sarcastic": 0/1, "score": ...}"""
    if model_load_error:
        return jsonify({"error": model_load_error}), 503

    data = request.get_json(silent=True) or {}
    headline = (data.get("headline") or "").strip()
    if not headline:
        return jsonify({"error": "Field 'headline' is required."}), 400

    prediction, score = predict_headline(headline)
    return jsonify({
        "headline": headline,
        "is_sarcastic": prediction,
        "score": round(score, 3),
    })


@app.route("/health", methods=["GET"])
def health():
    status = "ok" if model is not None else "model_not_loaded"
    return jsonify({"status": status}), (200 if model is not None else 503)


# ---------------------------------------------------------------------
# Newsroom (data scientist) area: login-gated feeding of new training
# data and retraining on the combined (original + additional) dataset.
# ---------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if auth.is_locked_out():
            return render_template(
                "login.html",
                error="Too many failed attempts. Please wait a few minutes and try again.",
            ), 429

        auth.csrf_protect()

        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if auth.verify_login(username, password):
            auth.clear_failed_attempts()
            session.clear()  # drop anything set before login (session fixation)
            session["logged_in"] = True
            session["username"] = username
            session.permanent = True
            next_url = auth.safe_next_url(request.args.get("next", ""))
            return redirect(next_url)

        auth.record_failed_attempt()
        return render_template("login.html", error="Incorrect username or password.")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    auth.csrf_protect()
    session.clear()
    return redirect(url_for("login"))


@app.route("/admin", methods=["GET"])
@auth.login_required
def admin():
    return render_template(
        "admin.html",
        username=session.get("username"),
        additional_count=data_store.count_additional(),
        metadata=train_model.get_metadata(),
        model_error=model_load_error,
    )


@app.route("/admin/add", methods=["POST"])
@auth.login_required
def admin_add():
    auth.csrf_protect()
    headline = request.form.get("headline", "")
    label_raw = request.form.get("is_sarcastic")
    try:
        is_sarcastic = int(label_raw)
        data_store.add_single(headline, is_sarcastic)
        flash(f"Added headline to the training set: \u201c{headline.strip()}\u201d", "success")
    except (TypeError, ValueError) as exc:
        flash(str(exc), "error")
    return redirect(url_for("admin"))


@app.route("/admin/upload", methods=["POST"])
@auth.login_required
def admin_upload():
    auth.csrf_protect()
    uploaded = request.files.get("csv_file")
    if not uploaded or uploaded.filename == "":
        flash("Please choose a CSV file to upload.", "error")
        return redirect(url_for("admin"))

    result = data_store.add_bulk_csv(uploaded.stream)
    if result["added"]:
        flash(
            f"Added {result['added']} headline(s) from '{uploaded.filename}' to the training set.",
            "success",
        )
    for err in result["errors"]:
        flash(err, "error")
    if not result["added"] and not result["errors"]:
        flash("No valid rows found in that file.", "error")
    return redirect(url_for("admin"))


@app.route("/admin/retrain", methods=["POST"])
@auth.login_required
def admin_retrain():
    auth.csrf_protect()
    try:
        metadata = train_model.train()
        load_model()
        flash(
            f"Retrained on {metadata['total_headlines']} headlines "
            f"({metadata['additional_headlines']} from the newsroom desk) — "
            f"test accuracy {metadata['test_accuracy']:.2%}.",
            "success",
        )
    except Exception as exc:  # noqa: BLE001
        flash(f"Retraining failed: {exc}", "error")
    return redirect(url_for("admin"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
