"""
Flask deployment for the Sarcasm Detector.

Loads the pre-trained pipeline from model.pkl (created by train_model.py)
and serves a simple HTML/CSS form where a user can paste a headline and
get a sarcastic / not-sarcastic prediction, plus a JSON API endpoint.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

from features import add_features

MODEL_PATH = "model.pkl"

app = Flask(__name__)

model = None
model_load_error = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        model_load_error = str(exc)
else:
    model_load_error = (
        f"{MODEL_PATH} not found. Run 'python train_model.py' first to train "
        "and save the model."
    )


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
