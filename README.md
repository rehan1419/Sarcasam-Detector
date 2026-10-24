# Sarcasm Detector

A machine learning model that classifies news headlines as sarcastic or not, built with scikit-learn using TF-IDF text features combined with engineered punctuation/style features.

## Overview

This project trains and compares two classifiers — Logistic Regression and Linear SVM — to detect sarcasm in short text headlines. It combines TF-IDF vectorization of the headline text with handcrafted stylistic features (exclamation marks, question marks, all-caps words, word count) to capture both semantic and surface-level cues often present in sarcastic writing.

## Dataset

The model is trained on the combined **News Headlines Dataset for Sarcasm Detection** (v1 + v2), sourced from headlines scraped from *TheOnion* (sarcastic) and *HuffPost* (non-sarcastic):

- `Sarcasm_Headlines_Dataset.json` — ~26.7K labeled headlines
- `Sarcasm_Headlines_Dataset_v2.json` — ~28.6K labeled headlines

Each record contains a `headline` (text) and `is_sarcastic` (0 or 1) label. The two files are concatenated and de-duplicated before training.

## Approach

1. **Data loading & cleaning** — merge both JSON datasets, keep only `headline` and `is_sarcastic`, drop duplicates.
2. **Feature engineering** — derive four stylistic signals from each headline:
   - Exclamation mark count
   - Question mark count
   - All-caps word count
   - Word count
3. **Train/test split** — 80/20 split, stratified on the target label.
4. **Preprocessing pipeline** — a `ColumnTransformer` combining:
   - `TfidfVectorizer` on the headline text (unigrams + bigrams, English stop words removed, top 5,000 features)
   - `StandardScaler` on the four numeric meta-features
5. **Model comparison** — 5-fold cross-validation on the training set:
   - Logistic Regression (`liblinear` solver)
   - Linear SVM (`LinearSVC`)
6. **Final model** — the Linear SVM pipeline is trained on the full training set and evaluated on the held-out test set.
7. **Output** — predictions on the test set are saved to `Predictions.csv` (headline + predicted label).

## Results

| Model | 5-Fold CV Accuracy |
|---|---|
| Logistic Regression | ~78.3% |
| Linear SVM | ~77.8% |

**Final model (Linear SVM) test set accuracy: ~78.8%**

## Repository Structure

```
Sarcasam-Detector/
├── sarcasm_detector.ipynb           # Main notebook: data prep, feature engineering, training, evaluation
├── Sarcasm_Headlines_Dataset.json   # Dataset v1
├── Sarcasm_Headlines_Dataset_v2.json # Dataset v2
├── Predictions.csv                  # Model predictions on the test set
└── Report.pdf                       # Written report/summary of the project
```

## Tech Stack

- **pandas / numpy** — data loading and manipulation
- **scikit-learn** — `TfidfVectorizer`, `ColumnTransformer`, `Pipeline`, `LogisticRegression`, `LinearSVC`, cross-validation, metrics
- **re** — regex-based feature extraction (all-caps detection)

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/rehan1419/Sarcasam-Detector.git
cd Sarcasam-Detector
pip install pandas numpy scikit-learn jupyter
```

### Running

```bash
jupyter notebook sarcasm_detector.ipynb
```

Run all cells in order. The notebook will:
1. Load and combine both dataset files
2. Engineer features and split the data
3. Cross-validate both models and print their scores
4. Train the final model and print test accuracy
5. Write `Predictions.csv` with predictions on the test set

## Future Improvements

- Try additional classifiers (e.g. Random Forest, Gradient Boosting, or a fine-tuned transformer model)
- Expand feature set with sentiment scores, POS tag patterns, or embeddings
- Hyperparameter tuning via `GridSearchCV`

## Deployment (Flask + HTML/CSS)

The notebook logic has been turned into a small web app — **"The Sarcasm Gazette"** — so anyone can paste a headline into a form and get a live prediction, no notebook required.

### New files

```
Sarcasam-Detector/
├── features.py            # Shared feature engineering (used by training + app)
├── train_model.py         # Trains the pipeline from the JSON datasets, saves model.pkl
├── model.pkl              # Pre-trained pipeline (checked in, ~220 KB)
├── app.py                 # Flask app: form route, JSON API route, health check
├── templates/
│   └── index.html         # Newspaper-styled form + result page
├── static/
│   └── style.css          # All styling (newsprint theme, "stamp" verdict)
├── requirements.txt
├── Procfile                # For Heroku/Render-style process managers
└── .gitignore
```

### Run locally

```bash
git clone https://github.com/rehan1419/Sarcasam-Detector.git
cd Sarcasam-Detector
pip install -r requirements.txt

# Only needed if you want to retrain instead of using the checked-in model.pkl
python train_model.py

python app.py
```

Visit `http://127.0.0.1:5000` in your browser, type a headline, and click **"Run the desk check"**.

### Routes

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Renders the HTML form |
| `/predict` | POST (form) | Runs a prediction, re-renders the page with the verdict |
| `/api/predict` | POST (JSON `{"headline": "..."}`) | Returns `{"headline", "is_sarcastic", "score"}` as JSON |
| `/health` | GET | Returns `200 {"status": "ok"}` if the model is loaded, `503` otherwise |

Example API call:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"headline": "Local Man Discovers Motivation, Loses It Again By Noon"}'
```

### Deploying to a host (Render / Railway / Heroku-style)

1. Push this repo (including `model.pkl`) to GitHub.
2. Create a new **Web Service** pointing at the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app` (already set as the `Procfile` default).
5. No environment variables are required. The app reads `PORT` from the environment automatically, which most PaaS providers set for you.

If you'd rather train the model at build time instead of committing `model.pkl`, add `python train_model.py` as a pre-start/build step — `app.py` will report a clear "presses are down" message on the page if `model.pkl` is missing.

## The Newsroom Desk (data scientist login + continual learning)

There's a login-gated admin area where a data scientist can feed the model more labeled headlines and retrain it — without touching the code or a Jupyter notebook.

**Where:** `/login` (also linked at the bottom of the homepage as "Newsroom login"). Once signed in you land on `/admin`.

**What it does:**
- **Add one headline** — type a headline, mark it Sarcastic / Not sarcastic, and it's appended to `data/additional_data.csv`.
- **Upload a batch (CSV)** — upload a CSV with `headline` and `is_sarcastic` (0/1) columns; valid rows are appended, invalid rows (blank headline, a label that isn't 0/1) are reported and skipped.
- **Retrain on combined data** — re-fits the pipeline from scratch on the **original datasets + everything added from the desk so far** (every past retrain's data stays in the mix — nothing is thrown away between retrains). The dashboard shows how many headlines are in the live model, the last test accuracy, and when it was last retrained. As soon as retraining finishes, the running app swaps in the new `model.pkl` automatically — no restart needed.

Note on "combined learning": TF‑IDF + LinearSVC can't be updated incrementally the way a neural net can (the vocabulary itself depends on the whole corpus), so "learns from previous and current data combined" is implemented as a full retrain over the accumulated dataset each time, not a partial/incremental fit. This is the statistically correct way to do it for this model type, and the accumulated data is never discarded — each retrain just starts from the full history plus whatever's new.

### Login credentials

Credentials come from environment variables — nothing is hardcoded in the repo:

| Variable | Purpose | Default |
|---|---|---|
| `DS_USERNAME` | Data scientist username | `datascientist` |
| `DS_PASSWORD` | Data scientist password | none — see below |
| `FLASK_SECRET_KEY` | Signs the login session cookie | auto-generated once and saved to `data/.flask_secret_key` if not set, so it survives restarts. Set this explicitly in production instead. |
| `FORCE_SECURE_COOKIES` | Set to `1` once you're serving over HTTPS, so the session cookie is marked `Secure` | `0` |

If `DS_PASSWORD` isn't set, the app generates a random password once at startup and **prints it to the console** so local/demo use works out of the box:

```
============================================================
No DS_PASSWORD set — generated a temporary newsroom login:
  username: datascientist
  password: <random>
Set DS_USERNAME / DS_PASSWORD env vars for a real deployment.
============================================================
```

For any real deployment, set `DS_USERNAME`, `DS_PASSWORD`, `FLASK_SECRET_KEY`, and `FORCE_SECURE_COOKIES=1` as environment variables on your host (Render/Railway/Heroku all have an "Environment Variables" section in the dashboard) rather than relying on the generated ones.

### How the login is secured

Being logged in once, and then jumping straight to `/admin` without seeing the login form again, is expected — that's what a session cookie is for. The things that actually needed hardening, and are now in place:

- **Sessions expire.** A login is only good for 20 minutes of inactivity (`PERMANENT_SESSION_LIFETIME`), not forever. Any request resets the clock, so an active session doesn't drop mid-task.
- **Cookies are locked down.** The session cookie is `HttpOnly` (JavaScript can't read it) and `SameSite=Lax` (other sites can't ride along on it). Set `FORCE_SECURE_COOKIES=1` once you're on HTTPS so it's also `Secure`.
- **CSRF tokens on every state-changing form** — login, logout, add headline, upload CSV, retrain. A request with a missing or wrong token gets a `400`, so another site can't submit these on the data scientist's behalf.
- **Login attempts are rate-limited.** 5 wrong passwords from the same IP locks that IP out for 5 minutes (`429 Too Many Requests`), even if the 6th attempt has the correct password.
- **No open redirect.** `/login?next=...` only ever redirects to a local path on this same app — a crafted link can't use the login page to bounce someone to an external site after they type their password.
- **Session fixation is cleared on login.** The session is wiped and rebuilt the moment a login succeeds, so nothing set beforehand (by an attacker sharing a pre-set session, for instance) carries over.
- **The secret key persists.** If `FLASK_SECRET_KEY` isn't set, one is generated once and saved to `data/.flask_secret_key` rather than regenerated every restart — otherwise every restart would silently invalidate all sessions in a confusing way (or, worse, several app workers would each mint their own key and reject each other's cookies).

What this doesn't cover, if you take this further: the login rate-limit and CSRF token store are both in-memory, so they reset if the process restarts and won't be shared across multiple worker processes/machines — fine for a single small deployment, but swap in something like Redis if you scale out to several workers.

### New files this adds

```
├── auth.py                 # Login, CSRF protection, rate limiting, secret-key persistence
├── data_store.py            # Appends/validates rows into data/additional_data.csv
├── templates/login.html     # Newsroom sign-in page
├── templates/admin.html      # Dashboard: add data, upload CSV, retrain, view stats
└── data/
    ├── additional_data.csv  # Headlines fed in via the desk (git-ignored, grows over time)
    ├── metadata.json         # Stats from the last retrain (git-ignored, regenerated each retrain)
    └── .flask_secret_key     # Auto-generated session signing key (git-ignored)
```
