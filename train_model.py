"""
Trains the sarcasm-detection pipeline on the ORIGINAL datasets combined
with any additional labeled headlines the data scientist has fed in
through the admin dashboard (data/additional_data.csv).

Every retrain uses the full accumulated dataset (original + all
additional data collected so far) — the model is not fine-tuned
incrementally on just the new rows, it is re-fit from scratch on
everything, which is the correct way to retrain a TF-IDF + LinearSVC
pipeline (the vocabulary itself can shift as new headlines come in).

Can be run standalone:
    python train_model.py

Or imported and called from the Flask admin route:
    from train_model import train
    result = train()
"""

import os
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from features import add_features, TEXT_FEATURE, META_FEATURES

MODEL_PATH = "model.pkl"
METADATA_PATH = "data/metadata.json"
ADDITIONAL_DATA_PATH = "data/additional_data.csv"


ORIGINAL_DATASET_FILES = [
    "Sarcasm_Headlines_Dataset.json",
    "Sarcasm_Headlines_Dataset_v2.json",
]


def _read_original_dataset(path: str) -> pd.DataFrame:
    """Read one of the original JSON-lines dataset files, with a clear
    error if the file is missing/empty/not-actually-JSON instead of a
    cryptic pandas ValueError.

    This is the #1 cause of "retraining failed" in the admin dashboard:
    prediction only needs model.pkl, but retraining needs these original
    ~5-6 MB dataset files physically present wherever the app is running.
    If you deployed without them (a size limit, a missed `git add`, a
    platform that doesn't ship large files by default, etc.) retraining
    will fail with exactly this kind of error even though the site's
    predictions keep working fine.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required dataset file '{path}' was not found next to app.py on this "
            "server. Retraining needs the original dataset files present, even "
            "though everyday predictions only need model.pkl. Make sure this file "
            "was actually deployed (check it's committed to git, not caught by "
            ".gitignore, and not skipped by a file-size limit on your host)."
        )
    if os.path.getsize(path) == 0:
        raise ValueError(f"Dataset file '{path}' is present but empty (0 bytes) — re-upload it.")

    try:
        return pd.read_json(path, lines=True)
    except ValueError as exc:
        raise ValueError(
            f"Dataset file '{path}' exists but isn't valid JSON-lines content "
            f"(got: {exc}). It may have been replaced by an error page, a Git LFS "
            "pointer file, or a partial/corrupted upload — re-download the original "
            "file from the repo and redeploy it."
        ) from exc


def load_combined_dataset():
    """Load the two original JSON datasets plus any additional_data.csv rows."""
    df1 = _read_original_dataset(ORIGINAL_DATASET_FILES[0])
    df2 = _read_original_dataset(ORIGINAL_DATASET_FILES[1])
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[["headline", "is_sarcastic"]]

    additional_count = 0
    if os.path.exists(ADDITIONAL_DATA_PATH):
        extra = pd.read_csv(ADDITIONAL_DATA_PATH)
        extra = extra[["headline", "is_sarcastic"]]
        additional_count = len(extra)
        df = pd.concat([df, extra], ignore_index=True)

    df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    return df, additional_count


def train(save_path: str = MODEL_PATH) -> dict:
    """Train on the full combined dataset and save the pipeline.

    Returns a metadata dict describing the run (also written to
    data/metadata.json) so the admin dashboard can display it.
    """
    df, additional_count = load_combined_dataset()
    df = add_features(df)

    target = "is_sarcastic"
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000),
                TEXT_FEATURE,
            ),
            ("meta", StandardScaler(), META_FEATURES),
        ],
        remainder="drop",
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LinearSVC(random_state=42)),
    ])

    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    acc = accuracy_score(y_test, test_predictions)

    joblib.dump(model, save_path)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "total_headlines": len(df),
        "additional_headlines": additional_count,
        "test_accuracy": round(float(acc), 4),
    }
    os.makedirs("data", exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def get_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    print("Loading combined dataset (original + any additional data)...")
    result = train()
    print(f"Total headlines used: {result['total_headlines']}")
    print(f"  of which added via admin dashboard: {result['additional_headlines']}")
    print(f"Final Model Accuracy on Test Set: {result['test_accuracy']:.4f}")
    print(f"Saved trained pipeline to {MODEL_PATH}")
