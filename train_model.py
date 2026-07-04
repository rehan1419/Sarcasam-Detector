"""
Trains the sarcasm-detection pipeline (same steps as sarcasm_detector.ipynb)
and saves the fitted pipeline to model.pkl for the Flask app to load.

Run once before starting the app:
    python train_model.py
"""

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


def main():
    print("Loading datasets...")
    df1 = pd.read_json("Sarcasm_Headlines_Dataset.json", lines=True)
    df2 = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[["headline", "is_sarcastic"]].drop_duplicates().reset_index(drop=True)
    print(f"Total headlines loaded: {len(df)}")

    print("Engineering features...")
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

    print("Training final model (Linear SVM)...")
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    acc = accuracy_score(y_test, test_predictions)
    print(f"Final Model Accuracy on Test Set: {acc:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
