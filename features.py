"""
Shared feature engineering for the Sarcasm Detector.

Both train_model.py (offline training) and app.py (Flask inference)
import add_features() so the exact same columns are computed at
train time and at prediction time.
"""

import re
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four stylistic meta-features used by the model.

    Expects a DataFrame with a 'headline' column. Returns a new
    DataFrame with exclamation_count, question_count, all_caps_count,
    and word_count columns added.
    """
    df = df.copy()
    df["exclamation_count"] = df["headline"].apply(lambda x: x.count("!"))
    df["question_count"] = df["headline"].apply(lambda x: x.count("?"))
    df["all_caps_count"] = df["headline"].apply(
        lambda x: len(re.findall(r"\b[A-Z]{2,}\b", x))
    )
    df["word_count"] = df["headline"].apply(lambda x: len(x.split()))
    return df


TEXT_FEATURE = "headline"
META_FEATURES = ["exclamation_count", "question_count", "all_caps_count", "word_count"]
