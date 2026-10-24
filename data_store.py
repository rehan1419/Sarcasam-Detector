"""
Handles the data the data scientist feeds in through the admin
dashboard. New rows are appended to data/additional_data.csv, which
train_model.py combines with the original datasets on every retrain.
"""

import os
import csv
import pandas as pd

DATA_DIR = "data"
ADDITIONAL_DATA_PATH = os.path.join(DATA_DIR, "additional_data.csv")
COLUMNS = ["headline", "is_sarcastic"]


def _ensure_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ADDITIONAL_DATA_PATH):
        with open(ADDITIONAL_DATA_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)


def count_additional() -> int:
    if not os.path.exists(ADDITIONAL_DATA_PATH):
        return 0
    return len(pd.read_csv(ADDITIONAL_DATA_PATH))


def add_single(headline: str, is_sarcastic: int):
    """Append one labeled headline. Raises ValueError on bad input."""
    headline = (headline or "").strip()
    if not headline:
        raise ValueError("Headline text is required.")
    if is_sarcastic not in (0, 1):
        raise ValueError("Label must be 0 (not sarcastic) or 1 (sarcastic).")

    _ensure_file()
    with open(ADDITIONAL_DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([headline, is_sarcastic])


def add_bulk_csv(file_stream) -> dict:
    """Append rows from an uploaded CSV file-like object.

    Expects columns 'headline' and 'is_sarcastic' (0/1). Returns a
    summary dict: {"added": int, "skipped": int, "errors": [str, ...]}.
    """
    try:
        df = pd.read_csv(file_stream)
    except Exception as exc:  # noqa: BLE001
        return {"added": 0, "skipped": 0, "errors": [f"Could not read CSV: {exc}"]}

    errors = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return {
            "added": 0,
            "skipped": 0,
            "errors": [f"CSV is missing required column(s): {', '.join(missing)}"],
        }

    df = df[COLUMNS].copy()
    # fillna BEFORE astype(str): otherwise a blank cell becomes the
    # literal string "nan" instead of an empty string and slips past
    # the emptiness check below.
    df["headline"] = df["headline"].fillna("").astype(str).str.strip()

    valid_mask = df["headline"].ne("") & df["is_sarcastic"].isin([0, 1])
    valid_rows = df[valid_mask]
    skipped = len(df) - len(valid_rows)
    if skipped:
        errors.append(
            f"Skipped {skipped} row(s) with a missing headline or a label "
            "that wasn't exactly 0 or 1."
        )

    if len(valid_rows):
        _ensure_file()
        valid_rows.to_csv(ADDITIONAL_DATA_PATH, mode="a", header=False, index=False)

    return {"added": int(len(valid_rows)), "skipped": int(skipped), "errors": errors}
