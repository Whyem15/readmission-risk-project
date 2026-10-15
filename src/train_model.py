"""
30-Day Hospital Readmission Risk — Training Pipeline

Downloads the UCI Diabetes 130-US Hospitals dataset, cleans it with an
explicit leakage check, trains a baseline and two candidate models, and
saves the chosen model for deployment via app.py.

Run: python train_model.py
"""

import zipfile
import urllib.request

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)


def download_data() -> pd.DataFrame:
    """Download and extract the UCI diabetes readmission dataset."""
    urllib.request.urlretrieve(DATA_URL, "diabetes.zip")
    with zipfile.ZipFile("diabetes.zip") as z:
        z.extractall("diabetes_data")
    return pd.read_csv("diabetes_data/diabetic_data.csv", na_values="?")


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean columns, define the binary target, and deduplicate by patient.

    The dedup step is the critical leakage fix: ~30K rows in the raw data
    belong to patients with more than one hospital encounter. Without
    collapsing to one row per patient, the same patient could appear in
    both train and test, inflating evaluation scores in a way that would
    not hold up on genuinely new patients.
    """
    df = df.drop(
        columns=["weight", "payer_code", "medical_specialty", "encounter_id"]
    )

    df["target"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Keep each patient's first encounter only — the leakage fix
    df = df.sort_values("patient_nbr").drop_duplicates(
        subset="patient_nbr", keep="first"
    )

    return df


def build_features(df: pd.DataFrame):
    """Split into X/y and one-hot encode categorical columns."""
    y = df["target"]
    X = df.drop(columns=["target", "patient_nbr"])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def evaluate(name: str, model, X_test, y_test) -> float:
    """Print PR-AUC and classification report; return PR-AUC for comparison."""
    proba = model.predict_proba(X_test)[:, 1]
    score = average_precision_score(y_test, proba)
    print(f"\n{name} PR-AUC: {score:.4f}")
    print(classification_report(y_test, model.predict(X_test)))
    return score


def main():
    print("Downloading data...")
    df = download_data()
    print(f"Raw shape: {df.shape}")

    df = clean_and_prepare(df)
    print(f"After cleaning + leakage fix: {df.shape}")
    print(df["target"].value_counts(normalize=True))

    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Baseline: the floor any real model has to beat
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    evaluate("Baseline (majority class)", baseline, X_test, y_test)

    # Candidate 1: Logistic Regression
    logreg = LogisticRegression(max_iter=3000, class_weight="balanced")
    logreg.fit(X_train, y_train)
    logreg_score = evaluate("Logistic Regression", logreg, X_test, y_test)

    # Candidate 2: Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_score = evaluate("Random Forest", rf, X_test, y_test)

    # Pick the better model by PR-AUC (ties favor the simpler model)
    final_model = logreg if logreg_score >= rf_score else rf
    final_name = "Logistic Regression" if final_model is logreg else "Random Forest"
    print(f"\nSelected final model: {final_name}")

    joblib.dump(final_model, "readmission_model.pkl")
    joblib.dump(X_train.columns.tolist(), "model_columns.pkl")
    print("Saved readmission_model.pkl and model_columns.pkl")


if __name__ == "__main__":
    main()
