from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

NUMERIC = [
    "policyholder_tenure_years",
    "previous_claims_count",
    "claim_amount",
    "deductible",
    "days_between_incident_and_filing",
]
CATEGORICAL = ["claim_type", "state"]

def build_model():
    # HistGradientBoosting does not accept sparse OHE output, so use a dense
    # preprocessing pipeline for this moderate-size benchmark.
    prep = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42,
    )
    return Pipeline([("prep", prep), ("model", clf)])

def train_model(df):
    X = df[NUMERIC + CATEGORICAL]
    y = df["is_fraud_flagged_ground_truth"].astype(int)
    model = build_model()
    model.fit(X, y)
    return model

def save_model(model, path="artifacts/model.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path="artifacts/model.joblib"):
    return joblib.load(path)
