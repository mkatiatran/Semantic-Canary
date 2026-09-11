from pathlib import Path
import pandas as pd
import numpy as np

NUMERIC = [
    "policyholder_tenure_years",
    "previous_claims_count",
    "claim_amount",
    "deductible",
]
CATEGORICAL = ["claim_type", "state"]

def load_claims(rows=100000, seed=42, local_path=None):
    if local_path:
        df = pd.read_parquet(local_path)
    else:
        from datasets import load_dataset
        ds = load_dataset(
            "ziadatalabs/FreeInsuranceClaims100M",
            split="train",
            streaming=True,
        )
        sample = []
        for i, row in enumerate(ds):
            sample.append(row)
            if i + 1 >= rows:
                break
        df = pd.DataFrame(sample)

    df["incident_date"] = pd.to_datetime(df["incident_date"])
    df["claim_filed_date"] = pd.to_datetime(df["claim_filed_date"])
    df["days_between_incident_and_filing"] = (
        df["claim_filed_date"] - df["incident_date"]
    ).dt.days

    # Only information plausibly available at claim filing is used.
    keep = NUMERIC + CATEGORICAL + [
        "days_between_incident_and_filing",
        "is_fraud_flagged_ground_truth",
        "incident_date",
        "claim_filed_date",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.dropna(subset=["is_fraud_flagged_ground_truth"])
    return df

def split_reference_current(df, fraction=0.5):
    cut = int(len(df) * fraction)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
