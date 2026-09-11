import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from .data import load_claims
from .model import train_model, save_model
from .monitoring import psi, ks, relation_z_scores
from .canaries import SemanticCanary
from .failures import scenarios
from .report import write_report

NUMERIC = [
    "policyholder_tenure_years",
    "previous_claims_count",
    "claim_amount",
    "deductible",
    "days_between_incident_and_filing",
]

def train(rows):
    df = load_claims(rows=rows)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=df["is_fraud_flagged_ground_truth"]
    )
    model = train_model(train_df)
    p = model.predict_proba(test_df[NUMERIC + ["claim_type", "state"]])[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(test_df["is_fraud_flagged_ground_truth"], p)),
        "average_precision": float(average_precision_score(test_df["is_fraud_flagged_ground_truth"], p)),
        "rows": int(len(df)),
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/metrics.json").write_text(json.dumps(metrics, indent=2))
    save_model(model)
    train_df.to_parquet("artifacts/reference.parquet", index=False)
    test_df.to_parquet("artifacts/test.parquet", index=False)
    print(json.dumps(metrics, indent=2))

def benchmark(rows):
    df = load_claims(rows=rows)
    mid = len(df)//2
    reference, current = df.iloc[:mid].copy(), df.iloc[mid:].copy()
    model = train_model(reference)

    canary = SemanticCanary(model, NUMERIC).fit(reference)
    base_cf = canary.evaluate(current)
    results = {}

    for name, corrupted in scenarios(current).items():
        dist_alerts = 0
        for f in NUMERIC:
            p = psi(reference[f], corrupted[f])
            if p >= 0.20:
                dist_alerts += 1

        rel = relation_z_scores(reference, corrupted, NUMERIC)
        relation_alerts = int((rel["z"].abs() >= 4.0).sum())
        cf = canary.evaluate(corrupted)
        cf_alerts = int(cf["failed"].sum()) if len(cf) else 0

        results[name] = {
            "distribution_alerts": dist_alerts,
            "relation_alerts": relation_alerts,
            "counterfactual_alerts": cf_alerts,
            "top_relations": rel.head(5).to_dict("records"),
        }

    Path("artifacts/benchmark.json").write_text(json.dumps(results, indent=2, default=str))
    write_report(results)
    print(json.dumps(results, indent=2, default=str))

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--rows", type=int, default=100000)
    b = sub.add_parser("benchmark")
    b.add_argument("--rows", type=int, default=100000)
    sub.add_parser("report")
    args = parser.parse_args()

    if args.command == "train":
        train(args.rows)
    elif args.command == "benchmark":
        benchmark(args.rows)
    elif args.command == "report":
        print("Run benchmark first; artifacts/benchmark.json will be created.")

if __name__ == "__main__":
    main()
