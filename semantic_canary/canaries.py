import numpy as np
import pandas as pd

class SemanticCanary:
    def __init__(self, model, numeric_features, quantiles=(0.2, 0.4, 0.6, 0.8),
                 direction_confidence=0.65):
        self.model = model
        self.features = numeric_features
        self.quantiles = quantiles
        self.direction_confidence = direction_confidence
        self.rules = {}

    def fit(self, reference):
        X = reference[self.features].copy()
        base = self.model.predict_proba(reference[self.features + ["claim_type", "state"]])[:, 1]

        for feature in self.features:
            x = X[feature].astype(float)
            deltas = max(x.std() * 0.10, 1e-6)
            signs = []
            for q in self.quantiles:
                row = reference.copy()
                row[feature] = row[feature] + deltas
                pred = self.model.predict_proba(
                    row[self.features + ["claim_type", "state"]]
                )[:, 1]
                signs.extend(np.sign(pred - base).tolist())

            pos = np.mean(np.array(signs) > 0)
            neg = np.mean(np.array(signs) < 0)
            if max(pos, neg) >= self.direction_confidence:
                self.rules[feature] = {
                    "direction": 1 if pos > neg else -1,
                    "confidence": float(max(pos, neg)),
                    "delta": float(deltas),
                }
        return self

    def evaluate(self, current):
        rows = []
        cols = self.features + ["claim_type", "state"]
        for feature, rule in self.rules.items():
            x = current.copy()
            base = self.model.predict_proba(x[cols])[:, 1]
            x[feature] = x[feature].astype(float) + rule["delta"]
            changed = self.model.predict_proba(x[cols])[:, 1] - base
            observed = np.mean(np.sign(changed) == rule["direction"])
            rows.append({
                "feature": feature,
                "expected_direction": rule["direction"],
                "training_confidence": rule["confidence"],
                "counterfactual_consistency": float(observed),
                "failed": bool(observed < 0.90),
            })
        return pd.DataFrame(rows)
