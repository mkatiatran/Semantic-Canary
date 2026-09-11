import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def psi(reference, current, bins=10):
    ref = pd.Series(reference).dropna().astype(float)
    cur = pd.Series(current).dropna().astype(float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    r = np.histogram(ref, bins=edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, bins=edges)[0] / max(len(cur), 1)
    r = np.clip(r, 1e-6, None)
    c = np.clip(c, 1e-6, None)
    return float(np.sum((c-r) * np.log(c/r)))

def ks(reference, current):
    r = pd.Series(reference).dropna().astype(float)
    c = pd.Series(current).dropna().astype(float)
    stat, p = ks_2samp(r, c)
    return {"statistic": float(stat), "p_value": float(p)}

def relation_profile(df, features):
    numeric = df[features].corr(method="spearman")
    return numeric

def relation_z_scores(reference, current, features):
    r = relation_profile(reference, features)
    c = relation_profile(current, features)
    rows = []
    for a in features:
        for b in features:
            if a >= b:
                continue
            rv, cv = r.loc[a, b], c.loc[a, b]
            # Fisher z is unstable at exactly +/-1, so clip.
            rv = np.clip(rv, -0.999, 0.999)
            cv = np.clip(cv, -0.999, 0.999)
            rz = np.arctanh(rv)
            cz = np.arctanh(cv)
            se = np.sqrt(1/max(len(reference)-3,1) + 1/max(len(current)-3,1))
            z = (cz-rz)/se
            rows.append({
                "feature_a": a,
                "feature_b": b,
                "reference_corr": float(r),
                "current_corr": float(cv),
                "z": float(z),
            })
    return pd.DataFrame(rows).sort_values("z", key=lambda x: np.abs(x), ascending=False)

def schema_invariants(df):
    checks = {}
    if {"incident_date", "claim_filed_date"} <= set(df.columns):
        checks["filing_after_incident"] = float(
            (df["claim_filed_date"] >= df["incident_date"]).mean()
        )
    if {"claim_amount", "deductible"} <= set(df.columns):
        checks["nonnegative_claim_amount"] = float((df["claim_amount"] >= 0).mean())
        checks["nonnegative_deductible"] = float((df["deductible"] >= 0).mean())
    return checks
