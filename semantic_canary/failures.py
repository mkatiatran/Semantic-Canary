import numpy as np
import pandas as pd

def preserve_marginal_break_relationships(df, feature="claim_amount",
                                           conditioning="previous_claims_count",
                                           seed=42):
    """Shuffle a feature within quantile bins of itself.

    This keeps the feature's marginal distribution almost exactly unchanged,
    while changing its relationship to the other columns.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    bins = pd.qcut(out[feature], q=20, duplicates="drop")
    shuffled = out[feature].copy()

    for _, idx in out.groupby(bins, observed=True).groups.items():
        vals = shuffled.loc[idx].to_numpy(copy=True)
        rng.shuffle(vals)
        out.loc[idx, feature] = vals

    return out

def unit_scale_feature(df, feature="claim_amount", factor=12.0):
    out = df.copy()
    out[feature] = out[feature] * factor
    return out

def swap_numeric_columns(df, a="claim_amount", b="deductible"):
    out = df.copy()
    out[[a, b]] = out[[b, a]].to_numpy()
    return out

def add_date_corruption(df, days=365):
    out = df.copy()
    out["claim_filed_date"] = out["claim_filed_date"] + pd.to_timedelta(days, unit="D")
    return out

def scenarios(df):
    return {
        "marginal_preserving_relation_break": preserve_marginal_break_relationships(df),
        "unit_change_claim_amount": unit_scale_feature(df),
        "column_swap": swap_numeric_columns(df),
        "date_corruption": add_date_corruption(df),
    }
