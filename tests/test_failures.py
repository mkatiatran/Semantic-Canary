import pandas as pd
import numpy as np
from semantic_canary.failures import unit_scale_feature, swap_numeric_columns

def test_unit_scale_changes_only_target():
    df = pd.DataFrame({"claim_amount":[1.,2.], "deductible":[1.,1.]})
    out = unit_scale_feature(df, factor=12)
    assert out["claim_amount"].tolist() == [12.,24.]
    assert out["deductible"].tolist() == [1.,1.]

def test_swap():
    df = pd.DataFrame({"a":[1.,2.], "b":[3.,4.]})
    out = swap_numeric_columns(df, "a", "b")
    assert out["a"].tolist() == [3.,4.]
