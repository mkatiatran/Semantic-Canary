import numpy as np
import pandas as pd
from semantic_canary.monitoring import psi, ks

def test_psi_identical_is_near_zero():
    x = np.arange(1000)
    assert psi(x, x) < 1e-9

def test_ks_identical_has_high_pvalue():
    x = np.arange(1000)
    out = ks(x, x)
    assert out["p_value"] > 0.9
