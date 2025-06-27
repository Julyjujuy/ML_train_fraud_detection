import pandas as pd
import numpy as np
import pytest

from src.features import basic_clean_cc, basic_clean_ps

def test_basic_clean_cc_scales_and_drops_amount():
    # build a tiny toy DataFrame
    df = pd.DataFrame({
        "Amount": [0.0, 10.0, 100.0],
        "V1": [1, 2, 3],
        "Class": [0, 1, 0]
    })

    cleaned = basic_clean_cc(df)

    # ScaledAmount should exist
    assert "ScaledAmount" in cleaned.columns

    # Original Amount should be dropped
    assert "Amount" not in cleaned.columns

    # ScaledAmount has zero mean and unit variance (within numeric tolerance)
    arr = cleaned["ScaledAmount"].to_numpy()
    assert np.isclose(arr.mean(), 0, atol=1e-6)
    assert np.isclose(arr.std(ddof=0), 1, atol=1e-6)

def test_basic_clean_ps_encodes_and_logs():
    df = pd.DataFrame({
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [0.0, 100.0, 1000.0],
        "oldbalanceOrg": [0.0, 50.0, 500.0],
        "newbalanceOrig": [0.0, 150.0, 1500.0],
        "oldbalanceDest": [0.0, 20.0, 200.0],
        "newbalanceDest": [0.0, 120.0, 1200.0],
        # include labels so DataFrame stays valid; they won’t be touched
        "isFraud": [0, 1, 0],
        "isFlaggedFraud": [0, 0, 1]
    })

    cleaned = basic_clean_ps(df)

    # Original monetary columns are dropped
    for col in ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]:
        assert col not in cleaned.columns

    # Log columns are present
    for col in ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]:
        assert f"log_{col}" in cleaned.columns

    # Categorical 'type' was one-hot'd: we should have at least one dummy column
    # (e.g. type_TRANSFER, type_CASH_OUT if drop_first=True)
    dummy_cols = [c for c in cleaned.columns if c.startswith("type_")]
    assert len(dummy_cols) >= 2

    # Check that log transform did what we expect:
    # for amount=0 → log1p(0)=0; for amount=100 → ~4.615; etc.
    vals = cleaned["log_amount"].to_numpy()
    assert np.isclose(vals[0], 0.0, atol=1e-6)
    assert np.isclose(vals[1], np.log1p(100.0), atol=1e-6)
