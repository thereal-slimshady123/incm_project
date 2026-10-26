"""
Test suite verifying adherence to §3.1 Trial DataFrame Schema contract.
Guarantees seamless inter-operation between Track A (simulator) and Track B (IBL loader).
"""

import pytest
import numpy as np
import pandas as pd
from src.models.simulator import simulate_ddm


REQUIRED_COLUMNS = [
    "subject_id",
    "trial_idx",
    "contrast",
    "prior",
    "prev_choice",
    "choice",
    "rt",
]


def test_schema_columns_present():
    """Verify all 7 required columns are present in output."""
    df = simulate_ddm(params={"a": 1.0, "t0": 0.2, "k": 2.5}, n_trials=50, seed=1)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required contract column: {col}"


def test_schema_data_types_and_ranges():
    """Verify data types and physical value boundaries."""
    df = simulate_ddm(params={"a": 1.2, "t0": 0.25, "k": 3.0, "v_b": 0.5}, n_trials=100, seed=2)

    # Types
    assert pd.api.types.is_numeric_dtype(df["contrast"])
    assert pd.api.types.is_numeric_dtype(df["prior"])
    assert pd.api.types.is_integer_dtype(df["choice"]) or np.issubdtype(df["choice"].dtype, np.integer)
    assert pd.api.types.is_float_dtype(df["rt"])

    # Value ranges
    assert np.all(df["contrast"] >= -1.0) and np.all(df["contrast"] <= 1.0)
    assert np.all(np.isin(df["choice"].values, [-1, 1]))
    assert np.all(df["rt"] >= 0.25 - 1e-5), "RT must be >= non-decision time t0"
    assert np.all(df["rt"] <= 3.0), "RT exceeded reasonable max decision cutoff"


def test_contrast_monotonicity():
    """Verify higher contrast produces higher accuracy (fundamental psychometric law)."""
    df = simulate_ddm(params={"a": 1.0, "t0": 0.2, "k": 3.5}, n_trials=3000, seed=42)

    # Exclude 0 contrast
    df_nonzero = df[df["contrast"] != 0].copy()
    df_nonzero["correct"] = (df_nonzero["contrast"] > 0) == (df_nonzero["choice"] == 1)

    low_c = df_nonzero[np.abs(df_nonzero["contrast"]) <= 0.125]
    high_c = df_nonzero[np.abs(df_nonzero["contrast"]) >= 0.50]

    acc_low = low_c["correct"].mean()
    acc_high = high_c["correct"].mean()

    assert acc_high > acc_low, f"High contrast accuracy ({acc_high:.2f}) must exceed low contrast ({acc_low:.2f})"
