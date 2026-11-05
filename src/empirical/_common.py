"""
Shared helpers for the Track B empirical-analysis tools.

All tools accept any DataFrame that follows the §3.1 Trial DataFrame schema (empirical IBL
data, Track A simulator output or the synthetic mock generator) and only require the columns
they actually use.
"""

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

CORE_COLUMNS = ("subject_id", "contrast", "prior", "choice", "rt")


def require_columns(df: pd.DataFrame, columns: Iterable[str] = CORE_COLUMNS) -> None:
    """Raise a ValueError naming every missing §3.1 column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame violates the §3.1 trial schema; missing columns: {missing}")


def biased_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Trials from biased blocks only (prior = +1 or -1); neutral prior = 0 is dropped."""
    return df[df["prior"] != 0]


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for binomial proportions (robust at p near 0 or 1)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1.0 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def sem(x: np.ndarray) -> float:
    """Standard error of the mean (NaN-safe; NaN for fewer than 2 values)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
