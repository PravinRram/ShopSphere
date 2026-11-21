"""Which snapshot rows train and validate each fold.

For fold f: train on every snapshot of customers outside f; validate only on the real
30 Sep 2024 snapshot of customers in f. A validation customer's older snapshots never
enter training, so the model cannot learn that customer's own history from their earlier
rows and then be scored on them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cutoff_rules import SNAPSHOT


def split(rows: pd.DataFrame, folds: pd.DataFrame, fold: int, use_snapshots: bool = True):
    """Boolean masks (train, valid) over `rows` (columns customer_id, cutoff)."""
    f = rows[["customer_id"]].merge(folds, on="customer_id", how="left").fold.values
    if np.isnan(f.astype(float)).any():
        raise ValueError("a row's customer is not in folds.csv (test customers must never appear)")
    real = (rows.cutoff == SNAPSHOT).values
    train = (f != fold) & (True if use_snapshots else real)
    valid = (f == fold) & real
    return train, valid
