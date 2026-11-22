"""Snapshot cutoffs and labels.

The real snapshot (30 Sep 2024) is labelled by train.csv's target. Each extra snapshot asks
the same question at an earlier month-end: did the customer order in the 30 days after the
cutoff? Those labels come from `transactions`, all dated on or before 30 Sep 2024.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .cutoff_rules import SNAPSHOT, Tables

LABEL_DAYS = 30
EXTRA_CUTOFFS = list(pd.date_range("2023-09-30", "2024-08-31", freq="ME"))   # 12 month-ends
ALL_CUTOFFS = EXTRA_CUTOFFS + [SNAPSHOT]


def load_train_labels(db_path: str) -> pd.DataFrame:
    """customer_id, target for the 64,000 train customers. test.csv is never read here."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    labels = pd.read_sql_query("SELECT customer_id, target FROM train", con)
    con.close()
    return labels


def label_window(cutoff) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end] of the label window: the 30 days after the cutoff."""
    c = pd.Timestamp(cutoff)
    return c, c + pd.Timedelta(days=LABEL_DAYS)


def snapshot_labels(t: Tables, train_labels: pd.DataFrame, cutoff) -> pd.DataFrame:
    """customer_id, cutoff, target for train customers signed up by the cutoff."""
    c = pd.Timestamp(cutoff)
    signed = t.customers.loc[t.customers.signup_date <= c, "customer_id"]
    rows = train_labels[train_labels.customer_id.isin(signed)][["customer_id"]].copy()
    if c == SNAPSHOT:
        rows = rows.merge(train_labels, on="customer_id")
    else:
        start, end = label_window(c)
        if end > SNAPSHOT:
            raise ValueError(f"label window for {c.date()} ends after the data snapshot")
        tx = t.transactions
        buyers = tx.loc[(tx.transaction_timestamp > start) & (tx.transaction_timestamp <= end), "customer_id"].unique()
        rows["target"] = rows.customer_id.isin(buyers).astype(int)
    rows["cutoff"] = c
    return rows[["customer_id", "cutoff", "target"]]
