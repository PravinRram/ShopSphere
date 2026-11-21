"""Leakage tests for the snapshot pipeline. Run: PYTHONPATH=src pytest -q tests/

Needs shopsphere.db (path in SHOPSPHERE_DB, default ./shopsphere.db) and folds.csv
(path in SHOPSPHERE_FOLDS, default ./folds.csv).
"""
import os

import numpy as np
import pandas as pd
import pytest

from shopsphere_dl.cutoff_rules import SNAPSHOT, Tables, as_of, date_columns, load_tables
from shopsphere_dl.features_static import FEATURES, build
from shopsphere_dl.folds import split
from shopsphere_dl.sequences import EPOCH, EventStore
from shopsphere_dl.snapshots import ALL_CUTOFFS, label_window, load_train_labels, snapshot_labels

DB = os.environ.get("SHOPSPHERE_DB", "shopsphere.db")
FOLDS = os.environ.get("SHOPSPHERE_FOLDS", "folds.csv")
TEST_CUTOFFS = [pd.Timestamp("2023-09-30"), pd.Timestamp("2024-03-31"), pd.Timestamp("2024-08-31"), SNAPSHOT]


@pytest.fixture(scope="module")
def tables():
    return load_tables(DB)


@pytest.fixture(scope="module")
def labels():
    return load_train_labels(DB)


def _assert_frames_equal(a, b):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=False, rtol=1e-9)


def _scramble_future(t: Tables, c: pd.Timestamp, seed: int = 0) -> Tables:
    """Replace everything NOT known at c with random values. Features must not change."""
    rng = np.random.default_rng(seed)
    far = pd.Timestamp("2030-01-01")

    s = t.sessions.copy()
    fut = s.session_timestamp > c
    s.loc[fut, "cart_additions"] = rng.integers(0, 50, fut.sum())
    s.loc[fut, "session_duration_capped"] = rng.uniform(0, 9999, fut.sum())

    x = t.transactions.copy()
    fut_tx = x.transaction_timestamp > c
    x.loc[fut_tx, "order_value"] = rng.uniform(0, 9999, fut_tx.sum())
    x.loc[fut_tx, "transaction_timestamp"] = far

    d = t.deliveries.copy()
    for col in ["ship_date", "promised_date", "delivered_date"]:
        d.loc[d[col] > c, col] = far
    unknown = ~(d.promised_date.le(c) & d.delivered_date.le(c))
    d.loc[unknown, "delay_days"] = rng.uniform(-50, 50, unknown.sum())

    k = t.tickets.copy()
    resolved_after = (k.created_date + pd.to_timedelta(k.resolution_hours, unit="h")) >= c + pd.Timedelta(days=1)
    k.loc[resolved_after & k.resolution_hours.notna(), "csat_score"] = 1.0
    k.loc[k.created_date > c, "priority"] = "High"

    r = t.contacts.copy()
    late = r.contact_date > c - pd.Timedelta(days=14)
    r.loc[late, "purchased_within_14d"] = 1
    r.loc[late, "redeemed"] = 1
    return Tables(t.customers, s, x, d, k, r)


# 1. Truncate-and-compare: features from the full tables equal features from tables cut at c.
@pytest.mark.parametrize("c", TEST_CUTOFFS)
def test_truncate_and_compare(tables, labels, c):
    full = build(tables, c, labels.customer_id)
    cut = Tables(
        tables.customers[tables.customers.signup_date <= c],
        tables.sessions[tables.sessions.session_timestamp <= c],
        tables.transactions[tables.transactions.transaction_timestamp <= c],
        tables.deliveries,
        tables.tickets[tables.tickets.created_date <= c],
        tables.contacts[tables.contacts.contact_date <= c],
    )
    _assert_frames_equal(full, build(cut, c, labels.customer_id))


# 1b. Stronger version: scrambling every value not known at c changes no feature.
@pytest.mark.parametrize("c", TEST_CUTOFFS)
def test_future_scramble(tables, labels, c):
    _assert_frames_equal(build(tables, c, labels.customer_id),
                         build(_scramble_future(tables, c), c, labels.customer_id))


# 2. No date after the cutoff anywhere in the as-of tables.
@pytest.mark.parametrize("c", TEST_CUTOFFS)
def test_no_date_after_cutoff(tables, c):
    for name, col in date_columns(as_of(tables, c)).items():
        latest = col.max()
        if name == "tickets.resolved_at":
            assert pd.isna(latest) or latest < c + pd.Timedelta(days=1), name
        else:
            assert pd.isna(latest) or latest <= c, f"{name}: {latest} after {c.date()}"


# 3. Contact outcomes only from contacts at least 14 days before the cutoff.
@pytest.mark.parametrize("c", TEST_CUTOFFS)
def test_contact_outcomes_14_days(tables, c):
    r = as_of(tables, c).contacts
    known = r.purchased_within_14d.notna() | r.redeemed.notna()
    assert (r.loc[known, "contact_date"] <= c - pd.Timedelta(days=14)).all()


# 4. Every event in every sequence is dated on or before its row's cutoff.
def test_sequences_respect_cutoff(tables):
    store = EventStore.from_tables(tables)
    rng = np.random.default_rng(1)
    ids = rng.choice(store.customer_ids, 3000)
    cuts = rng.choice(np.array(ALL_CUTOFFS, dtype="datetime64[ns]"), 3000)
    _, _, time, mask, length = store.window(ids, cuts, 82)
    assert (time[..., 0][mask] >= 0).all()                     # log1p(days before cutoff) >= 0  <=>  event <= cutoff
    # exact check against a direct filter for a sample
    ev = pd.concat([tables.sessions[["customer_id", "session_timestamp"]].rename(columns={"session_timestamp": "d"}),
                    tables.transactions[["customer_id", "transaction_timestamp"]].rename(columns={"transaction_timestamp": "d"})])
    counts = ev.groupby("customer_id").d.apply(np.array)
    for i in range(200):
        n_before = int((counts.get(ids[i], np.array([], dtype="datetime64[ns]")) <= cuts[i]).sum())
        assert length[i] == min(n_before, 82)


# 5. Every label window starts after the cutoff and ends within 30 days, by 30 Sep 2024.
@pytest.mark.parametrize("c", ALL_CUTOFFS[:-1])
def test_label_windows(tables, labels, c):
    start, end = label_window(c)
    assert start == c and end - c == pd.Timedelta(days=30) and end <= SNAPSHOT
    rows = snapshot_labels(tables, labels, c)
    buyers = set(tables.transactions[(tables.transactions.transaction_timestamp > start) &
                                     (tables.transactions.transaction_timestamp <= end)].customer_id)
    assert (rows.target == rows.customer_id.isin(buyers)).all()


# 6. No fold-f customer has any snapshot in fold f's training rows; validation is real-snapshot only.
def test_fold_rule(tables, labels):
    folds = pd.read_csv(FOLDS)
    rows = pd.concat([snapshot_labels(tables, labels, c) for c in [ALL_CUTOFFS[0], ALL_CUTOFFS[6], SNAPSHOT]])
    for f in range(5):
        tr, va = split(rows, folds, f)
        val_customers = set(folds.customer_id[folds.fold == f])
        assert not set(rows.customer_id[tr]) & val_customers
        assert (rows.cutoff[va] == SNAPSHOT).all() and va.sum() == len(val_customers)


# 7. No test.csv customer appears in any training row.
def test_no_test_customers(tables, labels):
    import sqlite3
    con = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    test_ids = set(pd.read_sql_query("SELECT customer_id FROM test", con).customer_id)
    for c in [ALL_CUTOFFS[0], SNAPSHOT]:
        assert not set(snapshot_labels(tables, labels, c).customer_id) & test_ids
        assert not set(build(tables, c, labels.customer_id).customer_id) & test_ids
