"""Model and batching tests. Run: PYTHONPATH=src pytest -q tests/test_model.py

Needs shopsphere.db (SHOPSPHERE_DB) and folds.csv (SHOPSPHERE_FOLDS), like the leakage tests.
"""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from shopsphere_dl.cutoff_rules import SNAPSHOT, load_tables
from shopsphere_dl.dataset import SEQ_COUNT_COLS, SnapshotRows, StaticPreprocessor
from shopsphere_dl.features_static import build
from shopsphere_dl.model import TwoBranchNet
from shopsphere_dl.sequences import CATEGORICAL, NUMERIC, TIME, EventStore
from shopsphere_dl.snapshots import load_train_labels, snapshot_labels
from shopsphere_dl.train import cross_fit_calibration, fit_temperature

DB = os.environ.get("SHOPSPHERE_DB", "shopsphere.db")
N = 82


@pytest.fixture(scope="module")
def setup():
    tables = load_tables(DB)
    labels = load_train_labels(DB).sample(3000, random_state=0)
    rows = pd.concat([snapshot_labels(tables, labels, c).merge(build(tables, c, labels.customer_id).drop(columns="cutoff"),
                                                               on="customer_id")
                      for c in [pd.Timestamp("2024-03-31"), SNAPSHOT]], ignore_index=True)
    store = EventStore.from_tables(tables)
    prep = StaticPreprocessor().fit(rows)
    ds = SnapshotRows(rows, prep, store, N)
    cards = [len(store.vocab[c]) + 1 for c in CATEGORICAL]
    return rows, store, prep, ds, cards


def _model(prep, cards, **kw):
    torch.manual_seed(0)
    return TwoBranchNet(prep.n_numeric, prep.cat_cardinalities, len(NUMERIC), len(TIME), cards, **kw).eval()


def test_batch_shapes_and_alignment(setup):
    rows, store, prep, ds, cards = setup
    idx = list(range(64))
    b = ds.collate(idx)
    assert b["seq_num"].shape == (64, N, len(NUMERIC)) and b["static_num"].shape == (64, prep.n_numeric)
    # right-aligned: positions < length hold the same events, in the same order, as the store's window
    num, _, time, mask, length = store.window(ds.customer_id[idx], ds.cutoff[idx], N)
    num[..., SEQ_COUNT_COLS] = np.log1p(num[..., SEQ_COUNT_COLS])
    for i in range(64):
        k = length[i]
        assert np.allclose(b["seq_num"][i, :k].numpy(), num[i][mask[i]])
        assert np.allclose(b["seq_time"][i, :k].numpy(), time[i][mask[i]])
        assert (b["seq_cat"][i, k:] == 0).all()                       # padding after the events


def test_padding_never_changes_output(setup):
    rows, store, prep, ds, cards = setup
    b = ds.collate(list(range(128)))
    junk = {k: v.clone() for k, v in b.items()}
    for i, k in enumerate(b["seq_len"].tolist()):                       # overwrite every padded position
        junk["seq_num"][i, k:] = 99.0
        junk["seq_time"][i, k:] = -7.0
        junk["seq_cat"][i, k:] = 1
    for pooling in ["attention", "mean", "last"]:
        m = _model(prep, cards, pooling=pooling)
        with torch.no_grad():
            assert torch.allclose(m(b), m(junk), atol=1e-6), pooling


def test_no_history_customers(setup):
    rows, store, prep, ds, cards = setup
    zero = [i for i in range(len(ds)) if store.window(ds.customer_id[[i]], ds.cutoff[[i]], N)[4][0] == 0][:16]
    assert zero, "sample should contain customers with no events before the cutoff"
    b = ds.collate(zero)
    out = _model(prep, cards)(b)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("kw", [dict(), dict(use_sequence=False), dict(use_static=False),
                                dict(pooling="mean"), dict(pooling="last")])
def test_ablations_run_and_learn(setup, kw):
    rows, store, prep, ds, cards = setup
    torch.manual_seed(0)
    m = TwoBranchNet(prep.n_numeric, prep.cat_cardinalities, len(NUMERIC), len(TIME), cards, **kw)
    b = ds.collate(list(range(256)))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    first = None
    for _ in range(30):                                                  # overfit one batch: loss must fall
        opt.zero_grad(); loss = loss_fn(m(b), b["target"]); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.7 * first


def test_preprocessor_fitted_on_training_rows_only(setup):
    rows, store, prep, ds, cards = setup
    a = StaticPreprocessor().fit(rows.iloc[:1000])
    b = StaticPreprocessor().fit(rows.iloc[1000:])
    assert not np.allclose(a.transform(rows.iloc[:50])[0], b.transform(rows.iloc[:50])[0])


def test_temperature_scaling():
    rng = np.random.default_rng(0)
    z = rng.normal(-2.0, 1.5, 50000)                                    # true log-odds
    y = (rng.random(50000) < 1 / (1 + np.exp(-z))).astype(float)        # labels drawn from them
    t = fit_temperature(z * 3.0, y)                                     # a model 3x over-confident
    assert 2.7 < t < 3.3                                                # the fitted temperature undoes it
    oof = pd.DataFrame({"logit": z * 3.0, "target": y.astype(int), "fold": rng.integers(0, 5, 50000)})
    cal = cross_fit_calibration(oof)
    assert abs(cal.score.mean() - y.mean()) < 0.005
