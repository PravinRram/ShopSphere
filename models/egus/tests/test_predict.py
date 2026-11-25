"""Exported model bundle: does it score like the trained model, and only from 30 Sep data?

Skipped when the bundle or the out-of-fold files are not present (e.g. on Kaggle before export).
Paths: SHOPSPHERE_DB, SHOPSPHERE_BUNDLE (default models/e4t_v1), SHOPSPHERE_OOF_DIR (default oof).
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from shopsphere_dl.cutoff_rules import load_tables
from shopsphere_dl.predict import customer_ids, score
from shopsphere_dl.sequences import EventStore

DB = os.environ.get("SHOPSPHERE_DB", "shopsphere.db")
BUNDLE = os.environ.get("SHOPSPHERE_BUNDLE", "models/e4t_v1")
OOF_DIR = os.environ.get("SHOPSPHERE_OOF_DIR", "oof")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(BUNDLE, "manifest.json")),
                                reason="no exported bundle")


@pytest.fixture(scope="module")
def data():
    t = load_tables(DB)
    return t, EventStore.from_tables(t)


def test_manifest_checks_passed():
    m = json.load(open(os.path.join(BUNDLE, "manifest.json")))
    assert m["models"] == 25
    for c in m["checks"]:                       # written by export.py after re-scoring each fold
        assert c["rank_corr"] > 0.999 and abs(c["pr_auc_export"] - c["pr_auc_oof"]) < 0.001


def test_one_model_reproduces_its_out_of_fold_scores(data):
    path = os.path.join(OOF_DIR, "oof_E4T_seed1.csv")
    if not os.path.exists(path):
        pytest.skip("no out-of-fold file")
    oof = pd.read_csv(path)
    ids = oof[oof.fold == 0].customer_id.values[:500]
    s = score(BUNDLE, DB, ids, *data, verbose=False, max_models=1)          # model_s1_f0 is first
    ref = oof.set_index("customer_id").loc[s.customer_id, "score"].values
    assert np.abs(s.score.values - ref).max() < 0.01


def test_scores_are_probabilities_for_test_customers(data):
    ids = customer_ids(DB, "test")[:300]
    s = score(BUNDLE, DB, ids, *data, verbose=False, max_models=2)
    assert len(s) == 300 and s.customer_id.is_unique
    assert s.score.between(0, 1).all() and 0.05 < s.score.mean() < 0.3
    assert sorted(s["rank"]) == list(range(1, 301))
