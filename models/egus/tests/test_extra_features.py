"""Tests for the E7 extra features (features_extra.py). Run: PYTHONPATH=src pytest -q tests/

The key rule: the crm columns exist only as their 30 Sep 2024 value, so they must be blank on
every earlier snapshot. Otherwise an earlier row would learn about months after its own cutoff.
"""
import os

import numpy as np
import pandas as pd
import pytest

from shopsphere_dl.cutoff_rules import SNAPSHOT, load_tables
from shopsphere_dl.dataset import NUMERIC_STATIC, StaticPreprocessor
from shopsphere_dl.features_extra import (CRM_CATEGORICAL, CRM_NUMERIC, GEO_CATEGORICAL, GEO_NUMERIC, NOT_KNOWN,
                                          add_extra, feature_lists, load_extra)
from shopsphere_dl.features_static import CATEGORICAL, build
from shopsphere_dl.snapshots import load_train_labels

DB = os.environ.get("SHOPSPHERE_DB", "shopsphere.db")
EARLIER = [pd.Timestamp("2023-09-30"), pd.Timestamp("2024-03-31"), pd.Timestamp("2024-08-31")]


@pytest.fixture(scope="module")
def rows():
    t = load_tables(DB)
    ids = load_train_labels(DB).customer_id.sample(3000, random_state=0)
    parts = [build(t, c, ids) for c in EARLIER + [SNAPSHOT]]
    return add_extra(pd.concat(parts, ignore_index=True), load_extra(DB), "crm,geo")


@pytest.fixture(scope="module")
def extra():
    return load_extra(DB).set_index("customer_id")


def test_crm_blank_on_every_earlier_snapshot(rows):
    early = rows[rows.cutoff < SNAPSHOT]
    assert len(early) > 0
    assert early[CRM_NUMERIC].isna().all().all(), "a profile column has a value before 30 Sep 2024"
    assert (early[CRM_CATEGORICAL] == NOT_KNOWN).all().all()


def test_crm_filled_on_the_real_snapshot(rows, extra):
    real = rows[rows.cutoff == SNAPSHOT].set_index("customer_id")
    ref = extra.loc[real.index]
    pd.testing.assert_frame_equal(real[CRM_NUMERIC].astype(float), ref[CRM_NUMERIC].astype(float), check_names=False)
    assert (real[CRM_CATEGORICAL].values == ref[CRM_CATEGORICAL].astype(str).values).all()
    assert real["loyalty_score"].notna().all()


def test_geo_same_on_every_snapshot(rows):
    per_customer = rows.groupby("customer_id")[GEO_NUMERIC + GEO_CATEGORICAL].nunique()
    assert (per_customer <= 1).all().all(), "regional data changes between snapshots"
    assert rows[GEO_NUMERIC].notna().all().all()


def test_feature_lists():
    num, cat = feature_lists("crm,geo")
    assert num == NUMERIC_STATIC + CRM_NUMERIC + GEO_NUMERIC
    assert cat == CATEGORICAL + CRM_CATEGORICAL + GEO_CATEGORICAL
    assert feature_lists("") == (NUMERIC_STATIC, CATEGORICAL)
    with pytest.raises(AssertionError):
        feature_lists("crm,typo")


def test_preprocessor_uses_the_extra_columns(rows):
    prep = StaticPreprocessor(*feature_lists("crm,geo")).fit(rows)
    num, cat = prep.transform(rows)
    assert num.shape[1] == prep.n_numeric and cat.shape[1] == len(CATEGORICAL) + 2
    assert "loyalty_score" in prep.missing_cols          # its missing flag marks the earlier snapshots
    assert np.isfinite(num).all()


def test_old_bundle_preprocessor_still_works(rows):
    """e4t_v1 pickled preprocessors without the column lists; they must keep the 39 base features."""
    prep = StaticPreprocessor().fit(rows)
    del prep.__dict__["numeric"], prep.__dict__["categorical"]   # what an e4t_v1 object looks like
    num, cat = prep.transform(rows)
    assert cat.shape[1] == len(CATEGORICAL)
    assert num.shape[1] == len(NUMERIC_STATIC) + len(prep.missing_cols) + 2
