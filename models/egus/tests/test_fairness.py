import numpy as np
import pandas as pd
import pytest

from shopsphere_dl.fairness import allocate


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    groups = pd.Series(["high"] * 2000 + ["low"] * 8000)
    p = pd.Series(np.r_[rng.beta(5, 5, 2000), rng.beta(1, 15, 8000)])
    return p, groups


def test_selects_exactly_k(toy):
    p, g = toy
    for a in (0, 0.5, 1):
        assert allocate(p, g, 1000, a).sum() == 1000


def test_alpha_zero_is_plain_top_k(toy):
    p, g = toy
    plain = p.rank(ascending=False, method="first") <= 1000
    assert (allocate(p, g, 1000, 0) == plain).all()


def test_alpha_one_equalises_expected_recall(toy):
    p, g = toy
    s = allocate(p, g, 1000, 1)
    recall = {grp: p[s & (g == grp)].sum() / p[g == grp].sum() for grp in g.unique()}
    assert abs(recall["high"] - recall["low"]) < 0.02


def test_never_uses_labels(toy):
    p, g = toy                                   # signature has no target: works on new customers
    assert "target" not in allocate.__code__.co_varnames
