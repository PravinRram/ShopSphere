"""Fair allocation of the weekly contact list (bias mitigation by post-processing).

Ranking every customer by predicted probability and taking the top k gives most of the
list to groups that buy more often (e.g. Platinum), so buyers in low-rate groups (e.g.
Bronze) are rarely reached. This module splits the k slots across groups so that the
*expected* share of each group's buyers who get contacted (recall, "equal opportunity")
is the same in every group.

It uses only the model's calibrated probabilities, never the labels, so it works on new
customers at prediction time. Calibration was checked per group (experiment results,
finding on bias): mean prediction is within 0.02 of the true rate in every tier, country
and age band, so the summed probability is a fair estimate of each group's buyers.

`alpha` sets how far to move from the plain ranking (0) to full equal opportunity (1),
so the business can choose the trade-off between precision and fairness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _equal_opportunity_counts(p: pd.Series, groups: pd.Series, k: int) -> pd.Series:
    """Slots per group so every group reaches the same expected recall (sum of k slots = k)."""
    order = {g: np.sort(p[groups == g].to_numpy())[::-1] for g in groups.unique()}
    cum = {g: np.cumsum(v) / v.sum() for g, v in order.items()}      # expected recall after n picks

    def total(r: float) -> int:                                       # slots needed for recall r
        return sum(int(np.searchsorted(c, r, side="left")) + 1 if r > 0 else 0 for c in cum.values())

    lo, hi = 0.0, 1.0
    for _ in range(60):                                               # bisection on the common recall
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if total(mid) <= k else (lo, mid)
    counts = pd.Series({g: min(len(order[g]), int(np.searchsorted(c, lo, side="left")) + 1)
                        for g, c in cum.items()})
    short = k - counts.sum()                                          # hand out leftover slots
    for g in pd.Series({g: cum[g][min(counts[g], len(cum[g]) - 1)] for g in cum}).sort_values().index:
        if short <= 0:
            break
        if counts[g] < len(order[g]):
            counts[g] += 1
            short -= 1
    return counts


def allocate(p: pd.Series, groups: pd.Series, k: int, alpha: float = 1.0) -> pd.Series:
    """Boolean Series: True for the k customers selected.

    p      calibrated purchase probabilities, indexed like `groups`
    groups the group of each customer (e.g. loyalty tier or age band)
    k      list size from the weekly budget
    alpha  0 = plain top-k by probability, 1 = equal expected recall across groups
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    groups = groups.reindex(p.index).fillna("unknown").astype(str)    # missing group is its own group
    plain = p.rank(ascending=False, method="first") <= k
    base = groups[plain].value_counts().reindex(groups.unique(), fill_value=0)
    fair = _equal_opportunity_counts(p, groups, k).reindex(base.index)
    want = (1 - alpha) * base + alpha * fair
    counts = np.floor(want).astype(int)
    for g in (want - counts).sort_values(ascending=False).index[: k - counts.sum()]:
        counts[g] += 1                                                # largest remainders
    chosen = pd.Series(False, index=p.index)
    for g, n in counts.items():
        chosen[p[groups == g].nlargest(n).index] = True
    return chosen
