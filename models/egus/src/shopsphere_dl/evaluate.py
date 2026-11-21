"""Shared evaluation for every ShopSphere purchase model.

Every member's model is scored with these functions, on the same folds, so the
bake-off compares models and not evaluation code.

Out-of-fold (OOF) file format, one row per customer per seed:
    customer_id, fold, seed, score
`score` is the predicted probability of at least one purchase in the 30-day window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_K = (1000, 2500)


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """Share of true buyers among the k highest-scored customers.

    Ties are broken by original order (stable sort), so the result is reproducible.
    """
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].mean())


def expected_calibration_error(y: np.ndarray, score: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-count bins.

    Equal-count (quantile) bins are used instead of equal-width bins because most
    scores sit below 0.2; equal-width bins would leave the upper bins nearly empty.
    ECE = sum over bins of (share of customers in bin) * |mean score - actual buy rate|.
    """
    order = np.argsort(score, kind="stable")
    bins = np.array_split(order, n_bins)
    n = len(y)
    return float(sum(len(b) / n * abs(score[b].mean() - y[b].mean()) for b in bins))


def evaluate(y: np.ndarray, score: np.ndarray, k_list=DEFAULT_K) -> dict:
    """All headline metrics for one set of scores."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    base = y.mean()
    out = {
        "ROC-AUC": roc_auc_score(y, score),
        "PR-AUC": average_precision_score(y, score),
        "ECE": expected_calibration_error(y, score),
        "mean_score": float(score.mean()),
        "base_rate": float(base),
    }
    for k in k_list:
        out[f"P@{k}"] = precision_at_k(y, score, k)
        out[f"Lift@{k}"] = out[f"P@{k}"] / base
    return out


def evaluate_oof(oof: pd.DataFrame, labels: pd.DataFrame, k_list=DEFAULT_K) -> tuple[pd.DataFrame, pd.Series]:
    """Score a long-format OOF file seed by seed.

    Returns (per-seed table, mean across seeds). The headline number is the mean of
    the per-seed metrics, so seed-to-seed spread can be reported alongside it.
    `labels` has columns customer_id, target.
    """
    rows = []
    for seed, part in oof.groupby("seed"):
        m = part.merge(labels, on="customer_id", how="inner", validate="one_to_one")
        if len(m) != len(labels):
            raise ValueError(f"seed {seed}: {len(m)} scored customers, expected {len(labels)}")
        rows.append({"seed": seed, **evaluate(m.target.values, m.score.values, k_list)})
    per_seed = pd.DataFrame(rows).set_index("seed")
    return per_seed, per_seed.mean()


def seed_mean_scores(oof: pd.DataFrame) -> pd.Series:
    """One score per customer: the mean across seeds (used for comparisons and the bootstrap)."""
    return oof.groupby("customer_id").score.mean()


def paired_bootstrap(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, k: int,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap of P@k(A) - P@k(B) on the same customers.

    Customers are resampled with replacement; both models are scored on the same
    resample, so the comparison removes the noise of which customers were drawn.
    A wins only if the whole 95% interval is above zero.
    """
    y = np.asarray(y, dtype=int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y, score_a and score_b must be aligned on the same customers")
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = precision_at_k(y[idx], a[idx], k) - precision_at_k(y[idx], b[idx], k)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    observed = precision_at_k(y, a, k) - precision_at_k(y, b, k)
    verdict = "A wins" if lo > 0 else ("B wins" if hi < 0 else "tie")
    return {"k": k, "diff": observed, "ci_low": lo, "ci_high": hi, "verdict": verdict}


def paired_bootstrap_metric(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, metric: str,
                            n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap of metric(A) - metric(B) for metric in {'ROC-AUC', 'PR-AUC'}.

    Same resampling as `paired_bootstrap`: customers drawn with replacement, both models
    scored on the same draw. A wins only if the whole 95% interval is above zero.
    """
    fn = {"ROC-AUC": roc_auc_score, "PR-AUC": average_precision_score}[metric]
    y = np.asarray(y, dtype=int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y, score_a and score_b must be aligned on the same customers")
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = fn(y[idx], a[idx]) - fn(y[idx], b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    observed = fn(y, a) - fn(y, b)
    verdict = "A wins" if lo > 0 else ("B wins" if hi < 0 else "tie")
    return {"metric": metric, "diff": observed, "ci_low": lo, "ci_high": hi, "verdict": verdict}
