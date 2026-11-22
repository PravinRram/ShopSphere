"""Re-run my own LightGBM baseline (03) under the lecturer's strict cutoff ruling of 5 Nov 2026.

The notebook `03_lightgbm_baseline_final.ipynb` was written before the ruling. This script runs the
notebook's own feature SQL (cell 8) and feature cell (cell 9), applies only the ruling's four fixes by
exact string replacement (each must match exactly once), then trains with 03's tuned settings on the
shared 5 customer folds:

  - retention outcomes (redeemed, purchased_within_14d) count only for contacts on or before 16 Sep 2024,
    so every 14-day window ends by the 30 Sep cutoff;
  - late deliveries and average delay count only orders whose promised date is on or before 30 Sep 2024.

Usage (from models/egus/baseline_03/):
    SHOPSPHERE_DB=path/to/shopsphere.db python rerun_03_strict_cutoff.py [n_seeds]
Writes oof_03_strict.csv (customer_id, fold, seed, score).
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DB = os.environ.get("SHOPSPHERE_DB", "shopsphere.db")
FOLDS = os.environ.get("SHOPSPHERE_FOLDS", str(HERE.parent / "folds.csv"))
n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5

nb = json.load(open(HERE / "03_lightgbm_baseline_final.ipynb"))
cell = lambda i: "".join(nb["cells"][i]["source"])
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def fix(sql, old, new):
    assert sql.count(old) == 1, f"expected exactly one match for: {old!r}"
    return sql.replace(old, new)


ns = {"SNAPSHOT": "2024-09-30"}
exec(cell(8).split("t0 = time.time()")[0], ns)          # defines D30, D90 and feature_sql
sql = ns["feature_sql"]
sql = fix(sql, "SUM(redeemed)                AS retention_redeemed_count",
          "SUM(CASE WHEN contact_date <= '2024-09-16' THEN redeemed END) AS retention_redeemed_count")
sql = fix(sql, "SUM(purchased_within_14d)    AS retention_purchased_count",
          "SUM(CASE WHEN contact_date <= '2024-09-16' THEN purchased_within_14d END) AS retention_purchased_count")
sql = fix(sql, "SUM(d.delay_days >= 1)              AS late_deliveries_count",
          "SUM(d.delay_days >= 1 AND d.promised_date <= '2024-09-30') AS late_deliveries_count")
sql = fix(sql, "AVG(d.delay_days)                   AS avg_delay_days",
          "AVG(CASE WHEN d.promised_date <= '2024-09-30' THEN d.delay_days END) AS avg_delay_days")

df = pd.read_sql_query(sql, con)
g = {"df": df, "np": np, "pd": pd, "print": lambda *a, **k: None}
exec(cell(9), g)                                         # derived ratios, categoricals, FEATURES, X, y
train, X, y, CAT = g["train"], g["X"], g["y"], g["CATEGORICAL"]
y = np.asarray(y, dtype=int)

best = {"learning_rate": 0.0103, "num_leaves": 21, "min_child_samples": 11, "feature_fraction": 0.4342,
        "bagging_fraction": 0.7020, "lambda_l1": 0.0308, "lambda_l2": 0.4781}   # 03's tuned values, as printed
params = {"objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt", "bagging_freq": 1,
          "verbose": -1, "n_jobs": -1, **best}

folds = pd.read_csv(FOLDS)
fold_id = train[["customer_id"]].merge(folds, on="customer_id", how="left").fold.values
assert (fold_id >= 0).all() and len(np.unique(fold_id)) == 5

rows, t0 = [], time.time()
for seed in range(1, n_seeds + 1):
    oof = np.zeros(len(y))
    for k in range(5):
        tr, va = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=CAT)
        dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=CAT, reference=dtr)
        m = lgb.train({**params, "seed": seed}, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    rows.append(pd.DataFrame({"customer_id": train.customer_id.values, "fold": fold_id, "seed": seed, "score": oof}))
pd.concat(rows).to_csv(HERE / "oof_03_strict.csv", index=False)
print(f"03 strict cutoff: {len(X.columns)} features, {n_seeds} seeds, {time.time() - t0:.0f}s")
