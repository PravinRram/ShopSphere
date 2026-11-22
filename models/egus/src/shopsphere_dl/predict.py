"""Score customers with an exported model bundle (see export.py).

    python -m shopsphere_dl.predict --bundle models/e4t_v1 --who test --out submission_E4T.csv
    python -m shopsphere_dl.predict --bundle models/e4t_v1 --who all  --out scores_all.csv   # dashboard

Every customer is scored as at the 30 Sep 2024 snapshot: the same features (`build`) and event
windows (`EventStore`) the models were trained on, under the same cutoff rules. Each of the 25
models gives sigmoid(logit / temperature); the score is their mean, the probability of at least
one purchase between 1 and 30 Oct 2024.

Output columns: customer_id, score (probability), rank (1 = most likely to buy).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd
import torch

from .cutoff_rules import SNAPSHOT, load_tables
from .dataset import SnapshotRows
from .export import build_model
from .features_extra import add_extra, load_extra
from .features_static import build
from .sequences import CATEGORICAL as SEQ_CATEGORICAL, EventStore


def customer_ids(db: str, who: str) -> np.ndarray:
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    q = {"test": "SELECT customer_id FROM test", "train": "SELECT customer_id FROM train",
         "all": "SELECT customer_id FROM customers"}[who]
    ids = pd.read_sql_query(q, con).customer_id.to_numpy()
    con.close()
    return ids


def score(bundle: str, db: str, ids, tables=None, store=None, verbose=True, max_models=None,
          batch_size=8192) -> pd.DataFrame:
    """Mean calibrated probability over the bundle's models (all 25, or the first `max_models`)."""
    with open(os.path.join(bundle, "manifest.json")) as f:
        manifest = json.load(f)
    t0 = time.time()
    tables = tables if tables is not None else load_tables(db)
    store = store if store is not None else EventStore.from_tables(tables)
    for c in SEQ_CATEGORICAL:                     # embedding sizes depend on these lists: they must match
        assert list(map(str, store.vocab[c])) == manifest["event_vocab"][c], f"event vocabulary changed: {c}"
    rows = build(tables, SNAPSHOT, ids)
    extra = manifest["settings"].get("extra", "")                 # e4t_v1 has none; e7 adds crm,geo
    if extra:
        rows = add_extra(rows, load_extra(db), extra)
    missing = set(ids) - set(rows.customer_id)
    assert not missing, f"{len(missing)} customers have no profile at the snapshot"
    rows["target"] = 0                            # placeholder: SnapshotRows expects the column

    device = torch.device("cpu")
    paths = sorted(glob.glob(os.path.join(bundle, "model_s*_f*.pt")))
    if max_models:
        paths = paths[:max_models]
    # The event windows do not depend on the model, so they are cut once and reused by every
    # model; only the static features differ (each fold has its own fitted preprocessor).
    first = torch.load(paths[0], map_location=device, weights_only=False)
    seq = SnapshotRows(rows, first["prep"], store, manifest["settings"]["n_events"]).collate(np.arange(len(rows)))
    t_data = time.time() - t0
    probs = []
    t1 = time.time()
    for path in paths:
        m = torch.load(path, map_location=device, weights_only=False)
        model = build_model(m["prep"], store.vocab, manifest["settings"])
        model.load_state_dict(m["state_dict"])
        model.eval()
        num, cat = m["prep"].transform(rows)
        logits = []
        with torch.no_grad():
            for i in range(0, len(rows), batch_size):
                sl = slice(i, i + batch_size)
                b = {"static_num": torch.from_numpy(num[sl]), "static_cat": torch.from_numpy(cat[sl]),
                     **{k: seq[k][sl] for k in ("seq_num", "seq_cat", "seq_time", "seq_len")}}
                logits.append(model(b).numpy())
        logits = np.concatenate(logits)
        probs.append(1 / (1 + np.exp(-logits / m["temperature"])))
    t_models = time.time() - t1
    out = pd.DataFrame({"customer_id": rows.customer_id.values, "score": np.mean(probs, axis=0)})
    out["rank"] = out.score.rank(ascending=False, method="first").astype(int)
    if verbose:
        print(f"scored {len(out):,} customers with {len(probs)} models: data {t_data:.0f}s, "
              f"models {t_models:.0f}s, mean score {out.score.mean():.4f}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--who", default="test", choices=["test", "train", "all"])
    p.add_argument("--out", required=True)
    p.add_argument("--max-models", type=int, default=None, help="use only the first N models (faster, less stable)")
    p.add_argument("--threshold", type=float, default=None,
                   help="also write <out>_labels.csv with target = 1 where score >= threshold")
    a = p.parse_args()
    db = os.environ.get("SHOPSPHERE_DB", "shopsphere.db")
    out = score(a.bundle, db, customer_ids(db, a.who), max_models=a.max_models)
    out.sort_values("customer_id").to_csv(a.out, index=False)
    if a.threshold is not None:
        lab = out.assign(target=(out.score >= a.threshold).astype(int))[["customer_id", "target"]]
        lab.sort_values("customer_id").to_csv(a.out.replace(".csv", "_labels.csv"), index=False)


if __name__ == "__main__":
    main()
