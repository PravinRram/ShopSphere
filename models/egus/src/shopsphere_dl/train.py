"""Train the two-branch model fold by fold and write out-of-fold scores.

    python -m shopsphere_dl.train --run E4 --seed 1          # GRU + MLP, all snapshots
    python -m shopsphere_dl.train --run E5 --seed 1          # GRU + MLP, real snapshot only
    python -m shopsphere_dl.train --run E4 --seed 1 --params best_params.json --name E4T   # tuned settings
    python -m shopsphere_dl.train --run E4 --seed 1 --params best_params.json --name E7 --extra crm,geo
        # E4T settings + profile/consent columns (30 Sep snapshot only) + regional data (features_extra.py)

Paths come from SHOPSPHERE_DB, SHOPSPHERE_FOLDS and SHOPSPHERE_OUT (Kaggle: /kaggle/working).

Per fold: preprocessing is fitted on the fold's training rows only; the model trains with
AdamW, a cosine learning-rate schedule and BCEWithLogitsLoss without pos_weight; after each
epoch it is scored on the fold's real-snapshot customers, and training stops when PR-AUC has
not improved for `patience` epochs. A checkpoint is written every epoch, so a Kaggle session
that hits its time limit resumes where it stopped.

Calibration: temperature scaling is cross-fitted. Fold f's temperature is fitted on the other
four folds' out-of-fold logits, so no customer's own label calibrates their own score.
Output: oof_<run>_seed<seed>.csv with customer_id, fold, seed, score (calibrated), logit.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from .cutoff_rules import SNAPSHOT, load_tables
from .dataset import SnapshotRows, StaticPreprocessor, loader
from .evaluate import evaluate
from .features_extra import add_extra, feature_lists, load_extra, parse_groups
from .features_static import build
from .folds import split
from .model import TwoBranchNet
from .sequences import CATEGORICAL as SEQ_CATEGORICAL, NUMERIC as SEQ_NUMERIC, TIME as SEQ_TIME, EventStore
from .snapshots import ALL_CUTOFFS, load_train_labels, snapshot_labels

PRESETS = {   # experiment ladder in claude/dl_training_plan.md
    "E2": dict(use_static=True, use_sequence=False, use_snapshots=False),
    "E3": dict(use_static=True, use_sequence=False, use_snapshots=True),
    "E4": dict(use_static=True, use_sequence=True, use_snapshots=True),
    "E5": dict(use_static=True, use_sequence=True, use_snapshots=False),
    "E6": dict(use_static=False, use_sequence=True, use_snapshots=True),
}


TUNABLE = ("hidden", "gru_layers", "dropout", "lr", "weight_decay", "n_events", "max_epochs")  # see tune.py


@dataclass
class TrainConfig:
    run: str = "E4"
    seed: int = 1
    use_static: bool = True
    use_sequence: bool = True
    use_snapshots: bool = True
    pooling: str = "attention"
    hidden: int = 64
    gru_layers: int = 2
    dropout: float = 0.2
    n_events: int = 82
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    eval_batch_size: int = 4096
    max_epochs: int = 20
    patience: int = 3
    num_workers: int = 2
    folds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    extra: str = ""                         # extra static feature groups, e.g. "crm,geo" (features_extra.py)
    max_train_rows: int | None = None       # smoke tests only


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Data:
    """Everything that does not depend on the fold: snapshot rows and the event store."""

    def __init__(self, db: str, folds_path: str, customer_ids=None, extra=()):
        t0 = time.time()
        self.extra = parse_groups(extra)
        tables = load_tables(db)
        labels = load_train_labels(db)
        if customer_ids is not None:
            labels = labels[labels.customer_id.isin(customer_ids)]
        parts = []
        for c in ALL_CUTOFFS:
            lab = snapshot_labels(tables, labels, c)
            parts.append(lab.merge(build(tables, c, lab.customer_id).drop(columns="cutoff"), on="customer_id"))
        self.rows = add_extra(pd.concat(parts, ignore_index=True), load_extra(db), self.extra)
        self.store = EventStore.from_tables(tables)
        self.folds = pd.read_csv(folds_path)
        print(f"data: {len(self.rows):,} snapshot rows, {len(self.store.day):,} events, "
              f"extra features {self.extra or 'none'}, {time.time() - t0:.0f}s", flush=True)


def _predict(model, dl, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, ys = [], []
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits.append(model(b).float().cpu())
            ys.append(b["target"].cpu())
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def train_fold(data: Data, cfg: TrainConfig, fold: int, out_dir: str, device: torch.device,
               epoch_callback=None) -> pd.DataFrame:
    """Train one fold; `epoch_callback(epoch, valid_pr_auc)` is called after every epoch (tuning uses
    it to stop unpromising trials early). Returns the fold's validation logits from the best epoch."""
    set_seed(cfg.seed * 100 + fold)
    tr, va = split(data.rows, data.folds, fold, use_snapshots=cfg.use_snapshots)
    train_rows, valid_rows = data.rows[tr], data.rows[va]
    if cfg.max_train_rows and len(train_rows) > cfg.max_train_rows:
        train_rows = train_rows.sample(cfg.max_train_rows, random_state=cfg.seed)
    assert parse_groups(cfg.extra) == data.extra, "Data was built with different extra features"
    prep = StaticPreprocessor(*feature_lists(cfg.extra)).fit(train_rows, seed=cfg.seed)   # training rows only
    ds_tr = SnapshotRows(train_rows, prep, data.store, cfg.n_events)
    ds_va = SnapshotRows(valid_rows, prep, data.store, cfg.n_events)
    dl_tr = loader(ds_tr, cfg.batch_size, shuffle=True, seed=cfg.seed * 100 + fold, num_workers=cfg.num_workers)
    dl_va = loader(ds_va, cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers)

    seq_cards = [len(data.store.vocab[c]) + 1 for c in SEQ_CATEGORICAL]
    model = TwoBranchNet(prep.n_numeric, prep.cat_cardinalities, len(SEQ_NUMERIC), len(SEQ_TIME), seq_cards,
                         hidden=cfg.hidden, gru_layers=cfg.gru_layers, dropout=cfg.dropout, pooling=cfg.pooling,
                         use_static=cfg.use_static, use_sequence=cfg.use_sequence).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs * len(dl_tr))
    scaler = (torch.amp.GradScaler if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler)(
        enabled=device.type == "cuda")                                         # older PyTorch: torch.cuda.amp
    loss_fn = torch.nn.BCEWithLogitsLoss()                                     # no pos_weight: keeps probabilities honest

    ckpt = os.path.join(out_dir, f"ckpt_{cfg.run}_seed{cfg.seed}_fold{fold}.pt")
    state = {"epoch": 0, "best": -1.0, "bad": 0, "best_state": None, "history": []}
    if os.path.exists(ckpt):                                                   # resume after a session limit
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"]); scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        print(f"fold {fold}: resumed at epoch {state['epoch']}", flush=True)

    while state["epoch"] < cfg.max_epochs and state["bad"] < cfg.patience:
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for b in dl_tr:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(b).float(), b["target"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item() * len(b["target"]); n += len(b["target"])
        logits, y = _predict(model, dl_va, device)
        pr = average_precision_score(y, logits)
        state["epoch"] += 1
        state["history"].append({"epoch": state["epoch"], "train_loss": total / n, "valid_pr_auc": pr,
                                 "seconds": time.time() - t0})
        if pr > state["best"]:
            state["best"], state["bad"] = pr, 0
            state["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state["bad"] += 1
        print(f"fold {fold} epoch {state['epoch']}: loss {total / n:.4f}, valid PR-AUC {pr:.4f}, "
              f"{time.time() - t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "state": state}, ckpt)
        if epoch_callback is not None:
            epoch_callback(state["epoch"], pr)

    model.load_state_dict(state["best_state"])
    logits, _ = _predict(model, dl_va, device)
    with open(os.path.join(out_dir, f"history_{cfg.run}_seed{cfg.seed}_fold{fold}.json"), "w") as f:
        json.dump(state["history"], f, indent=1)
    return pd.DataFrame({"customer_id": valid_rows.customer_id.values, "fold": fold, "seed": cfg.seed,
                         "logit": logits, "target": valid_rows.target.values})


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """One temperature T minimising log loss of sigmoid(logit / T)."""
    lt = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lt / log_t.exp(), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp())


def cross_fit_calibration(oof: pd.DataFrame) -> pd.DataFrame:
    """Calibrated score for each fold, using a temperature fitted on the other folds only."""
    oof = oof.copy()
    oof["score"] = np.nan
    oof["temperature"] = np.nan
    for f in oof.fold.unique():
        other = oof.fold != f
        if not other.any():                     # a single-fold test run: nothing to fit on, leave uncalibrated
            print("calibration skipped: only one fold in this run", flush=True)
            oof["score"] = 1 / (1 + np.exp(-oof.logit.values)); oof["temperature"] = 1.0
            return oof
        t = fit_temperature(oof.loc[other, "logit"].values, oof.loc[other, "target"].values)
        oof.loc[~other, "score"] = 1 / (1 + np.exp(-oof.loc[~other, "logit"].values / t))
        oof.loc[~other, "temperature"] = t
    return oof


def run(cfg: TrainConfig, data: Data, out_dir: str) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{cfg.run} seed {cfg.seed} on {device}: {json.dumps(asdict(cfg))}", flush=True)
    parts = [train_fold(data, cfg, f, out_dir, device) for f in cfg.folds]
    oof = cross_fit_calibration(pd.concat(parts, ignore_index=True))
    path = os.path.join(out_dir, f"oof_{cfg.run}_seed{cfg.seed}.csv")
    oof[["customer_id", "fold", "seed", "score", "logit"]].to_csv(path, index=False)
    m = evaluate(oof.target.values, oof.score.values)
    print(f"{cfg.run} seed {cfg.seed}: " + ", ".join(f"{k} {m[k]:.4f}" for k in
          ["ROC-AUC", "PR-AUC", "P@1000", "P@2500", "ECE", "mean_score"]), flush=True)
    return oof


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="E4", choices=sorted(PRESETS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pooling", default="attention", choices=["attention", "mean", "last"])
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--sample", type=int, default=None, help="train on N random customers (quick checks only)")
    p.add_argument("--params", default=None, help="JSON of tuned settings (best_params.json from tune.py)")
    p.add_argument("--name", default=None, help="run name for output files, e.g. E4T for the tuned E4")
    p.add_argument("--extra", default="", help="extra static feature groups: crm, geo (features_extra.py)")
    a = p.parse_args()
    cfg = TrainConfig(run=a.run, seed=a.seed, pooling=a.pooling, max_epochs=a.max_epochs,
                      folds=[int(f) for f in a.folds.split(",")], num_workers=a.num_workers,
                      extra=",".join(parse_groups(a.extra)), **PRESETS[a.run])
    if a.params:                                         # tuned settings override the starting ones
        with open(a.params) as f:
            tuned = json.load(f)["params"]
        unknown = set(tuned) - set(TUNABLE)
        assert not unknown, f"not tunable: {unknown}"
        for k, v in tuned.items():
            setattr(cfg, k, v)
    if a.name:
        cfg.run = a.name
    if a.pooling != "attention":
        cfg.run = f"{cfg.run}-{a.pooling}"
    out = os.environ.get("SHOPSPHERE_OUT", ".")
    db = os.environ.get("SHOPSPHERE_DB", "shopsphere.db")
    ids = None
    if a.sample:
        ids = load_train_labels(db).sample(a.sample, random_state=0).customer_id
        cfg.run = f"{cfg.run}-sample{a.sample}"
    data = Data(db, os.environ.get("SHOPSPHERE_FOLDS", "folds.csv"), customer_ids=ids, extra=cfg.extra)
    run(cfg, data, out)


if __name__ == "__main__":
    main()
