"""Export a trained run (e.g. E4T) into a self-contained model bundle for predict.py.

    python -m shopsphere_dl.export --run E4T --base E4 --params best_params.json \
        --ckpt-dir <dir with ckpt_E4T_seed*_fold*.pt> --oof-dir <dir with oof_E4T_seed*.csv> \
        --out models/e4t_v1
    python -m shopsphere_dl.export --run E7 --base E4 --params best_params.json --extra crm,geo ... --out models/e7_v1

The training checkpoints are for resuming training: they hold the optimiser and scheduler too,
but not the two things scoring needs besides the weights. This script adds them, per model
(5 seeds x 5 folds = 25 models):
- the fold's fitted preprocessor (medians, quantile transform, category codes). It is refitted
  exactly as train.train_fold fitted it: same training rows, same seed;
- the fold's temperature, read back from the out-of-fold file (score = sigmoid(logit / T)).
It then re-scores every fold's validation customers with the exported model and checks the
result against the out-of-fold file, so a wrong export cannot pass silently.

Bundle: one model_s<seed>_f<fold>.pt per model (best weights, preprocessor, temperature) and
manifest.json (settings, event vocabulary, check results).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from .dataset import SnapshotRows, StaticPreprocessor, loader
from .features_extra import feature_lists, parse_groups
from .folds import split
from .model import TwoBranchNet
from .sequences import CATEGORICAL as SEQ_CATEGORICAL, NUMERIC as SEQ_NUMERIC, TIME as SEQ_TIME
from .train import PRESETS, TUNABLE, Data, TrainConfig, _predict


def build_model(prep: StaticPreprocessor, vocab: dict, cfg: dict) -> TwoBranchNet:
    seq_cards = [len(vocab[c]) + 1 for c in SEQ_CATEGORICAL]
    return TwoBranchNet(prep.n_numeric, prep.cat_cardinalities, len(SEQ_NUMERIC), len(SEQ_TIME), seq_cards,
                        hidden=cfg["hidden"], gru_layers=cfg["gru_layers"], dropout=cfg["dropout"],
                        pooling=cfg["pooling"], use_static=cfg["use_static"], use_sequence=cfg["use_sequence"])


def temperature_from_oof(oof: pd.DataFrame) -> float:
    """Recover T from score = sigmoid(logit / T), using rows where both are far from 0.5."""
    s = oof.score.clip(1e-6, 1 - 1e-6)
    z = np.log(s / (1 - s))
    ok = (oof.logit.abs() > 1) & (z.abs() > 0.5)
    t = (oof.logit[ok] / z[ok]).median()
    assert np.isfinite(t) and t > 0, "could not recover the temperature"
    return float(t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="E4T")
    p.add_argument("--base", default="E4", choices=sorted(PRESETS))
    p.add_argument("--params", default=None, help="best_params.json used for the run (omit for an untuned run)")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--oof-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--extra", default="", help="extra feature groups the run was trained with, e.g. crm,geo")
    a = p.parse_args()
    extra = ",".join(parse_groups(a.extra))

    cfg = TrainConfig(run=a.run, **PRESETS[a.base])
    if a.params:
        with open(a.params) as f:
            for k, v in json.load(f)["params"].items():
                assert k in TUNABLE, k
                setattr(cfg, k, v)
    keep = ["hidden", "gru_layers", "dropout", "pooling", "use_static", "use_sequence", "n_events"]
    settings = {k: getattr(cfg, k) for k in keep}
    settings["extra"] = extra                                              # predict.py rebuilds the same columns

    data = Data(os.environ.get("SHOPSPHERE_DB", "shopsphere.db"), os.environ.get("SHOPSPHERE_FOLDS", "folds.csv"),
                extra=extra)
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cpu")
    checks = []
    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, f"ckpt_{a.run}_seed*_fold*.pt")))
    assert ckpts, f"no checkpoints for {a.run} in {a.ckpt_dir}"
    for path in ckpts:
        seed, fold = map(int, re.search(r"seed(\d+)_fold(\d+)", path).groups())
        oof = pd.read_csv(os.path.join(a.oof_dir, f"oof_{a.run}_seed{seed}.csv"))
        oof = oof[oof.fold == fold].set_index("customer_id")
        tr, va = split(data.rows, data.folds, fold, use_snapshots=cfg.use_snapshots)
        prep = StaticPreprocessor(*feature_lists(extra)).fit(data.rows[tr], seed=seed)   # exactly as in train_fold
        model = build_model(prep, data.store.vocab, settings)
        saved = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(saved["state"]["best_state"])
        t = temperature_from_oof(oof)

        # check: re-score this fold's validation customers and compare with the out-of-fold file
        valid = data.rows[va]
        logits, y = _predict(model.to(device), loader(SnapshotRows(valid, prep, data.store, cfg.n_events), 4096, False),
                             device)
        ref = oof.loc[valid.customer_id.values]
        diff = np.abs(logits - ref.logit.values)
        check = {"seed": seed, "fold": fold, "temperature": t, "customers": len(valid),
                 "max_abs_logit_diff": float(diff.max()), "mean_abs_logit_diff": float(diff.mean()),
                 "rank_corr": float(pd.Series(logits).corr(pd.Series(ref.logit.values), method="spearman")),
                 "pr_auc_export": float(average_precision_score(y, logits)),
                 "pr_auc_oof": float(average_precision_score(y, ref.logit.values))}
        checks.append(check)
        print(f"seed {seed} fold {fold}: T {t:.3f}, max |logit diff| {check['max_abs_logit_diff']:.4f}, "
              f"PR-AUC {check['pr_auc_export']:.4f} vs {check['pr_auc_oof']:.4f}", flush=True)
        assert check["rank_corr"] > 0.999, "exported model does not reproduce the out-of-fold scores"

        torch.save({"state_dict": {k: v.float() for k, v in model.state_dict().items()}, "prep": prep,
                    "temperature": t, "seed": seed, "fold": fold},
                   os.path.join(a.out, f"model_s{seed}_f{fold}.pt"))

    manifest = {"run": a.run, "base": a.base, "settings": settings, "models": len(checks),
                "event_vocab": {c: list(map(str, data.store.vocab[c])) for c in SEQ_CATEGORICAL},
                "score": "mean over models of sigmoid(logit / temperature)",
                "snapshot": "2024-09-30", "checks": checks}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"exported {len(checks)} models to {a.out}", flush=True)


if __name__ == "__main__":
    main()
