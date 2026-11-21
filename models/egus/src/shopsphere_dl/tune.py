"""Hyperparameter search for the GRU + MLP (E4) with Optuna, on one fold.

    python -m shopsphere_dl.tune --trials 30 --hours 3.5          # Kaggle GPU (E4T, 13 Nov)
    python -m shopsphere_dl.tune --extra crm,geo --tag E7T --start best_params.json --space wide
        # E7T: re-tune for E7's features; trial 0 = E7's current (E4T) settings; files *_E7T.*

What it searches (training plan, section 5, plus the epoch cap from experiment_results.md
finding 21):
    hidden size {32, 64, 128}, GRU layers {1, 2}, dropout 0-0.4, learning rate 3e-4 to 3e-3
    (log scale), events per customer N {48, 82, 128}, epoch cap {20, 30}

How it decides:
- Every trial trains on fold 0's training customers (all snapshots) and is scored on fold 0's
  validation customers at the real 30 Sep snapshot, like any other run. The objective is
  validation PR-AUC: it uses the whole ranking, so it is less noisy than precision@k on one
  fold's 12,800 customers. Precision at the fold-sized equivalents of k = 1,000 and 2,500
  (200 and 500 on one fold) is stored with each trial for the write-up.
- Trial 0 is the starting settings, so the table shows what tuning gained over them.
- TPE sampler (seeded) proposes settings; a median pruner stops a trial early when its
  validation PR-AUC is below the median of earlier trials at the same epoch.
- One seed per trial. Differences smaller than seed noise (PR-AUC sd ~0.0015 for E4) can
  be luck, which is why the chosen settings must then beat the untuned E4 over 5 folds x
  5 seeds before they replace it.
- Fold 0 is used to choose the settings, so fold 0's final score is slightly optimistic;
  the final comparison is also reported on folds 1-4 only.

Resumable: the study lives in tune_study.db (SQLite). If a session ends, attach the output
as an input and run again: finished trials are kept, an interrupted one is marked failed.
Outputs: tune_study.db, tune_trials.csv, best_params.json (read by train.py --params),
and tune/history_tune<N>_seed<S>_fold0.json for each trial.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from sklearn.metrics import average_precision_score

from .evaluate import precision_at_k
from .folds import split
from .features_extra import parse_groups
from .train import PRESETS, Data, TrainConfig, train_fold

STARTING = dict(hidden=64, gru_layers=2, dropout=0.2, lr=1e-3, n_events=82, max_epochs=20)

# "e4": the space E4T was tuned in (13 Nov). "wide": E4T's best settings sat on three edges of that
# space (hidden 128, 128 events, dropout ~0), so it extends those edges and drops the small sizes.
SPACES = {
    "e4": dict(hidden=[32, 64, 128], gru_layers=[1, 2], dropout=(0.0, 0.4), lr=(3e-4, 3e-3),
               n_events=[48, 82, 128], max_epochs=[20, 30]),
    "wide": dict(hidden=[64, 128, 192], gru_layers=[1, 2], dropout=(0.0, 0.3), lr=(3e-4, 3e-3),
                 n_events=[82, 128, 160], max_epochs=[20, 30]),
}


def suggest(trial: optuna.Trial, space: str = "e4") -> dict:
    s = SPACES[space]
    return dict(
        hidden=trial.suggest_categorical("hidden", s["hidden"]),
        gru_layers=trial.suggest_categorical("gru_layers", s["gru_layers"]),
        dropout=trial.suggest_float("dropout", *s["dropout"]),
        lr=trial.suggest_float("lr", *s["lr"], log=True),
        n_events=trial.suggest_categorical("n_events", s["n_events"]),
        max_epochs=trial.suggest_categorical("max_epochs", s["max_epochs"]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="E4", choices=sorted(PRESETS))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--trials", type=int, default=30, help="total trials in the study, including earlier sessions")
    p.add_argument("--hours", type=float, default=3.5, help="stop starting new trials after this many hours")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--sample", type=int, default=None, help="N random customers (rehearsal only)")
    p.add_argument("--extra", default="", help="extra static feature groups, e.g. crm,geo (features_extra.py)")
    p.add_argument("--tag", default=None, help="names the study and output files, e.g. E7T -> best_params_E7T.json")
    p.add_argument("--start", default=None, help="best_params.json whose settings become trial 0")
    p.add_argument("--space", default="e4", choices=sorted(SPACES))
    a = p.parse_args()
    extra = ",".join(parse_groups(a.extra))
    suffix = f"_{a.tag}" if a.tag else ""
    start = STARTING
    if a.start:
        with open(a.start) as f:
            start = json.load(f)["params"]
        s = SPACES[a.space]
        assert start["hidden"] in s["hidden"] and start["n_events"] in s["n_events"], "--start is outside --space"

    out = os.environ.get("SHOPSPHERE_OUT", ".")
    db = os.environ.get("SHOPSPHERE_DB", "shopsphere.db")
    work = os.path.join(out, f"tune{suffix}"); os.makedirs(work, exist_ok=True)
    ids = None
    if a.sample:
        from .snapshots import load_train_labels
        ids = load_train_labels(db).sample(a.sample, random_state=0).customer_id
    data = Data(db, os.environ.get("SHOPSPHERE_FOLDS", "folds.csv"), customer_ids=ids, extra=extra)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, va = split(data.rows, data.folds, a.fold)
    k_small = [max(1, round(k * int(va.sum()) / 64000)) for k in (1000, 2500)]    # 200 and 500 on a full fold

    storage = f"sqlite:///{os.path.join(out, f'tune_study{suffix}.db')}"
    name = f"{a.tag or a.run}_fold{a.fold}"
    try:                                     # a resumed session reseeds the sampler, so it does not repeat trials
        done = len(optuna.load_study(study_name=name, storage=storage).trials)
    except KeyError:
        done = 0
    study = optuna.create_study(
        study_name=name, storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=a.seed * 1000 + done, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4))
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):     # cut off by a session limit
        study.tell(t.number, state=TrialState.FAIL)
        print(f"trial {t.number} was interrupted; marked failed", flush=True)
    if not study.trials:
        study.enqueue_trial(start)                                                # trial 0 = starting settings

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, a.space)
        cfg = TrainConfig(run=f"tune{trial.number}", seed=a.seed, folds=[a.fold], num_workers=a.num_workers,
                          extra=extra, **PRESETS[a.run], **params)
        print(f"\ntrial {trial.number}: {params}", flush=True)

        def report(epoch, pr):
            trial.report(pr, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        t0 = time.time()
        try:
            oof = train_fold(data, cfg, a.fold, work, device, epoch_callback=report)
        finally:
            for ck in glob.glob(os.path.join(work, f"ckpt_tune{trial.number}_*.pt")):
                os.remove(ck)                                                     # keep the output small
            trial.set_user_attr("minutes", round((time.time() - t0) / 60, 1))
        y, s = oof.target.values, oof.logit.values
        pr = float(average_precision_score(y, s))
        for k in k_small:
            trial.set_user_attr(f"P@{k}_fold", precision_at_k(y, s, k))
        print(f"trial {trial.number}: PR-AUC {pr:.4f} in {(time.time() - t0) / 60:.1f} min", flush=True)
        return pr

    remaining = a.trials - len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, timeout=a.hours * 3600, catch=(RuntimeError,))

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))
    df.to_csv(os.path.join(out, f"tune_trials{suffix}.csv"), index=False)
    best = study.best_trial
    start = study.trials[0]
    result = {"params": best.params, "valid_pr_auc_fold": best.value, "trial": best.number,
              "starting_settings_pr_auc_fold": start.value if start.state == TrialState.COMPLETE else None,
              "fold": a.fold, "seed": a.seed, "run": a.run, "extra": extra, "tag": a.tag, "space": a.space,
              "trials": {s.name: int((df.state == s.name).sum()) for s in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)}}
    with open(os.path.join(out, f"best_params{suffix}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print("\nbest:", json.dumps(result, indent=1), flush=True)
    try:
        imp = optuna.importance.get_param_importances(study)
        print("importance:", {k: round(v, 3) for k, v in imp.items()}, flush=True)
    except Exception as e:                                                        # needs several finished trials
        print("importance not available:", e, flush=True)


if __name__ == "__main__":
    main()
