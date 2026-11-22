"""Compare E7 with E4T and the baselines, and re-run the bias check on E7's scores.

    PYTHONPATH=src python scripts/compare_e7.py --new-dir /kaggle/working --ref-dir oof --out /kaggle/working
    # E7T (re-tuned) against E7, whose out-of-fold files are in another folder:
    PYTHONPATH=src python scripts/compare_e7.py --new-dir . --new-run E7T --ref-run E7 --ref-run-dir ref_e7 --ref-dir oof

Reads (all out-of-fold files have columns customer_id, fold, seed, score):
  new-dir/oof_E7_seed*.csv          the new run (any number of seeds)
  ref-dir/oof_E4T_seed*.csv         the current model
  ref-dir/oof_02_post.csv, oof_03_post.csv          team baseline 02 and own baseline 03 (5-fold)
  ref-dir/holdout_8020_ids.csv      the team's 80/20 holdout: train_test_split(test_size=0.2,
                                    stratify=y, random_state=42) in the 02 notebook
  ref-dir/holdout_02_8020_seed42.csv  team baseline 02 retrained on that split (our rerun)

Writes compare_5fold_<run>.csv, compare_holdout_<run>.csv, bootstrap_<run>.csv, bias_<run>.csv.
The team baseline is kept separate from Egus's own baseline (03) throughout.
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3

import numpy as np
import pandas as pd

from shopsphere_dl.evaluate import evaluate, paired_bootstrap, paired_bootstrap_metric
from shopsphere_dl.fairness import allocate

TEAM_REPORTED = {   # as reported by the team on 17 Nov 2026 (not reproduced; our rerun is listed too)
    "Team baseline 02 (team-reported)": {"ROC-AUC": 0.8752, "PR-AUC": 0.5725},
    "Xin Ying TabularNN (as reported)": {"ROC-AUC": 0.8740, "PR-AUC": 0.5728},
}
K_FULL = (1000, 2500, 2950, 6709)
K_HOLD = (200, 500)        # same share of customers on 12,800 as k = 1000 / 2500 on 64,000
LIST_K = 6709              # weekly list size used for the bias check


def seed_files(directory: str, run: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(directory, f"oof_{run}_seed*.csv")),
                   key=lambda f: int(f.rsplit("seed", 1)[1].split(".")[0]))
    assert files, f"no oof_{run}_seed*.csv in {directory}"
    return files


def mean_score(files: list[str]) -> pd.Series:
    return pd.concat([pd.read_csv(f) for f in files]).groupby("customer_id").score.mean()


def metrics(y: pd.Series, s: pd.Series, ks) -> dict:
    s = s.reindex(y.index)
    assert s.notna().all(), "scores missing for some customers"
    return evaluate(y.values, s.values, ks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new-dir", required=True)
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--out", default=".")
    p.add_argument("--new-run", default="E7")
    p.add_argument("--ref-run", default="E4T", help="the run the new one is bootstrapped against")
    p.add_argument("--ref-run-dir", default=None, help="folder with oof_<ref-run>_seed*.csv (default: --ref-dir)")
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    db = os.environ.get("SHOPSPHERE_DB", "shopsphere.db")
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    y = pd.read_sql_query("SELECT customer_id, target FROM train", con).set_index("customer_id").target
    prof = pd.read_sql_query("""SELECT c.customer_id, c.age, g.region_tier FROM customers c
                                LEFT JOIN geo_data g ON g.geo_ip_region = c.region""", con).set_index("customer_id")
    con.close()

    new_files = seed_files(a.new_dir, a.new_run)
    ref_files = seed_files(a.ref_run_dir or a.ref_dir, a.ref_run)
    n = len(new_files)
    plural = "s" if n > 1 else ""
    assert len(ref_files) >= n, f"{a.ref_run} has {len(ref_files)} seeds, fewer than {a.new_run}'s {n}"
    models = {
        f"{a.new_run} ({n} seed{plural})": mean_score(new_files),
        f"{a.new_run} seed 1": mean_score(new_files[:1]),
        f"{a.ref_run} ({n} seed{plural}, same count)": mean_score(ref_files[:n]),
        f"{a.ref_run} seed 1": mean_score(ref_files[:1]),
    }
    e4t = seed_files(a.ref_dir, "E4T")
    models["E4T (5 seeds)"] = mean_score(e4t)
    if a.ref_run != "E4T":
        models["E4T seed 1"] = mean_score(e4t[:1])
    models["Own baseline 03 (LightGBM)"] = mean_score([os.path.join(a.ref_dir, "oof_03_post.csv")])
    models["Team baseline 02 (5-fold)"] = mean_score([os.path.join(a.ref_dir, "oof_02_post.csv")])
    new, ref = models[f"{a.new_run} ({n} seed{plural})"], models[f"{a.ref_run} ({n} seed{plural}, same count)"]

    # 1. Five-fold, all 64,000 customers
    rows = [{"model": k, **{m: round(v, 4) for m, v in metrics(y, s, K_FULL).items()
                            if m in ("ROC-AUC", "PR-AUC", "ECE", "mean_score") or m.startswith("P@")}}
            for k, s in models.items()]
    five = pd.DataFrame(rows)
    five.to_csv(os.path.join(a.out, f"compare_5fold_{a.new_run.lower()}.csv"), index=False)
    print("\n5-fold, all 64,000 customers\n", five.to_string(index=False))

    # 2. The team's 80/20 holdout (12,800 customers), with the team's rule: ROC-AUC change + PR-AUC change
    ids = pd.read_csv(os.path.join(a.ref_dir, "holdout_8020_ids.csv")).customer_id
    yh = y.loc[ids]
    hold = {k: metrics(yh, s, K_HOLD) for k, s in models.items() if "5-fold" not in k}
    hold["Team baseline 02 (our rerun of the notebook)"] = metrics(
        yh, pd.read_csv(os.path.join(a.ref_dir, "holdout_02_8020_seed42.csv")).set_index("customer_id").score, K_HOLD)
    hold.update(TEAM_REPORTED)
    hrows = []
    for ref_name in ["Team baseline 02 (team-reported)", "Team baseline 02 (our rerun of the notebook)"]:
        b = hold[ref_name]
        for k, m in hold.items():
            if k.startswith("Team baseline 02"):
                continue
            hrows.append({"against": ref_name, "model": k, "ROC-AUC": round(m["ROC-AUC"], 4), "PR-AUC": round(m["PR-AUC"], 4),
                          "ROC change": round(m["ROC-AUC"] - b["ROC-AUC"], 4), "PR change": round(m["PR-AUC"] - b["PR-AUC"], 4),
                          "team rule sum": round(m["ROC-AUC"] - b["ROC-AUC"] + m["PR-AUC"] - b["PR-AUC"], 4)})
    holdout = pd.DataFrame(hrows)
    holdout.to_csv(os.path.join(a.out, f"compare_holdout_{a.new_run.lower()}.csv"), index=False)
    print("\nTeam 80/20 holdout, 12,800 customers (team rule = ROC change + PR change)\n", holdout.to_string(index=False))

    # 3. Paired bootstrap: new run against the reference run with the same number of seeds.
    #    Folds 1-4 leave out fold 0, which the tuning used to choose settings.
    folds = pd.read_csv(os.environ.get("SHOPSPHERE_FOLDS", "folds.csv")).set_index("customer_id").fold
    y14 = y[folds.reindex(y.index) != 0]
    out = []
    for label, yy in [("5-fold", y), ("folds 1-4 (no tuning fold)", y14), ("80/20 holdout", yh)]:
        aa, bb = new.reindex(yy.index).values, ref.reindex(yy.index).values
        for metric in ("ROC-AUC", "PR-AUC"):
            out.append({"evaluation": label, **paired_bootstrap_metric(yy.values, aa, bb, metric, n_boot=a.n_boot)})
        for k in (K_FULL[:2] if label == "5-fold" else K_HOLD if label == "80/20 holdout" else (800, 2000)):
            r = paired_bootstrap(yy.values, aa, bb, k, n_boot=a.n_boot)
            out.append({"evaluation": label, "metric": f"P@{k}", **{c: v for c, v in r.items() if c != "k"}})
    boot = pd.DataFrame(out)
    tag = a.new_run.lower()
    boot.to_csv(os.path.join(a.out, f"bootstrap_{tag}.csv"), index=False)
    print(f"\n{a.new_run} vs {a.ref_run} (same seed count), paired bootstrap\n", boot.round(4).to_string(index=False))

    # 4. Bias check on the new scores: calibration and recall by age band and region tier,
    #    then equal-opportunity allocation by age (claude/bias_mitigation.md)
    d = prof.reindex(y.index).assign(target=y, p=new.reindex(y.index))
    d["age_band"] = pd.cut(d.age, [0, 24, 34, 44, 54, 200], labels=["<25", "25-34", "35-44", "45-54", "55+"]).astype(str)
    d["region_tier"] = d.region_tier.fillna("unknown")
    plain = d.p.rank(ascending=False, method="first") <= LIST_K
    fair = allocate(d.p, d.age_band, LIST_K, alpha=1.0)
    brow = []
    for col in ("age_band", "region_tier"):
        for g, x in d.groupby(col):
            buyers = x.target == 1
            brow.append({"group": col, "value": g, "customers": len(x), "actual_rate": round(x.target.mean(), 4),
                         "mean_prediction": round(x.p.mean(), 4),
                         "recall_plain": round(plain[x.index][buyers].mean(), 3),
                         "recall_age_fair": round(fair[x.index][buyers].mean(), 3)})
    bias = pd.DataFrame(brow)
    bias.to_csv(os.path.join(a.out, f"bias_{tag}.csv"), index=False)
    print("\nBias check on", a.new_run, "scores (list of", LIST_K, ")\n", bias.to_string(index=False))
    for col in ("age_band", "region_tier"):
        x = bias[bias.group == col]
        print(f"{col}: recall gap {x.recall_plain.max() - x.recall_plain.min():.3f} plain, "
              f"{x.recall_age_fair.max() - x.recall_age_fair.min():.3f} after age allocation")
    print(f"list precision: plain {d.target[plain].mean():.4f}, age allocation {d.target[fair].mean():.4f}")


if __name__ == "__main__":
    main()
