# My own baseline: LightGBM (03)

A tuned LightGBM on 48 customer features, built before the GRU + MLP to answer one question: is a deep
learning model worth it over a strong tree model? It is separate from the team baseline `02`
(`models/baseline_lgbm/`), which every member's model was compared with in the bake-off.

## Files
| File | What it is |
|---|---|
| `03_lightgbm_baseline_final.ipynb` | The notebook as run on 5 Nov 2026: feature SQL, Optuna tuning, 5 seeds × 5 folds, calibration and fairness checks |
| `rerun_03_strict_cutoff.py` | Re-runs the notebook's own feature code under the lecturer's strict cutoff ruling (see below) |
| `oof_03_strict.csv` | Out-of-fold scores from that re-run (`customer_id, fold, seed, score`; 5 seeds) |

## The strict cutoff ruling
The notebook was written the same day as the lecturer's ruling (5 Nov 2026), so it still counts two things
the ruling excludes: retention outcomes for contacts after 16 Sep 2024 (their 14-day window runs past the
30 Sep cutoff), and delivery delays for orders promised after 30 Sep. `rerun_03_strict_cutoff.py` applies
exactly those four fixes to the notebook's SQL and nothing else. The scores barely moved (ROC-AUC 0.8744
before and after, mean of seeds), and every number below comes from the re-run.

## Results (strict cutoff)
| | ROC-AUC | PR-AUC | P@1000 | P@2500 |
|---|---|---|---|---|
| All 64,000 customers, 5-fold, 5 seeds averaged | 0.8745 | 0.5739 | 0.870 | 0.769 |
| Team 80/20 holdout (12,800), 5 seeds averaged | 0.8762 | 0.5761 | | |
| Team 80/20 holdout, seed 1 | 0.8762 | 0.5762 | | |

## Against the GRU + MLP (E7)
Paired bootstrap, both models averaged over 5 seeds, same customers (`../results/bootstrap_e7_vs_03.csv`).
A difference counts only if its whole 95% interval is above zero.

| Evaluation | Measure | E7 − 03 | Verdict |
|---|---|---|---|
| 5-fold, 64,000 customers | PR-AUC | +0.0063 | E7 wins |
| | P@2500 | +0.017 | E7 wins |
| | P@6709 (weekly budget list) | +0.006 | E7 wins |
| | ROC-AUC | +0.0007 | tie |
| | P@1000 | +0.014 | tie |
| Holdout, 12,800 customers | ROC-AUC | +0.0010 | tie |
| | PR-AUC | +0.0023 | tie |

So the GRU + MLP's gain over a well-tuned LightGBM is real but small, and it shows in how well the contact
list is ranked rather than in ROC-AUC. On the holdout with one seed each the two are level (E7 0.8761,
03 0.8762 ROC-AUC); the larger 5-fold test is where the difference is measurable.

## How to run
```bash
SHOPSPHERE_DB=path/to/shopsphere.db python rerun_03_strict_cutoff.py 5
```
Uses `../folds.csv`. One seed took 33 seconds on a 2-core machine; the first seed reproduces
`oof_03_strict.csv` exactly (checked 21 Nov 2026).
