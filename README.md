# ShopSphere
Y2S2 AI Deep Learning Project
# ShopSphere: 30-day purchase prediction

IT2226 AI Innovation and Deep Learning Project, team Shop Circle.

ShopSphere Global wants to know which customers will buy in the next 30 days, and to help
staff act on it within each country's budget and company policy. This repo holds our
supervised models, the agents that use them, and the staff dashboard.

| Member | Owns |
|---|---|
| Egus (lead) | `models/egus/` |
| Pravin | `models/pravin/` |
| Mahdiya | `models/mahdiya/` |
| Xin Ying | `models/xinying/` |

Agent owners for Progress Review 2 are decided after the model bake-off.

## Folder layout

```
data/            NOT committed. shopsphere.db, train.csv, test.csv go here locally
data_prep/       scripts that clean the raw data pack and build shopsphere.db
shared/          code every model uses: folds.csv, cutoff rules, features, evaluation
notebooks/       team notebooks: data exploration, PR1 checkpoint
models/          one folder per model, same rules for all (see models/README.md)
  baseline_lgbm/   team baseline 02: every model is compared with this
  egus/ pravin/ mahdiya/ xinying/
    notebooks/ results/ weights/ (weights not committed)
evaluation/
  models/        bake-off: every model scored the same way against the baseline
  agents/        agent test cases and results
agents/          targeting/ compliance/ analytics/ + llm/ (shared LLM wrapper)
dashboard/       Streamlit app for staff
deploy/          Dockerfile and scoring service (PR2 and Final)
outputs/         precomputed scores, weekly contact lists, approval/audit log
submissions/     test-set submissions
tests/           repo-wide tests
docs/            AI Use Logs, meetings, specs, reports, slides, ethics
```

## Rules everyone follows

1. **Data cutoff.** The data pack is as at 30 Sep 2024 and the target is a purchase between
   1 and 30 Oct 2024. No feature may use data dated after 30 Sep 2024.
2. **Validation.** Every model uses the shared 5 customer folds in `shared/folds.csv`.
   `test.csv` is never used for validation or model choice.
3. **Metrics.** Primary metric: precision@k, with k from the weekly retention budget
   (headline k: to be confirmed by the team). ROC-AUC and PR-AUC are reported alongside.
   Use the evaluation code in `shared/` so everyone's numbers are comparable.
4. **Nothing large or secret in Git.** Data stays in `data/`. Model weights are shared as a zip on
   a GitHub Release and linked from the model's README. API keys go in `.env`, never in code.
5. **Commits.** Commit from your own account, small and often, with clear messages. Work produced
   mostly by an AI tool goes in its own commit and matches an entry in your AI Use Log
   (`docs/ai_use_logs/<name>.md`).
6. **Branches.** Work on a branch named `<name>/<topic>` (e.g. `pravin/xgboost`) and open a
   pull request into `main`. Changes to `shared/` need a review from the team.

## Setup

```
git clone https://github.com/PravinRram/ShopSphere.git
cd ShopSphere
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

Put the data pack in `data/` (see `data/README.md`). To use a trained model, download its
weights zip from the repo's Releases page and unzip it into that model's `weights/` folder.

## Where things stand

| Milestone | Week | Status |
|---|---|---|
| Idea Presentation | 3 | done |
| Progress Review 1: supervised model | 5 | in progress |
| Progress Review 2: Gen AI / agentic capability | 14 | |
| Final presentation, report, code and version history | 18 | |