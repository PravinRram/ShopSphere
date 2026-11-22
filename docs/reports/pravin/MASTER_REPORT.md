# MASTER REPORT: Data Cleaning, EDA, Baseline and FT-Transformer (Progress Review 1)

**Course:** IT2226 AI Innovation and Deep Learning Project | **Review:** Progress Review 1 (Week 5)
**Student:** [Student name] | **Admin no.:** [Admin no.] | **Team:** [Team name] | **Tutor:** [Tutor name] | **Repository:** [Repository link]
**Data snapshot:** 30 September 2024 | **Prediction target:** purchase between 1 and 30 October 2024

---

## 0. HOW TO READ THIS DOCUMENT (instructions for an AI reader)

- This document is the **only source of facts** about the student's work. Do not use outside knowledge to add, change or round any result.
- Section 2 is the **KEY FACTS SHEET**: every number on a slide or in a script must come from it (each fact has an ID such as F12).
- Section 12 maps every marking criterion to its evidence. Section 13 lists the figure files available for the slide. Section 14 lists claims that must NOT be made.
- Square brackets such as [Student name] are placeholders the student will fill in. Leave them as they are.
- The data is **synthetic** (created for teaching). ShopSphere Global is a fictional company.

---

## 1. CONTEXT

### 1.1 The project in three sentences
ShopSphere Global is a fictional online retailer. Its CRM team builds a weekly "win-back" list by hand and often sends discounts to customers who would have bought anyway. The project asks the team to predict which customers will purchase in the next 30 days (supervised deep learning) and to build an AI agent that turns those predictions into policy-compliant, budget-aware recommendations. This report covers one team member's work up to Progress Review 1: data cleaning and EDA, a fair baseline, and the FT-Transformer deep learning model.

### 1.2 What the student did (scope of "my part")
1. **Cleaned and explored four customer files:** customers.csv, customer_preferences.csv, train.csv, test.csv (notebook 01).
2. **Built the FT-Transformer** (a deep model for tables) in PyTorch, with leakage-safe per-fold preprocessing, 5-fold cross-validation, and a seed-noise study (notebook 03 and a seed-study notebook).
3. **Audited and corrected the team's LightGBM baseline** so that the deep model is compared fairly, and compared the two models (notebooks 02 and the comparison notebook).
The other tables (sessions, transactions, deliveries, support tickets, retention contacts) were cleaned by other team members and are used through the team's agreed database. Other team members' models are out of scope for this report and must not be mentioned.

### 1.3 The presentation this document supports
- Progress Review 1, Week 5. Individual presenter, **5 minutes**, **one slide**, shown to a tutor or panel.
- The tutor separately runs an unprepared code walkthrough and a live coding task, so the slide does not need to contain code.
- The slide is the student's individual part of the team deck. Team-level items (overall progress update, revised plan) belong to the group and are out of scope.

### 1.4 Key messages the audience should remember
1. **The keys are clean, the columns are messy.** Every issue was found, fixed or deliberately kept, logged and validated.
2. **The EDA found the signals the model later confirmed** (loyalty, tier, engagement), and each finding changed a decision (no imputation in files, missing-value flags, probability metrics instead of accuracy).
3. **A fair baseline came first.** The LightGBM baseline had four defects; they were corrected before any comparison.
4. **The FT-Transformer was built from scratch** with a leakage-safe per-fold pipeline and documented optimisation evidence.
5. **Evaluation is honest and in depth:** slightly but reliably ahead of LightGBM, well calibrated, accuracy shown to be misleading, limits stated.
6. **The code is organised and reproducible** (three notebooks, a CONFIG cell, assertions, logs, shared files), and the next steps are clear.

---

## 2. KEY FACTS SHEET (the only numbers allowed on the slide or in the script)

### 2.1 Task and data
| ID | Fact |
|---|---|
| F01 | Task: predict target = 1 (at least one purchase between 1 and 30 October 2024). 80,000 customers: 64,000 labelled (train) and 16,000 unlabelled (test). Data snapshot 30 September 2024. |
| F02 | Class balance: 9,326 buyers out of 64,000 = 14.57%; about 5.9 non-buyers per buyer. Predicting "no purchase" for everyone is 85.43% accurate but useless. |
| F03 | Data source: the ShopSphere data pack provided by the teaching team (synthetic): 17 data tables and 4 policy documents. **No external data was used.** |
| F04 | Tables used by the model: 7 (customers, customer_preferences, transactions, sessions, deliveries, support_tickets, retention_contact_history), plus the train and test labels. |
| F05 | The student cleaned 4 files: customers (80,000 rows, 17 raw columns, 14 clean), customer_preferences (80,000 rows, 7 columns), train (64,000 rows, 2 columns), test (16,000 rows, 1 column). |
| F06 | Key integrity: 12 integrity checks (duplicate IDs, duplicate rows, train/test overlap, orphan IDs, targets outside 0 or 1) all returned 0 problems. Joins are one-to-one with no row loss. |

### 2.2 Cleaning (notebook 01)
| ID | Fact |
|---|---|
| F07 | country_code had 15 spellings of 5 countries (US, UK, DE, FR, IN), 54,721 rows affected (68.4%). Standardised. Clean counts: US 31,956; UK 16,075; DE 15,822; FR 8,096; IN 8,051. |
| F08 | age: the text value "unknown" in 4,119 rows (5.1%). Set to missing and flagged with age_is_missing. |
| F09 | email_open_rate: 14,559 blanks (18.2%). Left missing and flagged with email_open_rate_is_missing. |
| F10 | avg_review_score: 313 values (0.4%) above the 1 to 5 scale (maximum 6.5, mean about 5.8). Capped to 5.0. Evidence: the capped group buys at 11.0% vs 11.7% for customers at exactly 5.0 (chi-square p = 0.73), so they behave like top scorers. Trade-off: capping is an imputation without a flag. |
| F11 | credit_limit stored as text with $ and commas (for example "$17,757"): converted to numeric (80,000 rows). Opt-in and do-not-contact fields were Y/N text: converted to 1/0 (80,000 rows). |
| F12 | Five noise_customer_* columns carry no signal (single-feature AUC 0.497 to 0.508; one nominal p = 0.013 does not survive a Bonferroni correction). Dropped from the saved file. customer_id has AUC 0.497, so it carries no signal either. |
| F13 | Placeholders kept and documented, not changed: 14,122 signups on 2022-01-01 (17.7%, the first day of history, probably left-censored); 14,005 preference records with last_updated = 2024-09-30 (17.5%, the cutoff day); 5,125 customers at age 18 (6.4%, about six times the count at ages 19 and 20). |
| F14 | Consent conflict: 20,595 customers (25.7%) list a preferred channel they have not opted in to (12,792 email, 7,803 SMS). Kept unchanged and flagged for the agent. All 1,634 do_not_contact = Y customers have both opt-ins set to N (consistent). App push and WhatsApp have no opt-in column, so consent for them cannot be checked. |
| F15 | Cleaning policy (agreed with the team): fix formats now; **do not impute in the files** (impute inside the model pipeline, fitted on training folds only, so nothing from validation or test leaks into training); one deliberate exception (the review-score cap); raw data is never modified; every change is written to a cleaning log. |
| F16 | Cleaning log: 14 entries (file, column, issue, action, rows affected). Validation assertions all passed (unique keys, five countries only, value ranges, no future dates, indicators match missing counts, no review score above 5). Remaining missing values are intentional: age 4,119 and email_open_rate 14,559. |
| F17 | The cleaning notebook rebuilds the four cleaned files from the raw data. Outputs: customers_clean 80,000 x 14; customer_preferences_clean 80,000 x 7; train_clean 64,000 x 2; test_clean 16,000 x 1; cleaning_log 14 rows. |
| F18 | Train vs test shift check over 16 features (KS test for numeric, chi-square for categorical): largest effect size 0.012 (negligible). Test looks like a random split of the same population. Every date is on or before the cutoff, so no feature was recorded inside the prediction window. |

### 2.3 EDA findings (notebook 01)
| ID | Fact |
|---|---|
| F19 | loyalty_score is the strongest single predictor: AUC 0.795 on its own. The top loyalty decile purchases at 48%, the bottom decile at 0.9%. |
| F20 | customer_tier is the strongest categorical signal (Cramer's V 0.411): purchase rate 5.4% Bronze, 19.3% Silver, 30.9% Gold, 52.7% Platinum. Tier is based on 12-month spend up to the cutoff, so it is legitimate but overlaps with transaction information. |
| F21 | Weaker signals: email_open_rate AUC 0.591; discount_usage_rate AUC 0.581; email_opt_in purchase rate 17.2% vs 9.0%. credit_limit is indistinguishable from random (AUC 0.502). |
| F22 | Missingness is informative: customers with a blank email_open_rate buy at 11.8% vs 15.2%; the 2022-01-01 placeholder signups also buy at 11.8% vs 15.2%. This is why indicator columns are kept instead of silently imputing. |
| F23 | Region carries real signal: 100 regions, purchase rates from 8.0% to 22.0%; the spread across regions (standard deviation 2.5 points) is almost double what chance would give (1.4 points); chi-square p = 1.1e-21; Cramer's V 0.068. Country, language and preferred channel barely move the rate (about 13.8% to 15.0%). |
| F24 | Newer signups buy more: 12.5% (2022), 15.5% (2023), 16.7% (2024). tenure_days and days_since_pref_update are 0.94 correlated (redundant). |
| F25 | Language follows country exactly except India, which splits between English (3,997) and Hindi (4,054). |

### 2.4 Features and validation design
| ID | Fact |
|---|---|
| F26 | The model uses **47 engineered features**, computed as at 30 September 2024 from 7 tables (see section 5): profile and engagement 10; consent and channel 4; order history 10; browsing 8; delivery 4; support 4; retention contacts 3; derived ratios and flags 4. **14 of the 47 come directly from the cleaned customer tables** (10 from customers, 4 from customer_preferences), which the student's cleaning notebook rebuilds from the raw data (F17). |
| F27 | Validation: 5-fold stratified cross-validation using one shared fold file (folds.csv, StratifiedKFold, seed 42) used by every model. Each fold has 12,800 customers with a 14.57% to 14.58% purchase rate. Each fold model predicts its held-out fold, giving out-of-fold (OOF) predictions for all 64,000 customers; test predictions average the five fold models. |
| F28 | Leakage controls: (a) all features use data up to the cutoff; (b) of 1,419 retention contacts after 16 September, none records an October-only purchase (0 October-only; 91 flagged purchases); (c) 0 deliveries resolve after 30 September; (d) all preprocessing is fitted on the training fold only; (e) the SQL-built feature table was verified identical to the shared features_v1.parquet that the deep model reads. |

### 2.5 Baseline audit and correction (LightGBM)
| ID | Fact |
|---|---|
| F29 | The LightGBM baseline (gradient-boosted trees, same 47 features and folds) had four defects: (1) late_deliveries_count was always 0 because the SQL matched "Late" while the data uses "Delivered Late" (fixed; guarded by an assertion; 26,799 late deliveries in both the feature and the deliveries table); (2) early stopping watched AUC and log-loss together, and with class weights the log-loss worsened within about 18 trees, so the model was undertrained; (3) class weights inflated predicted probabilities (Brier 0.145 vs 0.088); (4) subsample had no effect because bagging_freq was unset. |
| F30 | Effect of the corrections (pooled OOF, seed 42): original reported baseline ROC-AUC 0.87025 / PR-AUC 0.54194; variant A (feature and bagging fixes, stopping on AUC + log-loss, class-weighted) 0.87165 / 0.55359 at 18.0 trees (Brier 0.1045); variant B (stopping on AUC only) 0.87340 / 0.56205 at 132.8 trees (Brier 0.1455); final (B without class weights) 0.87400 / 0.57148 at 147.8 trees (Brier 0.0880). PR-AUC rose from 0.542 to 0.571. |
| F31 | Against the original baseline the FT-Transformer appeared to lead by +0.031 pooled PR-AUC (0.5727 vs 0.5419); against the corrected baseline the pooled lead is +0.001. Fixing the baseline before comparing mattered more than any model tuning. |
| F32 | Final baseline settings: learning rate 0.03, 31 leaves, 80% row and feature sampling, early stopping on validation AUC (patience 100), no hyperparameter search. |

### 2.6 FT-Transformer design and training (notebook 03)
| ID | Fact |
|---|---|
| F33 | Architecture (written from scratch in PyTorch): each customer becomes 53 tokens (49 numeric inputs + 3 categorical inputs = 52 feature tokens, plus 1 learned [CLS] summary token); 3 transformer blocks; 4 attention heads; token size 64; feed-forward width 128; dropout 0.1; **108,161 trainable parameters**. Numeric token = x * w + b; categorical token = embedding lookup; the [CLS] vector feeds a head (LayerNorm, ReLU, Linear) that outputs a logit; sigmoid gives P(purchase in the next 30 days). Simplification vs the original paper: PyTorch's built-in pre-norm encoder layer with a GELU feed-forward block instead of ReGLU. |
| F34 | Preprocessing, fitted on each training fold only: median imputation plus a "was missing" flag for 5 columns (age, email_open_rate, recency_days, days_since_last_session, avg_csat_score); log1p on 30 skewed non-negative columns; standardise then clip at +/- 5 standard deviations; categorical columns (country, tier, preferred channel) as integer ids with id 0 for unseen values. 38 numeric + 5 flags + 6 binary = 49 numeric inputs. |
| F35 | Training setup: loss = binary cross-entropy on logits, **unweighted** (imbalance is mild and class weights would distort the probabilities the agent needs); AdamW, learning rate 3e-4, weight decay 1e-5; batch 512; at most 60 epochs; early stopping with patience 8 on validation ROC-AUC with best weights restored; gradient clipping at norm 1.0; Google Colab T4 GPU, PyTorch 2.14.1; about 1.5 minutes per fold. |
| F36 | Experiment log (fold 0 unless stated): run 1 smoke test (3 epochs, lr 1e-4): ROC-AUC 0.8693, PR-AUC 0.5612, pipeline works, not meaningful; run 2 (lr 1e-4, 60 epochs): 0.8801 / 0.5845, best epoch 53 of 60, 201 s; run 3 (lr 3e-4): 0.8800 / 0.5856, best epoch 22, 99 s; run 4 (lr 3e-4, all 5 folds, seed 42): fold mean 0.8757 / 0.5752; run 5 (seeds 43 and 44, same settings): fold means 0.8752 / 0.5741 and 0.8755 / 0.5749. |
| F37 | Optimisation evidence: (1) learning-rate diagnosis: run 2's best epoch (53 of 60) showed the optimiser was still converging, so lr was raised to 3e-4; the score did not change (0.8801 vs 0.8800) and training took half the time, which shows optimisation was not the bottleneck; (2) per-fold preprocessing with log transform and missing-value flags; (3) early stopping with best-weight restore; (4) gradient clipping; (5) a seed-noise study (3 FT seeds, 5 LightGBM seeds); (6) minimal tuning by design, no wider search. Disclosure: the lr was chosen after viewing fold 0's validation curve (a small peek at validation data; the score did not change). |

### 2.7 Evaluation results
| ID | Fact |
|---|---|
| F38 | Main comparison, mean over folds then over seeds (+/- is the standard deviation across seeds). **ROC-AUC:** LightGBM 0.8741 +/- 0.0001 (5 seeds) vs FT-Transformer 0.8754 +/- 0.0003 (3 seeds). **PR-AUC:** LightGBM 0.5720 +/- 0.0003 vs FT-Transformer 0.5747 +/- 0.0005. **Log-loss (pooled OOF):** 0.2827 vs 0.2821. **Brier score:** 0.0880 vs 0.0878. Standard deviation across folds (seed 42): ROC-AUC 0.0029 vs 0.0035; PR-AUC 0.0104 vs 0.0114. |
| F39 | Seed-noise study: the gap FT minus LightGBM is **+0.0014 ROC-AUC** (5.0 times the larger seed standard deviation, 0.16% relative) and **+0.0027 PR-AUC** (5.1 times, 0.5% relative). The ranges do not overlap. The FT-Transformer is ahead in **15 of 15 seed pairs** on the per-fold paired comparison (ROC-AUC gap +0.0010 to +0.0018; PR-AUC gap +0.0018 to +0.0036) and in all 5 folds for seed 42. Interpretation: reliable but small. Caveat: the five folds share most of their training data and the study measures model randomness only, so the ratio is not a formal significance test. |
| F40 | Calibration: predicted probabilities match observed purchase rates (the calibration curve sits on the diagonal; Brier about 0.088). This comes from using plain, unweighted loss and means the probabilities can be used for budget decisions. |
| F41 | Learning curves (fold 0, seed 42): training and validation loss both settle near 0.28 with almost no gap, so there is no overfitting and the model is limited by the information in the features, not by its size. |
| F42 | Operating points on the pooled OOF predictions (seed 42; LightGBM / FT-Transformer). Accuracy of predicting nobody: 85.43%. See the table in section 7.4. Top 5% of customers by predicted probability: 74.0% buy for the FT-Transformer (a lift of 5.1 times the 14.6% base rate). |
| F43 | Accuracy is a weak guide: at the default 0.50 threshold the models flag about 7% of customers and reach 87.86% / 87.98% accuracy (about 2.5 points above predicting nobody) but find only about a third of buyers; at the F1-optimal threshold (0.26 for LightGBM, 0.25 for FT) they flag 19.3% of customers and find about 63% to 64% of buyers, but accuracy falls to 84.50% / 84.63%, **below** the 85.43% of predicting nobody. Headline metrics are therefore ROC-AUC, PR-AUC, precision at k and calibration. |
| F44 | What drives predictions (out-of-fold SHAP on the LightGBM baseline, mean absolute contribution in log-odds): days_since_last_session 1.03; sessions_last_90d 0.43; loyalty_score 0.29; account_age_days 0.22; recency_days 0.09. Engagement dominates, in line with the EDA (loyalty_score AUC 0.80). The FT-Transformer probably relies on the same signals (SHAP for it is a next step). |

### 2.8 Source code and reproducibility
| ID | Fact |
|---|---|
| F45 | Three main notebooks plus two seed-study notebooks and one comparison notebook (section 8). Every notebook opens with a markdown section map, keeps settings in one CONFIG cell, uses small single-purpose functions, and includes assertions and checks. |
| F46 | Reproducibility: shared files folds.csv and features_v1.parquet; fixed seeds (FT 42 to 44, LightGBM 42 to 46); experiment logs in experiments.jsonl; OOF predictions saved; per-fold models and preprocessors saved. The baseline's seed 42 reproduced the main run exactly. |

---

## 3. PART A: INPUT DATA

### 3.1 Data source and integration ("collection")
- **Source:** the ShopSphere data pack supplied by the teaching team: 17 tables and 4 policy documents, all synthetic (F03). No external data was collected or used.
- **Integration:** the model draws on **7 tables** which are aggregated to one row per customer in SQL (SQLite) as at the 30 September 2024 cutoff, giving **47 features** (F04, F26). The team shares one database, one feature table (features_v1.parquet) and one fold file (folds.csv).
- **Why it is coherent:** all keys were audited (F06): customers, preferences and labels link one-to-one with no row loss. Every date is on or before the cutoff (F18), so nothing from the prediction window leaks into a feature.
- **Student's own scope:** the four customer files (F05). **14 of the 47 model features come straight from these cleaned tables** (F26), which the cleaning notebook rebuilds from the raw data (F17), so the cleaning work feeds the model directly.

### 3.2 The cleaning process (sophisticated and reproducible)
1. **Load raw as text** (everything read as strings first, so blanks, mixed spellings and "$" signs are visible before pandas converts anything).
2. **Audit** every file: keys and linkage (12 checks), blanks, text-stored numbers, category spellings, out-of-range values, impossible dates, pile-ups at range edges, contradictory fields.
3. **Clean** with small named functions (clean_customers, clean_preferences). Every change is written to a **cleaning log** (14 entries).
4. **Validate** with assertions (F16).
5. **Explore** (EDA) using training labels only; test is used only for the train-vs-test shift check (F18).
6. **Save** cleaned copies to a separate folder; the original data pack is never modified.

### 3.3 Issues found and how each was handled
| File | Column | Issue | Rows affected | Treatment |
|---|---|---|---|---|
| customers | country_code | 15 spellings of 5 countries | 54,721 (68.4%) | Standardised to US/UK/DE/FR/IN |
| customers | age | text "unknown" | 4,119 (5.1%) | Set to missing + age_is_missing |
| customers | age | pile-up at 18 (5,125 vs about 850 at 19 and 20) | 5,125 (6.4%) | Kept; noted in EDA |
| customers | signup_date | pile-up on 2022-01-01 (left-censored) | 14,122 (17.7%) | Kept; noted in EDA |
| customers | email_open_rate | blanks | 14,559 (18.2%) | Left missing + email_open_rate_is_missing |
| customers | avg_review_score | values above 5 (max 6.5) | 313 (0.4%) | Capped to 5.0 (evidence in F10) |
| customers | credit_limit | numbers stored as text | 80,000 | Converted to numeric |
| customers | referral_code | shared by 2+ customers (about what random 6-digit codes give by chance) | 6,900 (8.6%) | Kept (identifier only) |
| customers | noise_customer_0 to 4 | system fields, no signal | 5 columns | Tested, then dropped |
| preferences | opt-ins, do_not_contact | Y/N text | 80,000 | Converted to 1/0 |
| preferences | last_updated | pile-up on the cutoff date | 14,005 (17.5%) | Kept; noted in EDA |
| preferences | preferred_channel | email/SMS preferred without opt-in (consent risk) | 20,595 (25.7%) | Kept; flagged for the agent |
| train / test | customer_id, target | none found | 0 | Validated, cast to integer |

### 3.4 Exploratory data analysis: insight to decision
| EDA finding | Evidence | Decision it drove |
|---|---|---|
| Only 14.6% buy; accuracy is misleading | F02, figure A2 | Use ROC-AUC, PR-AUC, precision at k, calibration; stratified folds; accuracy only at stated operating points |
| loyalty_score and tier dominate; engagement matters | F19, F20, F21, figures A3, A4, A5, A6 | Keep both as features; expect the baseline to be strong, so a fair comparison is essential |
| Missingness is informative | F22 | Missing-value indicator columns; no silent imputation in the files |
| Placeholders pile up on boundary dates | F13, figure A1 | Kept and documented, not "fixed"; noted as a caution |
| Consent conflicts affect 25.7% of customers | F14, figure A7 | Passed to the agent: never choose a channel from preferred_channel alone |
| Region carries real signal (100 regions) | F23, figure A6 | Candidate for a learned embedding in a later feature experiment |
| Noise columns and customer_id carry no signal | F12 | Dropped; no ID feature |
| Test looks like train | F18 | Validation scores on train should carry over to test |
| India splits English and Hindi | F25 | Messages must use preferred_language (for the agent's drafting) |

---

## 4. PART B: SUPERVISED MODEL

### 4.1 Features and validation design
See F26 to F28. Feature groups:

| Feature group | No. | Source table | Examples |
|---|---|---|---|
| Profile and engagement | 10 | customers | age, loyalty_score, customer_tier, account_age_days, email_open_rate |
| Consent and channel | 4 | customer_preferences | email_opt_in, sms_opt_in, do_not_contact, preferred_channel |
| Order history | 10 | transactions | total_orders, total_spend_usd, orders_last_90d, recency_days |
| Browsing | 8 | sessions | sessions_last_90d, days_since_last_session, total_cart_additions |
| Delivery | 4 | deliveries | late_deliveries_count, lost_packages_count, avg_delay_days |
| Support | 4 | support_tickets | open_tickets_count, high_priority_open_tickets, avg_csat_score |
| Retention contacts | 3 | retention_contact_history | retention_contacts_count, retention_redeemed_count |
| Derived ratios and flags | 4 | derived | discount_order_ratio, bounce_rate, cart_abandonment_ratio, is_inactive_90d |

### 4.2 Why a fair baseline came first, and what was wrong with it
A deep model is only meaningful against a fair classical baseline. The LightGBM baseline had four defects (F29). The effect of fixing them:

| Baseline variant | Trees | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| Original reported baseline | n/a | 0.87025 | 0.54194 | n/a |
| A. Feature and bagging fixes; stopping on AUC + log-loss; class-weighted | 18.0 | 0.87165 | 0.55359 | 0.1045 |
| B. A, with stopping on AUC only | 132.8 | 0.87340 | 0.56205 | 0.1455 |
| Final: B without class weights | 147.8 | 0.87400 | 0.57148 | 0.0880 |

(Pooled OOF, seed 42. Brier scores are only comparable between unweighted models.) The corrections raised the baseline's PR-AUC from 0.542 to 0.571, and the FT-Transformer's apparent lead of +0.031 PR-AUC shrank to +0.001 pooled (F31).

### 4.3 The FT-Transformer in plain language
A transformer reads sequences of tokens, and a table row is not a sequence. The FT-Transformer therefore turns **every feature into a token** (a learned vector of 64 numbers) and adds one special **[CLS] token** that acts as a blank notepad. Through **self-attention**, every token can look at every other token (so the model can learn which features matter together). After three transformer blocks the [CLS] token summarises the whole customer, and a small head turns it into one number (a logit); a sigmoid converts that to a probability of purchase in the next 30 days. Details in F33 to F35. Figure B1 shows the architecture.

### 4.4 Why these design choices
- **FT-Transformer:** models interactions between features explicitly through attention, learns embeddings for categorical features, and (written from scratch in PyTorch) can be fully explained, modified and debugged. Reference: Gorishniy et al., 2021, "Revisiting Deep Learning Models for Tabular Data".
- **Per-fold preprocessing:** any statistic learned from all the data (medians, means, category lists) would leak validation information; fitting on the training fold only avoids this (F34).
- **Missing-value flags, log transform, clipping:** missingness is informative (F22), counts and money are heavy-tailed, and extreme values destabilise neural-network training.
- **Unweighted loss:** keeps probabilities calibrated (F35, F40).
- **Early stopping with best-weight restore and gradient clipping:** stable training and a model taken from its best epoch, not its last.

### 4.5 Training process and experiments
See F36 and F37 for the full experiment log and the optimisation evidence. In one sentence: a smoke test confirmed the pipeline, a longer run at lr 1e-4 showed the best epoch near the limit, raising lr to 3e-4 gave the same score in half the time, then all five folds and two extra seeds were run with identical settings.

---

## 5. DATA INTEGRATION DETAIL
### 5.1 Where the 47 features come from
Seven tables are aggregated to one row per customer as at 30 September 2024 using SQL: orders and recency (transactions), browsing (sessions), delivery problems (deliveries), support load and satisfaction (support_tickets), retention contact history (retention_contact_history), profile and consent (customers, customer_preferences). Four derived ratios and flags are added in Python. The same feature table and fold file are used by every model so that comparisons are fair.

## 6. RESULTS IN ONE PLACE
- **Ranking:** FT-Transformer 0.8754 ROC-AUC and 0.5747 PR-AUC vs corrected LightGBM 0.8741 and 0.5720 (F38).
- **Gap:** small but reliable (F39).
- **Probabilities:** well calibrated (F40).
- **Usefulness:** top 5% of customers by predicted probability: 74.0% buy (F42).
- **Accuracy:** misleading; shown only at stated operating points (F43).

## 7. EVALUATION IN DEPTH

### 7.1 Metrics used and why
- **ROC-AUC:** how well the model ranks buyers above non-buyers (0.5 = chance, 1.0 = perfect). Insensitive to class balance, so it can look flattering.
- **PR-AUC:** ranking quality focused on the minority class (buyers). Baseline value equals the base rate, 0.1457. More informative than ROC-AUC here.
- **Precision at k (and recall, lift):** if staff contact only the top k% of customers, what share actually buy? This is what the weekly win-back list depends on.
- **Calibration and Brier score:** do predicted probabilities match real purchase rates? Needed because the agent will weigh expected value against offer cost.
- **Out-of-fold evaluation:** every customer is scored by a model that never saw them.
- **Fold-to-fold spread:** the noise level from different customers. LightGBM and FT-Transformer ROC-AUC vary by about 0.003 across folds, much more than the 0.0014 gap, which is why the comparison uses paired per-fold differences and a seed study.

### 7.2 Results table (F38)
| Metric | LightGBM (5 seeds) | FT-Transformer (3 seeds) |
|---|---|---|
| ROC-AUC | 0.8741 +/- 0.0001 | 0.8754 +/- 0.0003 |
| PR-AUC | 0.5720 +/- 0.0003 | 0.5747 +/- 0.0005 |
| Log-loss (pooled OOF) | 0.2827 | 0.2821 |
| Brier score (pooled OOF) | 0.0880 | 0.0878 |

### 7.3 Interpretation (in depth)
- **The two models are practically comparable.** The FT-Transformer is ahead in 15 of 15 seed pairs, by +0.0014 ROC-AUC and +0.0027 PR-AUC, about 5 times the seed-to-seed spread, so the edge is reliable; but it is only 0.16% (ROC-AUC) and 0.5% (PR-AUC) relative, so it is small in practice (F39).
- **Why they are so close:** the learning curves show no overfitting and a plateau (F41), and SHAP shows engagement features dominate (F44). The information in the shared 47 features, not model capacity, is the likely limit. That is also why the EDA's loyalty finding (AUC 0.80 alone) predicted a strong baseline.
- **Calibration (F40):** plain loss keeps probabilities close to real purchase rates, so they can drive budget decisions. LightGBM's class-weighted variant did not (Brier 0.1455 vs 0.0880).
- **Optimism:** both models early-stop on the validation fold they are scored on, so both are slightly optimistic in the same way.

### 7.4 Operating points and accuracy (F42, F43)
Pooled OOF, seed 42. LGBM = LightGBM, FT = FT-Transformer. Accuracy of predicting nobody: 85.43%.

| Operating point | Customers flagged (LGBM / FT) | Precision (LGBM / FT) | Recall (LGBM / FT) | Accuracy (LGBM / FT) |
|---|---|---|---|---|
| Top 5% by probability | 3,200 | 73.3% / 74.0% | 25.2% / 25.4% | 87.76% / 87.83% |
| Probability at least 0.50 | 4,457 / 4,542 | 67.5% / 68.0% | 32.2% / 33.1% | 87.86% / 87.98% |
| Top 10% by probability | 6,400 | 61.0% / 61.3% | 41.9% / 42.0% | 87.63% / 87.68% |
| F1-optimal threshold (0.26 / 0.25) | 12,362 / 12,363 | 47.6% / 47.9% | 63.1% / 63.5% | 84.50% / 84.63% |

Accuracy rewards the cautious operating point (few customers flagged) and penalises the one that finds more buyers, so it is not a good headline. Which operating point to use is a business decision: it depends on the cost of an offer against the value of a recovered customer.

---

## 8. SOURCE CODE STRUCTURE

### 8.1 Notebooks (names as uploaded; adjust to the repository names)
| Notebook | Purpose | Structure |
|---|---|---|
| 01_EDA_and_Cleaning.ipynb (72 cells) | Audit, clean, validate and explore the four customer files | 8 numbered sections: setup; load raw as text; profile (keys, blanks, ranges, pile-ups, contradictions); cleaning functions + cleaning log; validation assertions; EDA (target, numeric, categorical, review capping, loyalty/region, noise columns, leakage, preferences); train-vs-test shift; save outputs and summary |
| 02_lightgbm_training_pipeline_v2 (and a v3 seed-study version) | Corrected LightGBM baseline | 12 sections: setup and CONFIG; SQL feature extraction; feature engineering and data checks; shared folds; cross-validation (run_cv); evaluation; diagnostics and out-of-fold SHAP; baseline-variant table; comparison with FT; saved outputs; customer demo; cheat sheet |
| 03_ft_transformer_training.ipynb | FT-Transformer training and evaluation | 10 sections: setup and CONFIG; load and sanity checks; TabularPreprocessor (fit/transform per fold); FeatureTokenizer and FTTransformer classes; training functions; 5-fold CV loop; evaluation; comparison; saved outputs; code-review cheat sheet |
| FT-Transformer seed-noise study (v2) | 3 seeds on identical folds | same notebook, seeds 42 to 44, plus per-seed results |
| Comparing_lightgbm_vs_ft_transformer.ipynb | Seed-noise comparison | paired per-fold gap over all 15 (FT seed, LightGBM seed) pairs |

### 8.2 Engineering practices that make the code well organised
- Section map in markdown at the top of every notebook; markdown explanations between code cells.
- **One CONFIG cell** holds every setting, so an experiment is one config plus one set of results.
- **Small single-purpose functions and classes:** clean_customers, clean_preferences, TabularPreprocessor (fit/transform), FeatureTokenizer, FTTransformer, train_one_fold, predict_proba, run_cv, evaluate_oof, log_run.
- **Assertions and checks at critical points:** cleaning validation; late-delivery total equals the deliveries table; model output-shape check; no non-finite values after preprocessing; shared features identical; folds identical to the seed-42 split.
- **Logs and shared artefacts:** cleaning_log.csv, experiments.jsonl (config, per-fold results, overall metrics, notes), folds.csv, features_v1.parquet, OOF files, saved fold models and preprocessors.
- **Reproducibility:** fixed seeds, shared folds, the baseline's seed 42 reproduces the main run exactly (F46).
- **Code-review preparation:** each modelling notebook ends with a cheat sheet listing what each piece does and what breaks if it is removed.

---

## 9. LIMITATIONS AND RESPONSIBLE AI
- **Synthetic data, small differences.** Nothing here generalises to real customers or to "deep learning versus trees" in general.
- **Tuning and optimism.** Neither model was tuned beyond a fixed configuration, so tuning could shift the gap; both early-stop on the fold they are scored on.
- **Scope of the comparison.** Results are conditional on the team's shared 47-feature set, held fixed so every model sees identical inputs; the set is a starting point, not a ceiling.
- **Fairness not yet measured.** Performance by country, tier and age band (including the 5.1% with unknown age) is still to be evaluated. Tier is a strong, spend-based predictor, so the model may favour high-loyalty customers and overlook customers who could be won back.
- **Likelihood is not persuadability.** A high score marks customers likely to buy, not customers an offer would change (the CRM team's "would have bought anyway" problem). The retention-offer history could support an uplift analysis, but it was not randomised.
- **Consent and policy sit outside the model.** 25.7% of customers list a preferred channel they have not opted in to (F14). Consent, open-complaint and tier checks must be in the agent layer, with human approval before anything is sent.
- **Snapshot validity.** Calibration was measured on one prediction window with random folds and may drift in the November to December peak season.

## 10. NEXT STEPS
1. Compute SHAP or permutation importance for the FT-Transformer and report performance by country, tier and age band (fairness).
2. **Feature experiment** applied to both models on the same folds: because the models are so close on identical inputs, better features are likely a bigger lever than architecture. The EDA flagged candidates not yet in the shared set: region (F23) and discount_usage_rate (AUC 0.581). Prune redundant inputs (credit_limit, which was noise in the EDA, and the duplicated age and email missing flags).
3. Package the fold models and preprocessors as an agent tool that returns a probability plus reasons, wrapped in consent and policy checks, with human approval.

## 11. DECLARATION OF AI USE
The student used Claude (Anthropic) to recommend the model and evaluation design, generate Colab notebooks (FT-Transformer, LightGBM baseline, seed studies), diagnose baseline defects from the student's notebook outputs, and draft reports. The student ran every notebook and checked results against the outputs: the model shape and parameter-count check, the leakage checks, the late-delivery assertion, the identical-features and identical-folds checks, the seed-42 reproducibility check and the learning-rate comparison. Full entries are in the student's AI Use Log. [Student to confirm this matches the AI Use Log.]

---

## 12. RUBRIC EVIDENCE MAP (Progress Review 1, individual criteria)
The rubric bands are F (lowest), D, C, B, A (highest). The target is the **A band** in every row.

| Criterion (marks) | A-band descriptor | Evidence in this report | What the slide and script must show |
|---|---|---|---|
| Input Data: Collection (5 marks shared with EDA/Preparation) | "Data collected is relevant, substantial and coherent; process is sophisticated" | F03, F04, F05, F06, F26, F28; section 3.1 and 3.2 | Scope (17-table pack, no external data, 7 tables integrated into 47 features, 4 files cleaned); coherent one-to-one keys; a visible multi-step reproducible process (audit, clean, validate, save outputs) |
| Input Data: EDA/Preparation | "Excellent EDA with insightful details, use of visualisation resulting in comprehensive data preparation" | F07 to F25; section 3.3 and 3.4; figures A1 to A7 | 2 to 3 EDA visuals with the "so what"; a compact cleaning summary (14 logged issues); insight-to-decision links; preparation continued inside the model pipeline (per-fold, leakage-safe) |
| Supervised Model: Training (5 marks shared with Evaluation) | "Completed with custom training; no conceptual errors and evidence of optimisations" | F33 to F37, F29 to F32; section 4 | Architecture diagram; custom PyTorch training; leakage-safe design (no conceptual errors); explicit optimisation evidence (lr diagnosis with numbers, early stopping, gradient clipping, per-fold preprocessing, seed study); fair baseline corrected |
| Supervised Model: Evaluation | "Appropriate use and in-depth interpretation of performance metrics with no misconceptions" | F38 to F44; section 7 | Metrics suited to imbalance (ROC-AUC, PR-AUC, precision at k, calibration); results table; interpretation of the small gap and its caveats; the accuracy trap; limits |
| Source Codes: Structure (5 marks shared with Code Review) | "Well-organised, logical, modular with excellent use of comments or markdown" | F45, F46; section 8 | A mini map of the three notebooks and their sections; CONFIG, functions, assertions, logs, shared artefacts |
| Source Codes: Code Review | "Able to explain all critical components of the codes conveying excellent subject knowledge" | assessed live and orally | Not on the slide. The script may signal readiness by naming the critical components |
| Group: Slides (group mark, one slide contributes) | "Content is relevant, well organised, coherent and extensive" | whole report | Clear layout with visual hierarchy, coherent story left to right, extensive but readable content |

---

## 13. FIGURE ASSETS AVAILABLE (file names in the asset pack)
These are real figures from the student's notebooks. The slide image the AI produces should show **placeholders** where these go; the student will insert the real ones in Canva.

| File | What it shows | Best use |
|---|---|---|
| A1_signups_per_month_raw.png | Raw signups and preference-update dates per month: pile-ups on the first and last day | Data-quality example |
| A2_target_distribution.png | Target distribution: 54,674 (85.4%) vs 9,326 (14.6%) | Class imbalance, why accuracy misleads |
| A3_numeric_feature_distributions.png | Distribution of each numeric feature for buyers and non-buyers | EDA (loyalty separates strongly; credit_limit does not) |
| A4_spearman_correlation_heatmap.png | Correlation of numeric features with the target; the 0.94 redundancy | EDA / redundancy |
| A5_purchase_rate_by_category.png | Purchase rate by tier, country, opt-in, channel, missing-value flags | EDA (tier 5.4% to 52.7%; missingness informative) |
| A6_loyalty_signup_year_region.png | Purchase rate by loyalty decile, signup year and region | EDA (steep loyalty effect; region signal) |
| A7_customer_preferences.png | Preferred channel vs opt-in; language by country; consent counts | Consent conflict (25.7%) |
| B1_ft_transformer_architecture.png | Six-box architecture diagram | Model design |
| B2_ft_calibration_and_learning_curves.png | Calibration curve and fold-0 learning curves | Evaluation (calibrated, no overfitting) |
| B3_ft_roc_and_pr_curves.png | Out-of-fold ROC and precision-recall curves (seed 42) | Evaluation |
| B4_seed_noise_comparison.png | Per-seed results for both models (zoomed axes) | Is the gap real? |
| B5_lightgbm_eval_plots.png | LightGBM ROC, PR, calibration and fold-0 AUC per tree | Baseline evidence |
| B6_lightgbm_shap_and_gain_importance.png | Top-15 features by SHAP and by split gain | What drives predictions |

Suggested slide priorities (most valuable first): A2, A5 or A6, B1, B2 or B4, then A1 or A7.

---

## 14. CLAIMS TO AVOID (these would be wrong or unsupported)
- Do **not** say the FT-Transformer "beats" or "significantly outperforms" LightGBM. Say it is "slightly but reliably ahead" and "practically comparable".
- Do **not** present accuracy as the success metric, and do not call a model "88% accurate" without saying that predicting nobody scores 85.43%.
- Do **not** say the data is real. It is synthetic.
- Do **not** claim hyperparameter tuning beyond the learning-rate diagnosis and fixed configurations ("minimal tuning").
- Do **not** claim fairness has been measured. It is a next step.
- Do **not** claim region, discount_usage_rate or the other EDA candidates are in the model. They are planned for a later feature experiment.
- Do **not** claim the student cleaned the sessions, transactions, deliveries, support or retention tables. Other team members did.
- Do **not** mention the other team members' models.
- Do **not** claim SHAP has been computed for the FT-Transformer; the SHAP results are for the LightGBM baseline.
- Do **not** invent numbers, rounding, or results that are not in section 2.

## 15. PLAIN-LANGUAGE GLOSSARY (for the script)
- **Out-of-fold (OOF) prediction:** a prediction for a customer made by a model that never saw that customer in training.
- **Data leakage:** information from outside the training data (for example the future or the validation set) sneaking into training, which inflates scores.
- **ROC-AUC / PR-AUC:** ranking scores; PR-AUC focuses on how well the rarer class (buyers) is found.
- **Calibration:** whether "30% likely" really means about 30 in 100 buy.
- **Brier score:** the average squared error of predicted probabilities; lower is better.
- **Token / attention:** a token is a learned vector for one feature; attention lets each token weigh every other token.
- **Early stopping:** stop training when the validation score stops improving and keep the best version.
- **SHAP:** a method that shows how much each feature pushed a prediction up or down.
- **Seed:** the starting point of the random number generator; changing it shows how much results depend on luck.
