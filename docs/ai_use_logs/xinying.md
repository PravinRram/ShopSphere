# Annex B: AI Use Log

**Module / Course**: IT2226 – AI Innovation and Deep Learning Project  
**Project**: Win Back the Customer (ShopSphere Global)  
**Student Name**: Quek Xin Ying  
**Role / Domain**: Operations & Logistics Feature Engineering / Tabular Deep Learning  
**Repository**: `https://github.com/PravinRram/ShopSphere.git`  
**Review Checkpoint**: Progress Review 1 (Week 5 – 19 Nov 2026)

---

## 1. Overview and Logging Policy

In accordance with Section 7 and Annex B of the *IT2226 Project Guidebook*, this log documents all substantive interactions with AI assistants (Antigravity / Gemini Code Assistant). 

Rather than recording basic code autocomplete, this log captures:
1. Prompts that generated code blocks, architectural decisions, data processing pipelines, or diagnostic strategies.
2. The author's verification, critical analysis, and modifications before adoption.
3. Explicit rejections where AI suggestions failed domain logic, caused data leakage, or introduced architectural bugs.
4. Reconciliation with Git commit history per Section 7.5 of the Guidebook.

---

## 2. Chronological AI Interaction Log

| Date | Tool | What you asked for | Accepted / Changed / Rejected | Why — what you checked, found, and modified | Associated Commit(s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **27 Oct 2026** | Antigravity (Gemini) | Analyze `IT2226_ShopSphere_Brief_Student.docx` and `Project_Guidebook_v1.1` to extract stakeholder wish list requirements and grading rubrics. | **Accepted with modifications** | Reviewed the 10 stakeholder wishes. Verified mandatory constraints: W1 (30-day repeat purchase prediction), W3 (retention offer policy adherence), and W6 (multi-dimensional dashboard). Formulated team proposal covering W1, W2, W3, W4, W6, and W9. | — |
| **27 Oct 2026** | Antigravity (Gemini) | Determine whether classical machine learning or deep learning is required, and suggest how team members should divide baseline and DL models. | **Changed** | The assistant clarified that the project brief explicitly mandates a **supervised deep learning model** evaluated fairly against **at least one non-deep-learning baseline**. Agreed on team strategy: build a shared LightGBM baseline, while each member explores and tunes distinct model branches for the PR1 defense. | — |
| **29 Oct 2026** | Antigravity (Gemini) | Formulate a systematic data cleaning and auditing strategy across all 18 ShopSphere CSV tables. | **Accepted** | Inspected schemas across Part 1 (`sessions.csv`) and Part 2 (17 relational tables). Confirmed standardized handling of timestamp ISO conversions, currency strings, negative durations, and missing value indicators. | — |
| **29 Oct 2026** | Antigravity (Gemini) | Suggest whether to merge all 18 CSV tables into a single giant denormalized CSV file. | **Rejected** | Assistant initially suggested denormalization into a flat CSV. I rejected this because joining granular transaction logs (millions of rows) directly with customer profiles creates combinatorial record explosion, heavy memory overhead, and severe target leakage. Enforced a relational SQLite database (`shopsphere.db`) with customer UID aggregates instead. | — |
| **29 Oct 2026** | Antigravity (Gemini) | Generate automated data cleaning script for `deliveries`, `marketing_campaigns`, `offer_catalogue`, `retention_budget_2024Q4`, `retention_contact_history`, and `support_tickets`. | **Changed** | Ran generated scripts on sample records. Discovered that ticket CSAT scores had out-of-range anomalies and delivery dates contained future timestamps past the cutoff. Enforced explicit value clipping, temporal validity checks (`actual_delivery_date >= dispatch_date`), and missingness indicator flags. | — |
| **03 Nov 2026** | Antigravity (Gemini) | Suggest relational schema with Foreign Keys and populate `shopsphere.db` from cleaned tables. | **Accepted** | Verified SQLite table creation with indexed primary and foreign keys (`customer_id`, `transaction_id`, `session_id`, `ticket_id`). Verified referential integrity across relational tables using SQL count validations. | — |
| **03 Nov 2026** | Antigravity (Gemini) | Build a Customer 360 Feature Matrix SQL query combining RFM metrics, session engagement, delivery friction, and customer support tickets. | **Changed** | Audited SQL joins and found that events after 30 Sep 2024 were included in draft queries. Enforced strict temporal filtering (`timestamp < '2024-10-01'`) across transactions, deliveries, and browsing sessions to prevent future target leakage. | — |
| **03 Nov 2026** | Antigravity (Gemini) | Train a 5-Fold Stratified Cross-Validation LightGBM baseline directly from SQLite database. | **Changed** | Model defaulted to symmetric log-loss on an imbalanced dataset (~14.6% positive class). Added `scale_pos_weight = 5.86` to the objective and supplemented ROC-AUC with PR-AUC (Average Precision) and recall for business evaluation. | — |
| **03 Nov 2026** | Antigravity (Gemini) | Diagnose Colab runtime error: `KeyError` / `NameError` when querying customer tier and country code in batch evaluation. | **Accepted** | Traced bug to column naming mismatch between SQL query alias (`customer_tier`) and downstream script reference (`tier`). Standardized column aliases across all notebooks. | — |
| **05 Nov 2026** | Antigravity (Gemini) | Propose deep learning architecture for tabular customer data and assess computational requirements. | **Accepted** | Verified that a tabular Multi-Layer Perceptron (MLP) with learnable Entity Embeddings (`nn.Embedding`) is mathematically sound for mixed categorical and numerical data. Parameter footprint (~35k params) trains fast and cleanly on CPU. | — |
| **05 Nov 2026** | Antigravity (Gemini) | Generate PyTorch tabular neural network training script with 5-Fold Cross-Validation (`train_deeplearning_from_db.py`). | **Changed & Fixed** | Script produced `RuntimeError: Can't call numpy() on Tensor that requires grad`. Isolated bug to test inference inside training loop. Wrapped in `with torch.no_grad():` and `.detach()`. Ensured `StandardScaler` was fitted strictly on training folds to prevent validation leakage. | — |
| **05 Nov 2026** | Antigravity (Gemini) | Evaluate PyTorch Deep Learning model results and compare against LightGBM baseline. | **Accepted** | Verified metrics across folds: ROC-AUC: **0.8728**, PR-AUC: **0.5702**. Performed threshold sweeping to demonstrate that tuning probability cutoffs balances precision vs. recall depending on voucher margin cost. | — |
| **17 Nov 2026** | Antigravity (Gemini) | Reconcile model evaluation protocol across the team repository. Convert 5-Fold CV pipelines to standard 80/20 train/test holdout splits (`seed=42`). | **Accepted & Verified** | Teammate models (Egus E7 / TabNet) standardized on an 80/20 split (`train_test_split(..., test_size=0.2, random_state=42)`). Updated both `02_lightgbm_training_pipeline.ipynb` and `03_deep_learning_pipeline.ipynb`. Removed redundant `train.py` scripts to avoid confusion. Generated `models/xinying/results/metrics.json` (ROC-AUC: 0.8740, PR-AUC: 0.5728, Recall: 0.6097, F1: 0.5451). | `db82153`, `cf710e6`, `a86a345` |
| **18 Nov 2026** | Antigravity (Gemini) | Generate and arrange individual high-resolution Operations EDA charts for PR1 slide deck. | **Changed & Fixed** | Initial generated composite plot had overlapping text labels, cramped legends, and misaligned CSAT threshold lines. Separated into 6 standalone visual artifacts (`docs/slides/eda_individual_charts/`), adjusted subplot margins, increased DPI to 300, and moved the CSAT threshold label to top-center with expanded y-axis limits to eliminate visual clipping. | `fded5a0` |
| **22 Nov 2026** | Antigravity (Gemini) | Prepare presentation script for 5-minute PR1 walkthrough and audit Git repository against project rubric. | **Accepted** | Structured oral presentation timing (1 min Data Cleaning, 1.5 min Operations EDA, 1.5 min DL Model Architecture & Comparison, 1 min Agentic AI integration). Verified that no large database files or unneeded `.pt` weight binaries were pushed to Git. | `fded5a0` |

---

## 3. Reflection on AI Assistance

### What Worked Well:
- **Rapid Prototyping & Boilerplate Generation**: Using AI expedited writing repetitive SQL CTE aggregations across `transactions`, `deliveries`, and `support_tickets`, allowing more time for feature analysis.
- **PyTorch Modular Architecture**: Provided clean scaffolding for PyTorch custom `Dataset`, `DataLoader`, and multi-layer tabular embeddings.
- **Visualization Layouts**: Iteratively refined matplotlib chart aesthetics, annotations, and callouts for the presentation slides.

### Critical Verification & Independent Decisions:
- **Strict Cutoff Enforcement (Target Leakage Prevention)**: AI-generated queries initially omitted cutoff filters on sessions and deliveries. I audited all timestamps and strictly capped all historical data at `2024-09-30 23:59:59`.
- **Normalization Isolation**: AI code frequently fitted scalers on the entire dataset before splitting. I mandated that `StandardScaler.fit()` occurs strictly on the 80% train split and only transforms the 20% holdout test split.
- **Metric Selection for Business Utility**: AI defaults frequently emphasized overall accuracy. Because the repeat-purchase positive rate is ~14.6%, I enforced **PR-AUC**, **ROC-AUC**, and **Precision/Recall trade-off analysis** as the decisive metrics.
- **Git Hygiene**: Prevented committing unnecessary 500MB+ database files, virtual environments, or heavy PyTorch checkpoint binaries to the shared GitHub repository.
