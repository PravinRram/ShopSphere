# Annex B: AI Use Log

**Module / Course**: IT2226 – AI Innovation and Deep Learning Project  
**Project**: Win Back the Customer (ShopSphere Global)  
**Student Name**: [Your Name / Team Member]  
**Repository**: [GitHub Repository URL]  
**Review Checkpoint**: Progress Review 1 (Week 5 – 19 Nov 2026)

---

## 1. Overview and Logging Policy

In accordance with Section 7 and Annex B of the *IT2226 Project Guidebook*, this log documents all substantive interactions with AI assistants (Antigravity / Gemini Code Assistant). 

Rather than recording basic code autocomplete, this log captures:
1. Prompts that generated code blocks, architectural decisions, data processing pipelines, or diagnostic strategies.
2. The author's verification, critical analysis, and modifications before adoption.
3. Explicit rejections where AI suggestions failed domain logic, caused data leakage, or introduced architectural bugs.

---

## 2. Chronological AI Interaction Log

| Date | Tool | What you asked for | Accepted / Changed / Rejected | Why — what you checked, found, and modified |
| :--- | :--- | :--- | :--- | :--- |
| **27 Oct 2026** | Antigravity (Gemini) | Analyze `IT2226_ShopSphere_Brief_Student.docx` and `Project_Guidebook_v1.1` to extract stakeholder wish list requirements and grading rubrics. | **Accepted with modifications** | Reviewed the 10 stakeholder wishes. Verified mandatory constraints: W1 (30-day repeat purchase prediction), W3 (retention offer policy adherence), and W6 (multi-dimensional dashboard). Formulated team proposal covering W1, W2, W3, W4, W6, and W9. |
| **27 Oct 2026** | Antigravity (Gemini) | Determine whether classical machine learning or deep learning is required, and suggest how team members should divide baseline and DL models. | **Changed** | The assistant clarified that the project brief explicitly mandates a **supervised deep learning model** evaluated fairly against **at least one non-deep-learning baseline**. Agreed on team strategy: build a shared LightGBM baseline, while each member explores and tunes distinct model branches for the PR1 defense. |
| **29 Oct 2026** | Antigravity (Gemini) | Formulate a systematic data cleaning and auditing strategy across all 18 ShopSphere CSV tables. | **Accepted** | Inspected schemas across Part 1 (`sessions.csv`) and Part 2 (17 relational tables). Confirmed standardized handling of timestamp ISO conversions, currency strings, negative durations, and missing value indicators. |
| **29 Oct 2026** | Antigravity (Gemini) | Suggest whether to merge all 18 CSV tables into a single giant denormalized CSV file. | **Rejected** | Assistant initially discussed denormalization. I rejected a single flat CSV because combining granular transactions (millions of rows) with customer-level data introduces Cartesian product duplication, massive memory explosion, and high risk of target leakage. Decided instead to build a relational SQLite database (`shopsphere.db`) and aggregate features modularly per customer UID. |
| **29 Oct 2026** | Antigravity (Gemini) | Generate automated data cleaning script for `deliveries`, `marketing_campaigns`, `offer_catalogue`, `retention_budget_2024Q4`, `retention_contact_history`, and `support_tickets`. | **Changed** | Ran generated scripts on sample rows. Found that CSAT score in tickets had values outside the 1–5 range and delivery dates contained invalid future dates. Added explicit range clipping, date-consistency checks (`actual_delivery_date >= dispatch_date`), and missingness flags (`age_is_missing`, `email_open_rate_is_missing`). |
| **03 Nov 2026** | Antigravity (Gemini) | Suggest relational schema with Foreign Keys and populate `shopsphere.db` from cleaned tables. | **Accepted** | Verified SQLite table creation with indexed primary keys (`customer_id`, `transaction_id`, `session_id`, `ticket_id`). Validated referential integrity between tables using SQL count queries. |
| **03 Nov 2026** | Antigravity (Gemini) | Build a Customer 360 Feature Matrix SQL query combining RFM metrics, session engagement, delivery friction, and customer support tickets. | **Changed** | Checked the SQL aggregation logic. Found that transactions and sessions after the 30 September 2024 cutoff could leak into the target window (October 2024). Enforced a strict temporal barrier (`transaction_timestamp < '2024-10-01'` and `session_start_time < '2024-10-01'`) to ensure no future leakage occurred. |
| **03 Nov 2026** | Antigravity (Gemini) | Train a 5-Fold Stratified Cross-Validation LightGBM baseline directly from SQLite database. | **Changed** | Model code used default log-loss without addressing target imbalance (~14.6% positive class). Added `scale_pos_weight = 5.86` to the LightGBM objective and added PR-AUC (Average Precision) alongside ROC-AUC to prevent misleading evaluations. Baseline achieved 0.865+ ROC-AUC. |
| **03 Nov 2026** | Antigravity (Gemini) | Diagnose Colab runtime error: `NameError` / KeyError when querying customer tier and country code in batch evaluation. | **Accepted** | Traced the bug to a mismatched column naming convention between the SQL query alias (`customer_tier` vs `tier`). Standardized column aliases across the pipeline. |
| **05 Nov 2026** | Antigravity (Gemini) | Propose deep learning architecture for tabular customer data and assess computational requirements. | **Accepted** | Verified that a tabular Multi-Layer Perceptron (MLP) with learnable Entity Embeddings (`nn.Embedding`) is mathematically appropriate for mixed categorical and numerical data. Confirmed the parameter count (~35k params) runs efficiently on local CPU without requiring cloud GPU quota. |
| **05 Nov 2026** | Antigravity (Gemini) | Generate PyTorch tabular neural network training script with 5-Fold Cross-Validation (`train_deeplearning_from_db.py`). | **Changed & Fixed** | First run failed with: `RuntimeError: Can't call numpy() on Tensor that requires grad`. Isolated the issue to test-inference within the training epoch loop. Wrapped test evaluation in `with torch.no_grad():` and applied `.detach().numpy()`. Also ensured `StandardScaler` was fitted strictly inside each training fold to avoid data leakage into validation folds. |
| **05 Nov 2026** | Antigravity (Gemini) | Evaluate PyTorch Deep Learning model results and compare against LightGBM baseline. | **Accepted** | Verified 5-Fold CV metrics: Mean OOF ROC-AUC: **0.8728**, PR-AUC: **0.5702**, Accuracy: **85.33%**. Explored probability threshold tuning; demonstrated that a classification threshold of 0.70 optimizes F1-score (0.5427) by penalizing false positive voucher expenditures. |
| **05 Nov 2026** | Antigravity (Gemini) | Synthesize deliverables, code interrogation defense topics, and presentation structure for Week 5 Progress Review 1 (19 Nov). | **Accepted with modifications** | Mapped existing outputs directly against the rubric bands (Input Data: 5 marks, Supervised Model: 5 marks, Source Codes: 5 marks, Presentation: 5 marks). Structured slides and technical defense talking points around the code. |

---

## 3. Reflection on AI Assistance

### What Worked Well:
- **Rapid Prototyping & Boilerplate Generation**: Using AI allowed rapid drafting of complex SQL aggregation CTEs across 4 different relational tables (`transactions`, `sessions`, `deliveries`, `support_tickets`), cutting down manual join scripting.
- **Architectural Scaffolding**: Providing clear modular structure for PyTorch custom `Dataset`, `DataLoader`, and `TabularNN` class definitions.

### Critical Verification & Independent Decisions:
- **Preventing Target Leakage**: The AI did not initially restrict session timestamps to the September 30 cutoff date. I caught this during SQL review and enforced strict `< '2024-10-01'` filters.
- **Preventing Normalization Leakage**: In cross-validation, scaling all data beforehand is a common AI error. I required `StandardScaler.fit_transform()` to execute strictly on the training fold slice (`train_idx`) and `.transform()` on validation and test slices.
- **Domain Metric Selection**: AI outputs frequently defaulted to raw accuracy. Given that positive buyers constitute only 14.6% of the customer base, I enforced **ROC-AUC**, **PR-AUC**, and **F1 with threshold sweep** as the primary decision metrics.
