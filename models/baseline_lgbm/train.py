"""
ShopSphere LightGBM Pipeline - End-to-End from SQLite Database (shopsphere.db)
==============================================================================
1. Queries clean relational tables directly from shopsphere.db.
2. Extracts aggregated Customer 360 features using SQL.
3. Performs 5-Fold Stratified Cross-Validation with early stopping.
4. Evaluates ROC-AUC, PR-AUC, and F1-score.
5. Generates task_outputs/submission_lightgbm.csv and feature importance analysis.
"""

import os
import time
import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score
import lightgbm as lgb

DB_PATH = "shopsphere.db"
OUTPUT_DIR = "task_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("TRAINING LIGHTGBM DIRECTLY FROM RELATIONAL DATABASE (shopsphere.db)")
print("=" * 80)

# ==============================================================================
# 1. EXTRACT FEATURES DIRECTLY FROM SHOPSPHERE.DB VIA SQL
# ==============================================================================
print("\n[Step 1/5] Extracting RFM, Session, Delivery, and Ticket features via SQL...")
t0 = time.time()

conn = sqlite3.connect(DB_PATH)

feature_query = """
WITH cust_rfm AS (
    SELECT 
        customer_id,
        COUNT(transaction_id) AS total_orders,
        COALESCE(SUM(order_value), 0.0) AS total_spend_usd,
        COALESCE(AVG(order_value), 0.0) AS avg_order_value,
        COALESCE(MAX(order_value), 0.0) AS max_order_value,
        COALESCE(SUM(items_count), 0) AS total_items_bought,
        COALESCE(SUM(discount_applied), 0) AS discounted_orders_count,
        COALESCE(SUM(high_value_flag), 0) AS high_value_orders_count,
        COALESCE(SUM(CASE WHEN transaction_timestamp >= '2024-09-01' THEN 1 ELSE 0 END), 0) AS orders_last_30d,
        COALESCE(SUM(CASE WHEN transaction_timestamp >= '2024-07-01' THEN 1 ELSE 0 END), 0) AS orders_last_90d,
        ROUND(JULIANDAY('2024-09-30') - JULIANDAY(MAX(transaction_timestamp)), 1) AS recency_days
    FROM transactions
    GROUP BY customer_id
),
cust_sessions AS (
    SELECT 
        customer_id,
        COUNT(session_id) AS total_sessions,
        COALESCE(SUM(pages_viewed), 0) AS total_pages_viewed,
        COALESCE(SUM(cart_additions), 0) AS total_cart_additions,
        COALESCE(SUM(bounce_flag), 0) AS total_bounced_sessions,
        COALESCE(AVG(session_duration_capped), 0.0) AS avg_session_duration_s,
        COALESCE(SUM(CASE WHEN session_timestamp >= '2024-09-01' THEN 1 ELSE 0 END), 0) AS sessions_last_30d,
        COALESCE(SUM(CASE WHEN session_timestamp >= '2024-07-01' THEN 1 ELSE 0 END), 0) AS sessions_last_90d,
        ROUND(JULIANDAY('2024-09-30') - JULIANDAY(MAX(session_timestamp)), 1) AS days_since_last_session
    FROM sessions
    GROUP BY customer_id
),
cust_deliveries AS (
    SELECT 
        t.customer_id,
        COUNT(d.transaction_id) AS total_deliveries,
        COALESCE(SUM(CASE WHEN d.delivery_status = 'Late' THEN 1 ELSE 0 END), 0) AS late_deliveries_count,
        COALESCE(SUM(CASE WHEN d.delivery_status = 'Lost' THEN 1 ELSE 0 END), 0) AS lost_packages_count,
        COALESCE(AVG(d.delay_days), 0.0) AS avg_delay_days
    FROM transactions t
    JOIN deliveries d ON t.transaction_id = d.transaction_id
    GROUP BY t.customer_id
),
cust_tickets AS (
    SELECT 
        customer_id,
        COUNT(ticket_id) AS total_support_tickets,
        COALESCE(SUM(CASE WHEN status = 'Open' THEN 1 ELSE 0 END), 0) AS open_tickets_count,
        COALESCE(SUM(CASE WHEN status = 'Open' AND priority IN ('High', 'Urgent') THEN 1 ELSE 0 END), 0) AS high_priority_open_tickets,
        AVG(csat_score) AS avg_csat_score
    FROM support_tickets
    GROUP BY customer_id
),
cust_retention AS (
    SELECT 
        customer_id,
        COUNT(contact_id) AS retention_contacts_count,
        COALESCE(SUM(redeemed), 0) AS retention_redeemed_count,
        COALESCE(SUM(purchased_within_14d), 0) AS retention_purchased_count
    FROM retention_contact_history
    GROUP BY customer_id
)
SELECT 
    c.customer_id,
    c.age,
    c.age_is_missing,
    c.country_code,
    c.customer_tier,
    c.credit_limit,
    c.loyalty_score,
    c.email_open_rate,
    c.email_open_rate_is_missing,
    c.avg_review_score,
    ROUND(JULIANDAY('2024-09-30') - JULIANDAY(c.signup_date), 1) AS account_age_days,
    
    -- Preferences
    COALESCE(p.email_opt_in, 0) AS email_opt_in,
    COALESCE(p.sms_opt_in, 0) AS sms_opt_in,
    COALESCE(p.do_not_contact, 0) AS do_not_contact,
    COALESCE(p.preferred_channel, 'unknown') AS preferred_channel,
    
    -- RFM
    COALESCE(rfm.total_orders, 0) AS total_orders,
    COALESCE(rfm.total_spend_usd, 0.0) AS total_spend_usd,
    COALESCE(rfm.avg_order_value, 0.0) AS avg_order_value,
    COALESCE(rfm.max_order_value, 0.0) AS max_order_value,
    COALESCE(rfm.total_items_bought, 0) AS total_items_bought,
    COALESCE(rfm.discounted_orders_count, 0) AS discounted_orders_count,
    COALESCE(rfm.high_value_orders_count, 0) AS high_value_orders_count,
    COALESCE(rfm.orders_last_30d, 0) AS orders_last_30d,
    COALESCE(rfm.orders_last_90d, 0) AS orders_last_90d,
    rfm.recency_days,
    
    -- Sessions
    COALESCE(s.total_sessions, 0) AS total_sessions,
    COALESCE(s.total_pages_viewed, 0) AS total_pages_viewed,
    COALESCE(s.total_cart_additions, 0) AS total_cart_additions,
    COALESCE(s.total_bounced_sessions, 0) AS total_bounced_sessions,
    COALESCE(s.avg_session_duration_s, 0.0) AS avg_session_duration_s,
    COALESCE(s.sessions_last_30d, 0) AS sessions_last_30d,
    COALESCE(s.sessions_last_90d, 0) AS sessions_last_90d,
    s.days_since_last_session,
    
    -- Deliveries
    COALESCE(d.total_deliveries, 0) AS total_deliveries,
    COALESCE(d.late_deliveries_count, 0) AS late_deliveries_count,
    COALESCE(d.lost_packages_count, 0) AS lost_packages_count,
    COALESCE(d.avg_delay_days, 0.0) AS avg_delay_days,
    
    -- Tickets
    COALESCE(t.total_support_tickets, 0) AS total_support_tickets,
    COALESCE(t.open_tickets_count, 0) AS open_tickets_count,
    COALESCE(t.high_priority_open_tickets, 0) AS high_priority_open_tickets,
    t.avg_csat_score,
    
    -- Retention
    COALESCE(r.retention_contacts_count, 0) AS retention_contacts_count,
    COALESCE(r.retention_redeemed_count, 0) AS retention_redeemed_count,
    COALESCE(r.retention_purchased_count, 0) AS retention_purchased_count,
    
    -- Split flags & Target
    tr.target,
    CASE WHEN te.customer_id IS NOT NULL THEN 1 ELSE 0 END AS is_test

FROM customers c
LEFT JOIN customer_preferences p ON c.customer_id = p.customer_id
LEFT JOIN cust_rfm rfm ON c.customer_id = rfm.customer_id
LEFT JOIN cust_sessions s ON c.customer_id = s.customer_id
LEFT JOIN cust_deliveries d ON c.customer_id = d.customer_id
LEFT JOIN cust_tickets t ON c.customer_id = t.customer_id
LEFT JOIN cust_retention r ON c.customer_id = r.customer_id
LEFT JOIN train tr ON c.customer_id = tr.customer_id
LEFT JOIN test te ON c.customer_id = te.customer_id;
"""

df = pd.read_sql_query(feature_query, conn)
conn.close()

print(f"Extracted {len(df):,} customer records in {time.time() - t0:.2f} seconds.")

# ==============================================================================
# 2. FEATURE ENGINEERING (RATIOS & INTERACTION SIGNALS)
# ==============================================================================
print("\n[Step 2/5] Engineering derived behavioral ratios...")
df["discount_order_ratio"] = np.where(df["total_orders"] > 0, df["discounted_orders_count"] / df["total_orders"], 0.0)
df["bounce_rate"] = np.where(df["total_sessions"] > 0, df["total_bounced_sessions"] / df["total_sessions"], 0.0)
df["cart_abandonment_ratio"] = np.where(
    df["total_cart_additions"] > 0,
    np.clip(1.0 - (df["total_orders"] / df["total_cart_additions"]), 0.0, 1.0),
    0.0
)
df["is_inactive_90d"] = ((df["orders_last_90d"] == 0) & (df["sessions_last_90d"] == 0)).astype(int)

# Target categorical encoding
CATEGORICAL_COLS = ["country_code", "customer_tier", "preferred_channel"]
for col in CATEGORICAL_COLS:
    df[col] = df[col].astype("category")

EXCLUDE_COLS = ["customer_id", "target", "is_test"]
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

# Train vs Test splits
train_df = df[df["is_test"] == 0].copy()
test_df = df[df["is_test"] == 1].copy()

X = train_df[FEATURE_COLS]
y = train_df["target"].astype(int)
X_test = test_df[FEATURE_COLS]

print(f"Training features count: {len(FEATURE_COLS)}")
print(f"Train samples: {len(X):,} (Positives: {y.sum():,} | {y.mean():.2%})")
print(f"Test samples:  {len(X_test):,}")

# ==============================================================================
# 3. CONFIGURE LIGHTGBM HYPERPARAMETERS
# ==============================================================================
scale_pos_weight = (len(y) - y.sum()) / y.sum()

params = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

# ==============================================================================
# 4. 5-FOLD STRATIFIED CROSS-VALIDATION
# ==============================================================================
print("\n[Step 3/5] Running 5-Fold Stratified Cross Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))
feature_importances = np.zeros(len(FEATURE_COLS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CATEGORICAL_COLS)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, categorical_feature=CATEGORICAL_COLS)
    
    model = lgb.train(
        params,
        trn_data,
        num_boost_round=1200,
        valid_sets=[trn_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    feature_importances += model.feature_importance(importance_type="gain") / skf.n_splits
    
    fold_auc = roc_auc_score(y_va, val_pred)
    fold_pr_auc = average_precision_score(y_va, val_pred)
    print(f"  Fold {fold} | Best Iteration: {model.best_iteration:4d} | ROC-AUC: {fold_auc:.5f} | PR-AUC: {fold_pr_auc:.5f}")

# ==============================================================================
# 5. EVALUATION, OPTIMAL THRESHOLDING & SUBMISSION
# ==============================================================================
print("\n[Step 4/5] Model Evaluation & Threshold Tuning...")
overall_auc = roc_auc_score(y, oof_preds)
overall_pr_auc = average_precision_score(y, oof_preds)

print(f"\n=======================================================")
print(f"  Overall OOF ROC-AUC : {overall_auc:.5f}")
print(f"  Overall OOF PR-AUC  : {overall_pr_auc:.5f}")
print(f"=======================================================")

# Search for best F1 threshold
best_thresh = 0.5
best_f1 = 0.0
for thresh in np.arange(0.30, 0.70, 0.02):
    f1 = f1_score(y, (oof_preds >= thresh).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Optimal Decision Threshold (Max F1): {best_thresh:.2f} (F1 = {best_f1:.4f})")
print("\nClassification Report at Optimal Threshold:")
print(classification_report(y, (oof_preds >= best_thresh).astype(int), digits=4))

# Feature Importance
fi_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "gain": feature_importances
}).sort_values(by="gain", ascending=False)

print("\nTop 12 Most Informative Features (by Gain):")
print(fi_df.head(12).to_string(index=False))

# Save submission
print("\n[Step 5/5] Generating Kaggle/Holdout Submission File...")
sub_df = pd.DataFrame({
    "customer_id": test_df["customer_id"].astype(int),
    "target": (test_preds >= best_thresh).astype(int)
})

submission_path = os.path.join(OUTPUT_DIR, "submission_lightgbm.csv")
sub_df.to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path}")
print(f"Predicted positive conversions: {sub_df['target'].sum():,} / {len(sub_df):,} ({sub_df['target'].mean():.2%})")

# Also save OOF predictions and feature importances for ensemble/inspection
fi_path = os.path.join(OUTPUT_DIR, "lightgbm_feature_importance.csv")
fi_df.to_csv(fi_path, index=False)
print(f"Feature importance saved to: {fi_path}")

print("\nTRAINING & INFERENCE COMPLETED SUCCESSFULLY!")
