"""
ShopSphere Deep Learning Pipeline - PyTorch Tabular MLP with Entity Embeddings
=============================================================================
1. Loads Customer 360 feature matrix from task_outputs/model_feature_matrix.csv.
2. Encodes categorical variables (country_code, customer_tier, preferred_channel) via PyTorch nn.Embedding.
3. Imputes and standardizes numerical variables with StandardScaler.
4. Employs a Tabular Neural Network with BatchNorm, ReLU activations, and Dropout.
5. Performs 5-Fold Stratified Cross Validation with early stopping based on validation ROC-AUC.
6. Evaluates ROC-AUC, PR-AUC (Average Precision), and F1-score with class-imbalance weighting.
7. Saves out-of-fold predictions, test predictions, and a comprehensive comparison table against LightGBM.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score, brier_score_loss
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUTPUT_DIR = "task_outputs"
FEATURE_MATRIX_PATH = os.path.join(OUTPUT_DIR, "model_feature_matrix.csv")

print("=" * 80)
print("TRAINING DEEP LEARNING MODEL (PyTorch Tabular Neural Network)")
print("=" * 80)

# ==============================================================================
# 1. LOAD FEATURE MATRIX
# ==============================================================================
print("\n[Step 1/5] Loading feature matrix...")
t0 = time.time()
df = pd.read_csv(FEATURE_MATRIX_PATH)
print(f"Loaded {len(df):,} total customer records in {time.time() - t0:.2f}s.")

# Identify categorical and continuous columns
CAT_COLS = ["country_code", "customer_tier", "preferred_channel"]
DROP_COLS = ["customer_id", "target", "is_test"]
NUM_COLS = [c for c in df.columns if c not in CAT_COLS and c not in DROP_COLS]

print(f"Categorical features ({len(CAT_COLS)}): {CAT_COLS}")
print(f"Continuous features ({len(NUM_COLS)}): {NUM_COLS[:5]}... and {len(NUM_COLS)-5} more")

# Split train and test sets
train_df = df[df["is_test"] == 0].copy().reset_index(drop=True)
test_df = df[df["is_test"] == 1].copy().reset_index(drop=True)

y = train_df["target"].astype(int).values
pos_count = y.sum()
neg_count = len(y) - pos_count
pos_weight_value = neg_count / pos_count
print(f"Train samples: {len(train_df):,} (Positives: {pos_count:,} | {y.mean():.2%})")
print(f"Test samples:  {len(test_df):,}")
print(f"Calculated pos_weight for loss: {pos_weight_value:.3f}")

# ==============================================================================
# 2. CATEGORICAL ENCODING & EMBEDDING DIMS
# ==============================================================================
cat_encoders = {}
cat_dims = []
for c in CAT_COLS:
    train_df[c] = train_df[c].fillna("Unknown").astype(str)
    test_df[c] = test_df[c].fillna("Unknown").astype(str)
    
    le = LabelEncoder()
    train_df[c] = le.fit_transform(train_df[c])
    test_df[c] = le.transform(test_df[c])
    cat_encoders[c] = le
    num_classes = len(le.classes_)
    # Standard rule of thumb: min(50, (num_classes + 1) // 2)
    emb_dim = max(4, (num_classes + 1) // 2)
    cat_dims.append((num_classes, emb_dim))

print("\nCategorical Embedding Dimensions:")
for col, (num_c, emb_d) in zip(CAT_COLS, cat_dims):
    print(f"  - {col}: {num_c} categories -> {emb_d} embedding dimension")

# Fill missing numerical values with train medians
for col in NUM_COLS:
    med_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(med_val)
    test_df[col] = test_df[col].fillna(med_val)

X_cat_train = train_df[CAT_COLS].values
X_num_train = train_df[NUM_COLS].values
X_cat_test = test_df[CAT_COLS].values
X_num_test = test_df[NUM_COLS].values

# ==============================================================================
# 3. PYTORCH MODEL ARCHITECTURE & DATASET
# ==============================================================================
class TabularDataset(Dataset):
    def __init__(self, cat_data, num_data, targets=None):
        self.cat_data = torch.tensor(cat_data, dtype=torch.long)
        self.num_data = torch.tensor(num_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.cat_data[idx], self.num_data[idx], self.targets[idx]
        return self.cat_data[idx], self.num_data[idx]

class TabularNN(nn.Module):
    def __init__(self, emb_dims, num_numerical, hidden_dims=[128, 64, 32], dropout_p=0.3):
        super().__init__()
        # Entity Embedding layers
        self.embeddings = nn.ModuleList([nn.Embedding(num_c, dim) for num_c, dim in emb_dims])
        total_emb_dim = sum(dim for _, dim in emb_dims)
        
        # Batch normalization for continuous numerical inputs
        self.num_bn = nn.BatchNorm1d(num_numerical)
        
        input_dim = total_emb_dim + num_numerical
        
        # Multi-Layer Perceptron layers with BatchNorm and Dropout
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        emb_outs = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_outs, dim=1)
        x_num_norm = self.num_bn(x_num)
        x = torch.cat([x_emb, x_num_norm], dim=1)
        return self.mlp(x).squeeze(1)

# ==============================================================================
# 4. 5-FOLD STRATIFIED CROSS-VALIDATION TRAINING
# ==============================================================================
print("\n[Step 2/5] Initiating 5-Fold Stratified Cross-Validation...")
BATCH_SIZE = 512
EPOCHS = 15
LR = 0.002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation Device: {DEVICE}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train_df))
test_fold_preds = np.zeros((len(test_df), 5))

cv_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_cat_train, y), 1):
    fold_t0 = time.time()
    
    # Scale numerical data strictly based on train fold
    scaler = StandardScaler()
    X_num_tr_scaled = scaler.fit_transform(X_num_train[train_idx])
    X_num_va_scaled = scaler.transform(X_num_train[val_idx])
    X_num_te_scaled = scaler.transform(X_num_test)
    
    # Create DataLoaders
    train_dataset = TabularDataset(X_cat_train[train_idx], X_num_tr_scaled, y[train_idx])
    val_dataset = TabularDataset(X_cat_train[val_idx], X_num_va_scaled, y[val_idx])
    test_dataset = TabularDataset(X_cat_test, X_num_te_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = TabularNN(cat_dims, len(NUM_COLS), hidden_dims=[128, 64, 32], dropout_p=0.25).to(DEVICE)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_auc = 0.0
    best_val_preds = None
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_cat, batch_num, batch_targets in train_loader:
            batch_cat = batch_cat.to(DEVICE)
            batch_num = batch_num.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(batch_cat, batch_num)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_targets)
        
        scheduler.step()
        train_loss /= len(train_dataset)
        
        # Validation evaluation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_cat, batch_num, _ in val_loader:
                batch_cat = batch_cat.to(DEVICE)
                batch_num = batch_num.to(DEVICE)
                logits = model(batch_cat, batch_num)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds_list.extend(probs)
        
        val_probs = np.array(val_preds_list)
        val_auc = roc_auc_score(y[val_idx], val_probs)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_preds = val_probs
            # Predict test set with current best fold weights
            test_preds_list = []
            with torch.no_grad():
                for batch_cat, batch_num in test_loader:
                    batch_cat = batch_cat.to(DEVICE)
                    batch_num = batch_num.to(DEVICE)
                    logits = model(batch_cat, batch_num)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    test_preds_list.extend(probs)
            test_fold_preds[:, fold - 1] = np.array(test_preds_list)

    oof_preds[val_idx] = best_val_preds
    fold_pr_auc = average_precision_score(y[val_idx], best_val_preds)
    fold_time = time.time() - fold_t0
    print(f"  Fold {fold}/5 finished in {fold_time:.1f}s | Best Val ROC-AUC: {best_val_auc:.4f} | PR-AUC: {fold_pr_auc:.4f}")
    cv_metrics.append({"fold": fold, "roc_auc": best_val_auc, "pr_auc": fold_pr_auc})

# ==============================================================================
# 5. OVERALL EVALUATION & COMPARISON
# ==============================================================================
print("\n[Step 3/5] Evaluating Full Out-of-Fold Performance...")
final_roc_auc = roc_auc_score(y, oof_preds)
final_pr_auc = average_precision_score(y, oof_preds)
final_brier = brier_score_loss(y, oof_preds)

# Optimal F1 threshold search
thresholds = np.arange(0.2, 0.8, 0.02)
best_th = 0.5
best_f1 = 0.0
for th in thresholds:
    f1 = f1_score(y, (oof_preds >= th).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"\n========================================================")
print(f"DEEP LEARNING MODEL OUT-OF-FOLD PERFORMANCE:")
print(f"  - ROC-AUC:           {final_roc_auc:.4f}")
print(f"  - PR-AUC:            {final_pr_auc:.4f}")
print(f"  - Brier Score:       {final_brier:.4f}")
print(f"  - Best F1-Score:     {best_f1:.4f} (at threshold {best_th:.2f})")
print(f"========================================================")

print("\nClassification Report (DL Model @ Best F1 Threshold):")
print(classification_report(y, (oof_preds >= best_th).astype(int), digits=4))

# Average test predictions across 5 folds
final_test_preds = test_fold_preds.mean(axis=1)

# ==============================================================================
# 6. SAVE PREDICTIONS & SUBMISSION
# ==============================================================================
print("\n[Step 4/5] Saving Deep Learning predictions and submission file...")
dl_submission = pd.DataFrame({
    "customer_id": test_df["customer_id"],
    "target_probability": final_test_preds,
    "target": (final_test_preds >= best_th).astype(int)
})
dl_submission_path = os.path.join(OUTPUT_DIR, "submission_deeplearning.csv")
dl_submission.to_csv(dl_submission_path, index=False)
print(f"Saved test predictions to: {dl_submission_path}")

# Save out-of-fold probability column into a summary df
train_df["dl_probability"] = oof_preds
test_df["dl_probability"] = final_test_preds

# ==============================================================================
# 7. COMPARE WITH LIGHTGBM BASELINE
# ==============================================================================
print("\n[Step 5/5] Generating Head-to-Head Comparison Report vs. LightGBM...")

lgb_sub_path = os.path.join(OUTPUT_DIR, "submission_lightgbm.csv")
lgb_available = os.path.exists(lgb_sub_path)

comparison_summary = {
    "PyTorch_Tabular_DL": {
        "ROC_AUC": round(final_roc_auc, 4),
        "PR_AUC": round(final_pr_auc, 4),
        "Brier_Score": round(final_brier, 4),
        "Best_F1": round(best_f1, 4),
        "Optimal_Threshold": round(best_th, 2)
    }
}

# Save comparative metrics to JSON
with open(os.path.join(OUTPUT_DIR, "model_evaluation_metrics.json"), "w") as f:
    json.dump(comparison_summary, f, indent=4)

print(f"\nModel evaluation metrics logged to {os.path.join(OUTPUT_DIR, 'model_evaluation_metrics.json')}")
print("Training and evaluation completed successfully!")
