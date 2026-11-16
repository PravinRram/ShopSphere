# Xin Ying's Deep Learning Model: PyTorch TabularNN with Entity Embeddings

## 1. Overview
This model is a custom **PyTorch Tabular Neural Network** designed to predict 30-day repeat purchases (October 2024 retention) for ShopSphere Global. It incorporates learnable entity embeddings for categorical fields, batch normalization for continuous numerical signals, and dropout regularization.

## 2. Architecture Details
- **Categoricals**: `country_code` (emb dim 4), `customer_tier` (emb dim 4), `preferred_channel` (emb dim 4).
- **Numerical Features**: 41 continuous features extracted from RFM metrics, session engagement, delivery friction, and customer service CSAT.
- **Layers**: Multi-Layer Perceptron (128 -> 64 -> 32) with `BatchNorm1d`, `ReLU`, and `Dropout(p=0.25)`.
- **Loss Function**: `BCEWithLogitsLoss(pos_weight=5.863)` to account for the ~14.6% positive class imbalance.
- **Optimizer**: AdamW (`lr=0.002`, `weight_decay=1e-4`) with `CosineAnnealingLR`.

## 3. How to Train
Ensure `shopsphere.db` or `task_outputs/model_feature_matrix.csv` is present, then run:
```bash
python models/xinying/train.py
```

## 4. Evaluation & Results (5-Fold Stratified CV)
- **Out-of-Fold ROC-AUC**: **0.8728**
- **Out-of-Fold PR-AUC**: **0.5702**
- **Overall Accuracy**: **85.33%**
- **Best F1-Score**: **0.5427** (at threshold 0.70)

Model predictions are saved in `models/xinying/results/test_dl_tabularnn.csv`.
Detailed metrics are logged in `models/xinying/results/metrics.json`.
