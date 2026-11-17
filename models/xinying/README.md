# Xin Ying's Deep Learning Model: PyTorch TabularNN with Entity Embeddings

## 1. Overview
This model is a custom **PyTorch Tabular Neural Network** designed to predict 30-day repeat purchases (October 2024 retention) for ShopSphere Global. It incorporates learnable entity embeddings for categorical fields, batch normalization for continuous numerical signals, and dropout regularization.

## 2. Validation & Training Setup
- **Validation Strategy**: **80/20 Stratified Train/Test Holdout Split** (51,200 training samples, 12,800 test holdout samples).
- **Random Seed**: Fixed **42** across data splitting, PyTorch tensor initialization, and optimizer.
- **Data Preprocessing**: `StandardScaler` fitted strictly on the 80% training set to guarantee zero data leakage into the 20% test holdout.

## 3. Training Notebook
The complete training and evaluation pipeline is contained in:
- `models/xinying/notebooks/03_deep_learning_pipeline.ipynb`

## 4. Evaluation & Results (80/20 Holdout Test Set)
- **ROC-AUC**: **0.8740**
- **PR-AUC**: **0.5728**
- **F1-Score**: **0.5451** (optimal threshold = 0.70)
- **Recall**: **60.97%**
- **Accuracy**: **85.41%**

Compared against the baseline notebook:
- `models/baseline_lgbm/notebooks/02_lightgbm_training_pipeline.ipynb` (80/20 split, seed=42)
