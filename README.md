# ShopSphere Global – AI Innovation & Deep Learning (IT2226)

This repository contains the end-to-end data processing, feature engineering, and machine learning pipelines for the **ShopSphere Global** customer retention project.

## Project Structure

- `01_data_cleaning_pipeline.ipynb`: Interactive Jupyter Notebook walking through data audit, standardization, and cleaning across raw datasets.
- `02_lightgbm_training_pipeline.ipynb`: LightGBM baseline modeling, stratified 5-fold cross-validation, and feature importance analysis.
- `train_lightgbm_from_db.py`: Standalone script to extract Customer 360 features from `shopsphere.db` via SQL and train the LightGBM baseline.
- `train_deeplearning_from_db.py`: Custom PyTorch Tabular Neural Network with learnable entity embeddings, batch normalization, dropout, and 5-fold cross-validation.
- `AI_USE_LOG.md`: Required Annex B AI Use Log documenting prompts, verifications, modifications, and rejections.
- `task_outputs/`: Model evaluation metrics, feature importances, and submission probability predictions.

## Machine Learning & Deep Learning Benchmarks

| Model | Architecture | ROC-AUC | PR-AUC | Key Strengths |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM (Baseline)** | Gradient Boosted Decision Trees | ~0.865 | ~0.550 | Fast non-linear tree splits, native categorical handling |
| **PyTorch TabularNN** | Multi-Layer Perceptron + Entity Embeddings | **0.8728** | **0.5702** | Smooth geometric representations, dense customer embeddings |

## Running the Pipelines

### 1. Requirements
```bash
pip install torch numpy pandas scikit-learn lightgbm
```

### 2. Train LightGBM Baseline
```bash
python train_lightgbm_from_db.py
```

### 3. Train PyTorch Deep Learning Model
```bash
python train_deeplearning_from_db.py
```
