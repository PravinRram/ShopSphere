CANVA ASSET PACK - real figures from the student's notebooks
=============================================================
Use these to fill the figure placeholders on the slide. File codes (A1, B1, ...) match
Section 13 of MASTER_REPORT.md and the "FIG xx" labels in the slide spec.

A = data cleaning and EDA (01_EDA_and_Cleaning.ipynb)
  A1_signups_per_month_raw.png                 raw signups / preference updates: pile-ups on first and last day
  A2_target_distribution.png                   54,674 (85.4%) no purchase vs 9,326 (14.6%) purchase
  A3_numeric_feature_distributions.png         buyers vs non-buyers for each numeric feature
  A4_spearman_correlation_heatmap.png          correlation with the target; the 0.94 redundancy
  A5_purchase_rate_by_category.png             purchase rate by tier, country, opt-in, channel, missing flags
  A6_loyalty_signup_year_region.png            purchase rate by loyalty decile, signup year and region
  A7_customer_preferences.png                  preferred channel vs opt-in; language by country; consent counts

B = model and evaluation
  B1_ft_transformer_architecture.png           six-box architecture diagram (108,161 parameters)
  B2_ft_calibration_and_learning_curves.png    calibration curve + fold-0 learning curves (FT, seed 42)
  B3_ft_roc_and_pr_curves.png                  out-of-fold ROC and precision-recall curves (FT, seed 42)
  B4_seed_noise_comparison.png                 per-seed results for both models (y-axes are zoomed)
  B5_lightgbm_eval_plots.png                   LightGBM ROC, PR, calibration, fold-0 AUC per tree
  B6_lightgbm_shap_and_gain_importance.png     top-15 features by SHAP and by split gain (LightGBM)

Suggested priority for a single dense slide: A2, A5 or A6, B1, B2 or B4, then A1 or A7.
Note: B4 uses zoomed axes (the gap is about 0.0014 ROC-AUC); say so if you show it.
