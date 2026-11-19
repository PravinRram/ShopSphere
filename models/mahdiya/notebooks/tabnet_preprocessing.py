"""Preprocessing for the ShopSphere TabNet repurchase model.

Kept in its own module (not inside the notebook) so the fitted object saved in
models/tabnet_artifacts.joblib can be loaded by any other script, e.g. the agent:

    import joblib
    from tabnet_preprocessing import TabNetPreprocessor  # noqa: F401 (needed for unpickling)
    artifacts = joblib.load('models/tabnet_artifacts.joblib')
    X = artifacts['preprocessor'].transform(customer_rows)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TabNetPreprocessor:
    """Fit-on-train / transform-anywhere preprocessing for TabNet.

    fit(): learns medians, which columns to log1p, category vocabularies and scaler on the TRAIN split only.
    transform(): adds _isna indicators, imputes, log-transforms, scales and integer-encodes categoricals (0 = unseen).
    """
    def __init__(self, numeric_cols, categorical_cols, skew_threshold=2.0):
        self.numeric_cols, self.categorical_cols, self.skew_threshold = list(numeric_cols), list(categorical_cols), skew_threshold

    def fit(self, data):
        num = data[self.numeric_cols]
        self.nan_cols_ = [c for c in self.numeric_cols if num[c].isna().any()]
        self.medians_ = num.median()
        self.log_cols_ = [c for c in self.numeric_cols
                          if num[c].min() >= 0 and num[c].skew() > self.skew_threshold]
        self.vocab_ = {c: sorted(data[c].astype(str).unique()) for c in self.categorical_cols}
        self.scaler_ = StandardScaler().fit(self._numeric_block(data))
        self.columns_ = self.numeric_cols + [c + '_isna' for c in self.nan_cols_] + self.categorical_cols
        self.cat_idxs_ = [self.columns_.index(c) for c in self.categorical_cols]
        self.cat_dims_ = [len(self.vocab_[c]) + 1 for c in self.categorical_cols]   # +1 for 'unseen'
        return self

    def _numeric_block(self, data):
        num = data[self.numeric_cols].fillna(self.medians_)
        for c in self.log_cols_:
            num[c] = np.log1p(num[c].clip(lower=0))
        return num

    def transform(self, data):
        out = pd.DataFrame(self.scaler_.transform(self._numeric_block(data)), columns=self.numeric_cols, index=data.index)
        for c in self.nan_cols_:
            out[c + '_isna'] = data[c].isna().astype(float)
        for c in self.categorical_cols:
            mapping = {v: i + 1 for i, v in enumerate(self.vocab_[c])}
            out[c] = data[c].astype(str).map(mapping).fillna(0).astype(float)
        return out[self.columns_].values.astype(np.float32)
