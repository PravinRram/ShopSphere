"""Batches for the two-branch model: static features + event sequences, per snapshot row.

Preprocessing is fitted on the TRAINING rows of each fold only, so no statistic from a
validation customer leaks into training:
  - numeric static features: missing -> flag + training median; then a quantile transform to
    a normal distribution (spend and counts are heavy-tailed; networks are scale-sensitive);
  - categorical static features: integer codes (0 = unseen);
  - snapshot month: sin/cos, so December sits next to January.

Sequences come from the EventStore at each row's own cutoff. They are right-aligned (events
first, padding after) so the GRU can skip padding with pack_padded_sequence. Count features
in the events go through log1p.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer

from .features_static import CATEGORICAL, FEATURES
from .sequences import EventStore, NUMERIC as SEQ_NUMERIC

NUMERIC_STATIC = [f for f in FEATURES if f not in CATEGORICAL]
SEQ_COUNT_COLS = [SEQ_NUMERIC.index(c) for c in ["pages_viewed", "cart_additions", "items_count"]]


class StaticPreprocessor:
    """Fit on training rows only; apply to any rows.

    `numeric` and `categorical` default to the 39 base features (E4, E4T). E7 passes the longer
    lists from features_extra.feature_lists. The lists are stored with the fitted object, so an
    exported model always transforms exactly the columns it was trained on."""

    def __init__(self, numeric: list[str] | None = None, categorical: list[str] | None = None):
        self.numeric = list(numeric) if numeric is not None else list(NUMERIC_STATIC)
        self.categorical = list(categorical) if categorical is not None else list(CATEGORICAL)

    # Bundles exported before E7 (e4t_v1) pickled objects without these attributes: use the defaults.
    @property
    def _num(self) -> list[str]:
        return self.__dict__.get("numeric", NUMERIC_STATIC)

    @property
    def _cat(self) -> list[str]:
        return self.__dict__.get("categorical", CATEGORICAL)

    def fit(self, rows: pd.DataFrame, seed: int = 0) -> "StaticPreprocessor":
        x = rows[self._num].astype(float)
        self.missing_cols = [c for c in self._num if x[c].isna().any()]
        self.medians = x.median()
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                      subsample=200_000, random_state=seed)
        self.qt.fit(x.fillna(self.medians).to_numpy())
        self.vocab = {c: {v: i + 1 for i, v in enumerate(sorted(rows[c].astype(str).unique()))} for c in self._cat}
        return self

    @property
    def n_numeric(self) -> int:
        return len(self._num) + len(self.missing_cols) + 2            # + missing flags + month sin/cos

    @property
    def cat_cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in self._cat]            # + 1 for code 0 (unseen)

    def transform(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = rows[self._num].astype(float)
        flags = x[self.missing_cols].isna().to_numpy(np.float32)
        num = self.qt.transform(x.fillna(self.medians).to_numpy()).astype(np.float32)
        month = rows.snapshot_month.to_numpy()
        cyc = np.stack([np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)], 1).astype(np.float32)
        cat = np.stack([rows[c].astype(str).map(self.vocab[c]).fillna(0).to_numpy(np.int64) for c in self._cat], 1)
        return np.concatenate([num, flags, cyc], 1), cat


class SnapshotRows(torch.utils.data.Dataset):
    """One item per snapshot row; sequences are cut from the event store at load time."""

    def __init__(self, rows: pd.DataFrame, prep: StaticPreprocessor, store: EventStore, n_events: int):
        self.customer_id = rows.customer_id.to_numpy()
        self.cutoff = pd.to_datetime(rows.cutoff).to_numpy()
        self.target = rows.target.to_numpy(np.float32)
        self.static_num, self.static_cat = prep.transform(rows)
        self.store, self.n_events = store, n_events

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, i):
        return i

    def collate(self, idx: list[int]) -> dict[str, torch.Tensor]:
        idx = np.asarray(idx)
        num, cat, time, mask, length = self.store.window(self.customer_id[idx], self.cutoff[idx], self.n_events)
        num[..., SEQ_COUNT_COLS] = np.log1p(num[..., SEQ_COUNT_COLS])
        # left-padded -> right-aligned: roll each row so its events start at position 0
        shift = (self.n_events - length)[:, None]
        cols = (np.arange(self.n_events)[None, :] + shift) % self.n_events
        rows_ = np.arange(len(idx))[:, None]
        return {
            "static_num": torch.from_numpy(self.static_num[idx]),
            "static_cat": torch.from_numpy(self.static_cat[idx]),
            "seq_num": torch.from_numpy(num[rows_, cols]),
            "seq_cat": torch.from_numpy(cat[rows_, cols]),
            "seq_time": torch.from_numpy(time[rows_, cols]),
            "seq_len": torch.from_numpy(length),
            "target": torch.from_numpy(self.target[idx]),
        }


def loader(ds: SnapshotRows, batch_size: int, shuffle: bool, seed: int = 0, num_workers: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                                       collate_fn=ds.collate, num_workers=num_workers)
