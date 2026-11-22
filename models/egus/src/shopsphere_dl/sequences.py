"""Event store for the GRU branch: sessions and orders, stored once, sliced at any cutoff.

Storing ~700,000 padded sequences (one per snapshot row) would take gigabytes. Instead each
customer's full event list is stored once, sorted by time, and a row's sequence is cut out
when it is loaded: binary-search the cutoff, take the last N events on or before it, and
compute the time features relative to that row's cutoff.

Event order: by date, then sessions before orders on the same day (browse, then buy), then
by ID. Dates are whole days, so this tie-break is fixed and documented, not random.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cutoff_rules import SNAPSHOT, Tables

EPOCH = pd.Timestamp("2022-01-01")
# Categorical codes start at 1 in sorted order (event_type: order = 1, session = 2); 0 = padding.

NUMERIC = ["log_duration", "pages_viewed", "pages_missing", "cart_additions", "bounce",
           "log_order_value", "items_count", "discount_applied", "high_value"]
CATEGORICAL = ["event_type", "traffic_source", "device_type", "from_campaign", "payment_method", "shipping_speed"]
TIME = ["log_days_before_cutoff", "log_gap_days"]


def _codes(s: pd.Series) -> tuple[np.ndarray, list]:
    """Integer codes starting at 1; 0 means padding or 'not applicable to this event type'."""
    cats = sorted(s.dropna().unique().tolist())
    mapping = {v: i + 1 for i, v in enumerate(cats)}
    return s.map(mapping).fillna(0).astype(np.int64).values, cats


@dataclass
class EventStore:
    customer_ids: np.ndarray         # sorted customer IDs with at least one event
    offsets: np.ndarray              # events of customer_ids[i] are rows offsets[i]:offsets[i+1]
    day: np.ndarray                  # event date as days since 2022-01-01
    numeric: np.ndarray              # [n_events, len(NUMERIC)] float32
    categorical: np.ndarray          # [n_events, len(CATEGORICAL)] int64
    vocab: dict                      # category lists, for embedding sizes

    @classmethod
    def from_tables(cls, t: Tables) -> "EventStore":
        s, x = t.sessions, t.transactions
        sess = pd.DataFrame({
            "customer_id": s.customer_id.values, "date": s.session_timestamp.values,
            "type_order": 0, "event_id": s.session_id.values, "event_type": "session",
            "log_duration": np.log1p(s.session_duration_capped.values),
            "pages_viewed": s.pages_viewed.where(s.pages_viewed_is_missing == 0).fillna(0).values,
            "pages_missing": s.pages_viewed_is_missing.values,
            "cart_additions": s.cart_additions.values, "bounce": s.bounce_flag.values,
            "traffic_source": s.traffic_source.values, "device_type": s.device_type.values,
            "from_campaign": np.where(s.campaign_id.notna(), "campaign", "none"),
        })
        orders = pd.DataFrame({
            "customer_id": x.customer_id.values, "date": x.transaction_timestamp.values,
            "type_order": 1, "event_id": x.transaction_id.values, "event_type": "order",
            "log_order_value": np.log1p(x.order_value.values), "items_count": x.items_count.values,
            "discount_applied": x.discount_applied.values, "high_value": x.high_value_flag.values,
            "payment_method": x.payment_method.values, "shipping_speed": x.shipping_speed.values,
        })
        ev = pd.concat([sess, orders], ignore_index=True)
        if ev.date.max() > SNAPSHOT:
            raise ValueError("event store contains an event after the data snapshot")
        ev = ev.sort_values(["customer_id", "date", "type_order", "event_id"], kind="stable").reset_index(drop=True)
        vocab = {}
        cat = np.zeros((len(ev), len(CATEGORICAL)), dtype=np.int64)
        for j, col in enumerate(CATEGORICAL):
            cat[:, j], vocab[col] = _codes(ev[col])
        num = ev[NUMERIC].fillna(0).to_numpy(np.float32)
        cust, start = np.unique(ev.customer_id.values, return_index=True)
        offsets = np.append(start, len(ev))
        day = ((ev.date - EPOCH).dt.days).to_numpy(np.int32)
        return cls(cust, offsets, day, num, cat, vocab)

    def window(self, customer_ids, cutoffs, n: int):
        """Last n events on or before each row's cutoff, left-padded.

        Returns numeric [B, n, len(NUMERIC)], categorical [B, n, len(CATEGORICAL)],
        time [B, n, 2], mask [B, n] (True = real event) and length [B].
        A customer with no events before the cutoff gets length 0 (the model adds a
        'no history' token).
        """
        customer_ids = np.asarray(customer_ids)
        cut_days = ((pd.to_datetime(pd.Series(cutoffs)) - EPOCH).dt.days).to_numpy(np.int32)
        b = len(customer_ids)
        numeric = np.zeros((b, n, self.numeric.shape[1]), np.float32)
        categorical = np.zeros((b, n, self.categorical.shape[1]), np.int64)
        time = np.zeros((b, n, len(TIME)), np.float32)
        mask = np.zeros((b, n), bool)
        length = np.zeros(b, np.int64)
        pos = np.searchsorted(self.customer_ids, customer_ids)
        found = (pos < len(self.customer_ids)) & (self.customer_ids[np.minimum(pos, len(self.customer_ids) - 1)] == customer_ids)
        for i in np.where(found)[0]:
            lo, hi = self.offsets[pos[i]], self.offsets[pos[i] + 1]
            end = lo + np.searchsorted(self.day[lo:hi], cut_days[i], side="right")   # events with day <= cutoff
            start = max(lo, end - n)
            k = end - start
            if k == 0:
                continue
            days = self.day[start:end]
            numeric[i, n - k:] = self.numeric[start:end]
            categorical[i, n - k:] = self.categorical[start:end]
            time[i, n - k:, 0] = np.log1p(cut_days[i] - days)
            prev = self.day[start - 1] if start > lo else days[0]       # gap to the event before the window
            time[i, n - k:, 1] = np.log1p(np.diff(np.concatenate([[prev], days])))
            mask[i, n - k:] = True
            length[i] = k
        return numeric, categorical, time, mask, length
