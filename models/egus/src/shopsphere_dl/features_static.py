"""Static (customer-level) features at any cutoff.

`build(tables, cutoff)` first applies the cutoff rules (`as_of`), so every feature is
computed only from what was known at the end of the cutoff day. The definitions follow
`03_lightgbm_baseline_final.ipynb`, minus the columns that cannot be rebuilt for earlier
dates (see claude/pretraining_checks.md, section 7).

39 features + snapshot_month. The same function builds the real snapshot and every extra
snapshot, so they cannot drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cutoff_rules import Tables, as_of

TIER_BANDS = [-np.inf, 120, 400, 900, np.inf]                 # loyalty_programme.md, section 1
TIER_NAMES = ["Bronze", "Silver", "Gold", "Platinum"]
CATEGORICAL = ["country_code", "customer_tier"]

FEATURES = [
    # kept as is (3)
    "age", "age_is_missing", "country_code",
    # profile, recomputed (2)
    "customer_tier", "account_age_days",
    # orders (10)
    "total_orders", "total_spend_usd", "avg_order_value", "max_order_value", "total_items_bought",
    "discounted_orders_count", "high_value_orders_count", "orders_last_30d", "orders_last_90d", "recency_days",
    # sessions (9)
    "total_sessions", "total_pages_viewed", "sessions_pages_missing", "total_cart_additions",
    "total_bounced_sessions", "avg_session_duration_s", "sessions_last_30d", "sessions_last_90d",
    "days_since_last_session",
    # deliveries (4)
    "total_deliveries", "late_deliveries_count", "lost_packages_count", "avg_delay_days",
    # tickets (4)
    "total_support_tickets", "open_tickets_count", "high_priority_open_tickets", "avg_csat_score",
    # retention (3)
    "retention_contacts_count", "retention_redeemed_count", "retention_purchased_count",
    # derived (4)
    "discount_order_ratio", "bounce_rate", "cart_abandonment_ratio", "is_inactive_90d",
]
assert len(FEATURES) == 39


def build(tables: Tables, cutoff, customer_ids=None) -> pd.DataFrame:
    """One row per customer signed up by `cutoff`: customer_id, cutoff, snapshot_month, FEATURES."""
    c = pd.Timestamp(cutoff)
    t = as_of(tables, c)                                       # the only data feature code sees
    d30 = c - pd.Timedelta(days=29)                            # 30 days including the cutoff day (as in 03)
    d90 = c - pd.Timedelta(days=89)

    cust = t.customers[["customer_id", "age", "age_is_missing", "country_code", "signup_date"]].copy()
    if customer_ids is not None:
        cust = cust[cust.customer_id.isin(customer_ids)]
    cust["account_age_days"] = (c - cust.signup_date).dt.days.astype(float)
    out = cust.drop(columns="signup_date").set_index("customer_id")

    # Orders
    tx = t.transactions
    g = tx.groupby("customer_id")
    orders = pd.DataFrame({
        "total_orders": g.size(),
        "total_spend_usd": g.order_value.sum(),
        "avg_order_value": g.order_value.mean(),
        "max_order_value": g.order_value.max(),
        "total_items_bought": g.items_count.sum(),
        "discounted_orders_count": g.discount_applied.sum(),
        "high_value_orders_count": g.high_value_flag.sum(),
        "orders_last_30d": tx[tx.transaction_timestamp >= d30].groupby("customer_id").size(),
        "orders_last_90d": tx[tx.transaction_timestamp >= d90].groupby("customer_id").size(),
        "recency_days": (c - g.transaction_timestamp.max()).dt.days.astype(float),
    })
    spend_12m = tx[tx.transaction_timestamp > c - pd.DateOffset(years=1)].groupby("customer_id").order_value.sum()

    # Sessions
    s = t.sessions
    gs = s.groupby("customer_id")
    real_pages = s.pages_viewed.where(s.pages_viewed_is_missing == 0)
    sessions = pd.DataFrame({
        "total_sessions": gs.size(),
        "total_pages_viewed": real_pages.groupby(s.customer_id).sum(),
        "sessions_pages_missing": gs.pages_viewed_is_missing.sum(),
        "total_cart_additions": gs.cart_additions.sum(),
        "total_bounced_sessions": gs.bounce_flag.sum(),
        "avg_session_duration_s": gs.session_duration_capped.mean(),
        "sessions_last_30d": s[s.session_timestamp >= d30].groupby("customer_id").size(),
        "sessions_last_90d": s[s.session_timestamp >= d90].groupby("customer_id").size(),
        "days_since_last_session": (c - gs.session_timestamp.max()).dt.days.astype(float),
    })

    # Deliveries (already masked by as_of)
    dl = t.deliveries.merge(tx[["transaction_id", "customer_id"]], on="transaction_id")
    gd = dl.groupby("customer_id")
    deliveries = pd.DataFrame({
        "total_deliveries": gd.size(),
        "late_deliveries_count": (dl.delay_days >= 1).groupby(dl.customer_id).sum(),
        "lost_packages_count": gd.lost_known.sum(),
        "avg_delay_days": gd.delay_days.mean(),
    })

    # Tickets (status as at c)
    k = t.tickets
    gk = k.groupby("customer_id")
    is_open = k.status == "Open"
    tickets = pd.DataFrame({
        "total_support_tickets": gk.size(),
        "open_tickets_count": is_open.groupby(k.customer_id).sum(),
        "high_priority_open_tickets": (is_open & (k.priority == "High")).groupby(k.customer_id).sum(),
        "avg_csat_score": gk.csat_score.mean(),
    })

    # Retention contacts (outcomes masked inside the 14-day window)
    r = t.contacts
    gr = r.groupby("customer_id")
    retention = pd.DataFrame({
        "retention_contacts_count": gr.size(),
        "retention_redeemed_count": gr.redeemed.sum(min_count=0),
        "retention_purchased_count": gr.purchased_within_14d.sum(min_count=0),
    })

    out = out.join([orders, sessions, deliveries, tickets, retention])
    counts = [c_ for c_ in FEATURES if c_.startswith(("total_", "orders_last", "sessions_last", "discounted_",
                                                       "high_value_", "late_", "lost_", "open_", "high_priority",
                                                       "retention_", "sessions_pages"))]
    out[counts] = out[counts].fillna(0)                        # no activity = 0, as COALESCE in 03
    out[["avg_order_value", "max_order_value"]] = out[["avg_order_value", "max_order_value"]].fillna(0)  # as 03
    out["customer_tier"] = pd.cut(spend_12m.reindex(out.index).fillna(0), TIER_BANDS, right=False,
                                  labels=TIER_NAMES).astype(str)

    # Derived ratios (as in 03)
    out["discount_order_ratio"] = np.where(out.total_orders > 0, out.discounted_orders_count / out.total_orders.where(out.total_orders > 0), 0.0)
    out["bounce_rate"] = np.where(out.total_sessions > 0, out.total_bounced_sessions / out.total_sessions.where(out.total_sessions > 0), 0.0)
    out["cart_abandonment_ratio"] = np.where(
        out.total_cart_additions > 0,
        np.clip(1 - out.total_orders / out.total_cart_additions.where(out.total_cart_additions > 0), 0, 1), 0.0)
    out["is_inactive_90d"] = ((out.orders_last_90d == 0) & (out.sessions_last_90d == 0)).astype(int)

    out = out.reset_index()
    out.insert(1, "cutoff", c)
    out.insert(2, "snapshot_month", c.month)
    return out[["customer_id", "cutoff", "snapshot_month"] + FEATURES]
