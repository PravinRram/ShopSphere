"""Extra static features for E7: profile and consent columns, and regional data.

E4/E4T use only the 39 features in features_static, which can be rebuilt at every cutoff.
E7 adds two groups on top:

  crm  8 customer profile and consent columns that exist only as their value on 30 Sep 2024
       (DATA_README: customers.csv is the "profile as at the cutoff"; the latest
       customer_preferences.last_updated is 2024-09-30). There is no history, so they are
       filled in on the real 30 Sep 2024 snapshot ONLY. On every earlier snapshot they are
       left missing: the 30 Sep 2024 value would tell an earlier row about months after its
       own cutoff. The preprocessor's missing flags then tell the model which rows have them.

  geo  4 facts about the customer's region (geo_data), joined on customers.region. They
       describe the region, not the customer's behaviour, carry no dates, and are used on
       every snapshot, like country_code.

Not used: discount_usage_rate (identical to discount_order_ratio, correlation 1.0),
preferred_language (no gain in the check of 18 Nov), credit_limit (noise),
last_updated (a date, not needed), referral_code.

    rows = add_extra(rows, load_extra(db), ("crm", "geo"))
    num, cat = feature_lists(("crm", "geo"))     # column lists for StaticPreprocessor
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from .cutoff_rules import SNAPSHOT
from .features_static import CATEGORICAL, FEATURES

CRM_NUMERIC = ["loyalty_score", "email_open_rate", "email_open_rate_is_missing", "avg_review_score",
               "email_opt_in", "sms_opt_in", "do_not_contact"]
CRM_CATEGORICAL = ["preferred_channel"]
GEO_NUMERIC = ["average_income", "urban_ratio", "internet_penetration"]
GEO_CATEGORICAL = ["region_tier"]

GROUPS = {"crm": (CRM_NUMERIC, CRM_CATEGORICAL), "geo": (GEO_NUMERIC, GEO_CATEGORICAL)}
SNAPSHOT_ONLY = {"crm"}                         # groups that exist only as at 30 Sep 2024
NOT_KNOWN = "not_known"                         # categorical value on snapshots where a column is unknown


def parse_groups(extra) -> tuple[str, ...]:
    """'crm,geo' or ('crm', 'geo') or '' -> a tuple of known group names."""
    if isinstance(extra, str):
        extra = [g for g in extra.split(",") if g]
    extra = tuple(extra or ())
    unknown = set(extra) - set(GROUPS)
    assert not unknown, f"unknown feature groups: {unknown}"
    return extra


def feature_lists(extra=()) -> tuple[list[str], list[str]]:
    """Numeric and categorical static columns for a run: the 39 base features plus the groups."""
    numeric = [f for f in FEATURES if f not in CATEGORICAL]
    categorical = list(CATEGORICAL)
    for g in parse_groups(extra):
        numeric += GROUPS[g][0]
        categorical += GROUPS[g][1]
    return numeric, categorical


def load_extra(db_path: str) -> pd.DataFrame:
    """One row per customer: the crm and geo columns (read-only)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    df = pd.read_sql_query("""
        SELECT c.customer_id, c.loyalty_score, c.email_open_rate, c.email_open_rate_is_missing,
               c.avg_review_score, p.email_opt_in, p.sms_opt_in, p.do_not_contact, p.preferred_channel,
               g.average_income, g.urban_ratio, g.internet_penetration, g.region_tier
        FROM customers c
        LEFT JOIN customer_preferences p ON p.customer_id = c.customer_id
        LEFT JOIN geo_data g ON g.geo_ip_region = c.region""", con)
    con.close()
    return df


def add_extra(rows: pd.DataFrame, extra_df: pd.DataFrame, extra=()) -> pd.DataFrame:
    """Add the chosen groups to snapshot rows (columns customer_id and cutoff, or a real-snapshot
    frame without cutoff). Snapshot-only groups are blanked on every row whose cutoff is before
    30 Sep 2024."""
    groups = parse_groups(extra)
    if not groups:
        return rows
    cols = [c for g in groups for c in GROUPS[g][0] + GROUPS[g][1]]
    out = rows.merge(extra_df[["customer_id"] + cols], on="customer_id", how="left", validate="many_to_one")
    cutoff = pd.to_datetime(out["cutoff"]) if "cutoff" in out else pd.Series(SNAPSHOT, index=out.index)
    earlier = (cutoff != SNAPSHOT).to_numpy()
    for g in groups:
        num, cat = GROUPS[g]
        if g in SNAPSHOT_ONLY:
            out.loc[earlier, num] = float("nan")
            out.loc[earlier, cat] = NOT_KNOWN
        out[cat] = out[cat].fillna(NOT_KNOWN).astype(str)
    return out
