"""The strict cutoff ruling, in one place.

`as_of(tables, cutoff)` returns every table exactly as it was known at the end of the
cutoff day. Feature code only ever sees these as-of tables, so no feature can look past
its cutoff. Every member uses this module, so the ruling is applied identically.

Rules (lecturer ruling, 5 Nov 2026; claude/shopsphere_db_guide.md, "Cutoff rules"):
  1. No row dated after the cutoff.
  2. No date field after the cutoff: such dates become missing (NaT).
  3. delay_days is known only when promised_date and delivered_date are both <= cutoff.
  4. Retention outcomes (purchased_within_14d, redeemed, redemption_date) are known only
     for contacts at least 14 days before the cutoff, so their window has closed.
  5. A ticket is open at the cutoff unless resolved by then; resolution_hours and
     csat_score are known only for tickets resolved by the cutoff (no open ticket in the
     data has a CSAT score).
  6. A parcel is known to be lost once its "lost parcel" ticket exists; a lost parcel
     with no such ticket is treated as lost 7 days after its promised date.
  7. Customer columns stored "as at 30 Sep 2024" with no history are never used:
     loyalty_score, email_open_rate, avg_review_score, credit_limit, customer_tier
     (tier is recomputed from spend), discount_usage_rate (recomputed), and the
     preference-centre table.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp("2024-09-30")      # the data pack's cutoff
OUTCOME_WINDOW_DAYS = 14                   # retention outcome window
LOST_FALLBACK_DAYS = 7                     # lost parcels with no ticket

# Columns that exist only "as at 30 Sep 2024" and are never passed to feature code.
AS_AT_COLUMNS = ["loyalty_score", "email_open_rate", "email_open_rate_is_missing",
                 "avg_review_score", "credit_limit", "customer_tier", "discount_usage_rate",
                 "referral_code"]


@dataclass(frozen=True)
class Tables:
    customers: pd.DataFrame
    sessions: pd.DataFrame
    transactions: pd.DataFrame
    deliveries: pd.DataFrame
    tickets: pd.DataFrame
    contacts: pd.DataFrame


def load_tables(db_path: str) -> Tables:
    """Read the behaviour tables from shopsphere.db (read-only) with parsed dates."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    q = lambda sql: pd.read_sql_query(sql, con)
    customers = q("SELECT customer_id, age, age_is_missing, country_code, region, signup_date FROM customers")
    sessions = q("""SELECT session_id, customer_id, session_timestamp, session_duration_capped,
                           pages_viewed, pages_viewed_is_missing, cart_additions, bounce_flag,
                           traffic_source, device_type, campaign_id FROM sessions""")
    transactions = q("""SELECT transaction_id, customer_id, transaction_timestamp, order_value, items_count,
                               payment_method, discount_applied, shipping_speed, high_value_flag FROM transactions""")
    deliveries = q("""SELECT transaction_id, ship_date, promised_date, delivered_date, delay_days, delivery_status
                      FROM deliveries""")
    tickets = q("""SELECT ticket_id, customer_id, transaction_id, created_date, category, priority, status,
                          resolution_hours, csat_score FROM support_tickets""")
    contacts = q("""SELECT contact_id, customer_id, contact_date, offer_code, purchased_within_14d, redeemed,
                           redemption_date FROM retention_contact_history""")
    con.close()
    for df, cols in [(customers, ["signup_date"]), (sessions, ["session_timestamp"]),
                     (transactions, ["transaction_timestamp"]),
                     (deliveries, ["ship_date", "promised_date", "delivered_date"]),
                     (tickets, ["created_date"]), (contacts, ["contact_date", "redemption_date"])]:
        for c in cols:
            df[c] = pd.to_datetime(df[c])
    return Tables(customers, sessions, transactions, deliveries, tickets, contacts)


def ticket_resolved_at(tickets: pd.DataFrame) -> pd.Series:
    """When a ticket was resolved: created date + resolution hours (NaT if never resolved)."""
    return tickets.created_date + pd.to_timedelta(tickets.resolution_hours, unit="h")


def as_of(t: Tables, cutoff) -> Tables:
    """Every table as known at the end of `cutoff` (a date). See the module docstring."""
    c = pd.Timestamp(cutoff)
    if c > SNAPSHOT:
        raise ValueError(f"cutoff {c.date()} is after the data snapshot {SNAPSHOT.date()}")
    end_of_day = c + pd.Timedelta(days=1)           # anything strictly before this is known at c

    customers = t.customers[t.customers.signup_date <= c].copy()
    sessions = t.sessions[t.sessions.session_timestamp <= c].copy()
    transactions = t.transactions[t.transactions.transaction_timestamp <= c].copy()

    # Deliveries of known orders; future dates removed; outcomes only once known.
    d = t.deliveries[t.deliveries.transaction_id.isin(transactions.transaction_id)].copy()
    for col in ["ship_date", "promised_date", "delivered_date"]:
        d.loc[d[col] > c, col] = pd.NaT
    d["delay_days"] = d.delay_days.where(d.promised_date.notna() & d.delivered_date.notna())
    lost_ticket = (t.tickets[(t.tickets.category == "lost parcel") & t.tickets.transaction_id.notna()]
                   .groupby("transaction_id").created_date.min())
    lost_known_from = d.transaction_id.map(lost_ticket)
    raw_promised = t.deliveries.set_index("transaction_id").promised_date.reindex(d.transaction_id).values
    fallback = pd.Series(raw_promised, index=d.index) + pd.Timedelta(days=LOST_FALLBACK_DAYS)
    lost_known_from = lost_known_from.fillna(fallback)
    is_lost = (t.deliveries.set_index("transaction_id").delivery_status.reindex(d.transaction_id).values == "Lost")
    d["lost_known"] = (is_lost & (lost_known_from.values <= c)).astype(int)
    d["delivery_status"] = np.where(d.delivered_date.notna(), "Delivered",
                                    np.where(d.lost_known == 1, "Lost", "Pending"))

    # Tickets: open/closed as at c; resolution and CSAT only once resolved.
    k = t.tickets[t.tickets.created_date <= c].copy()
    resolved = ticket_resolved_at(k) < end_of_day
    k["status"] = np.where(resolved, "Closed", "Open")
    k["resolution_hours"] = k.resolution_hours.where(resolved)
    k["csat_score"] = k.csat_score.where(resolved)

    # Retention contacts: outcomes only when the 14-day window closed by c.
    r = t.contacts[t.contacts.contact_date <= c].copy()
    closed = r.contact_date <= c - pd.Timedelta(days=OUTCOME_WINDOW_DAYS)
    for col in ["purchased_within_14d", "redeemed"]:
        r[col] = r[col].where(closed)
    r["redemption_date"] = r.redemption_date.where(closed & (r.redemption_date <= c))

    return Tables(customers, sessions, transactions, d, k, r)


def date_columns(t: Tables) -> dict[str, pd.Series]:
    """Every date column in a Tables object, for the no-future-date test."""
    out = {}
    for name in ["customers", "sessions", "transactions", "deliveries", "tickets", "contacts"]:
        df = getattr(t, name)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                out[f"{name}.{col}"] = df[col]
    if len(t.tickets):
        out["tickets.resolved_at"] = ticket_resolved_at(t.tickets).dropna()
    return out
