> These are the cleaned files, committed so they can be inspected. `sessions.csv` is stored gzip-compressed as
> `sessions.csv.gz` (21 MB; GitHub blocks files over 100 MB); read it with `pd.read_csv('sessions.csv.gz')`.
> Rebuild everything from the data pack with [`personA_browsing.ipynb`](../personA_browsing.ipynb) (about 20 seconds).

# Browsing dataset (Person A)

Cleaned output of `personA_browsing.ipynb`. Source: ShopSphere data pack, snapshot 30 Sep 2024.
Cleaning only fixes representation. Nothing here is fitted: imputation, scaling and
transforms happen in the model pipeline, on training rows only.

## Files

| File | Rows | Grain |
|---|---|---|
| sessions.csv (committed as sessions.csv.gz) | 1,442,550 | one website/app session |
| geo_data.csv | 100 | one region |
| campaign_daily_performance.csv | 11,867 | one campaign-day |
| quality_log.csv | 10 | one data-quality issue found and how it was handled |

## sessions.csv columns

| Column | Type | Notes |
|---|---|---|
| session_id | int | unique |
| customer_id | int | joins customers, train, test |
| session_date | date | renamed from session_timestamp (date only, no time). All <= cutoff |
| session_duration | float, seconds | **null for 1,484 corrupted sessions** (see duration_corrupt) |
| pages_viewed | float | 71,609 still null (impute in pipeline). Bounced nulls set to 1 |
| cart_additions | int | unchanged. 144 sessions add more items than pages viewed: kept (one page can add several items, see 2.5) |
| bounce_flag | bool | six source encodings mapped. Rate ~0.5%, implausibly low: treat as weak |
| traffic_source | str | lowercase: ads, direct, email, organic, social |
| device_type | str | lowercase: desktop, mobile, tablet |
| campaign_id | nullable int | blank = not from a campaign (documented meaning, not missing) |
| geo_ip_region | str | joins geo_data. May differ from the customer's home region |
| from_campaign | bool | campaign_id is not blank |
| pages_viewed_missing | bool | pages_viewed was null in the source |
| duration_corrupt | bool | source duration > 5h: corrupted cluster, value nulled |

## geo_data.csv and campaign_daily_performance.csv

Validated, unchanged apart from `date` parsed to a date. Both passed every integrity check.
campaign_daily_performance reconciles exactly with sessions.csv (100,401 campaign sessions on
both sides), so it's a pre-aggregated view of the same data, useful for the W6 dashboard.
`spend` is actual; marketing_campaigns.campaign_budget is the plan (94 campaigns overspent, max 5.8%).

## Not done here, on purpose

- Imputing pages_viewed / session_duration: fit on the training split only.
- Feature engineering and the 30 Sep 2024 cutoff for features: features.py.
