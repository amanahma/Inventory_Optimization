# Week 3 — OR Inventory Optimization Layer

Builds a classical Operations-Research inventory policy on top of the Week-2
forecasts: safety stock, EOQ, reorder point, a Newsvendor model for perishable
FOODS, a budget-constrained LP, and a naive-vs-optimized business-impact study.

## ⚠️ Cost assumptions (M5 has NO cost data — these are standard industry values)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `HOLDING_COST_RATE` | 0.20 | 20% of item value held per year |
| `ORDERING_COST` | $5.00 | cost per order placed |
| `LEAD_TIME_DAYS` | 7 | supplier lead time (1 week) |
| `WORKING_DAYS_YEAR` | 365 | daily demand data |
| `SERVICE_LEVEL` | A 98% / B 95% / C 90% | fill-rate target by ABC class |
| `MARGIN_RATE` | 0.30 | profit margin on sell price (Newsvendor underage) |
| `DISPOSAL_COST_RATE` | 0.10 | waste/markdown cost (Newsvendor overage) |
| Budgets | $50K / $100K / $200K | LP procurement scenarios |

All dollar figures below are model estimates under these assumptions, not
observed costs.

## Core idea

Safety stock is driven by the **standard deviation of forecast ERROR**
(`residual = actual − best_forecast`), **not** raw demand std. An accurate
model ⇒ small residuals ⇒ small `sigma_L = forecast_error_std × √lead_time` ⇒
small, cheap safety stock. This is the direct financial link from forecasting
accuracy (Week 2) to inventory cost (Week 3).

Best model per item-store: **Croston SBA** for intermittent `Z`-class series,
**LightGBM** otherwise.

## Pipeline (run in order, all from `src/`)

| # | Script | Output |
|---|--------|--------|
| 1 | `config.py` | inventory constants (appended to Week-2 config) |
| 2 | `forecast_error_stats.py` | `item_forecast_stats.csv` |
| 3 | `safety_stock.py` | `safety_stock.csv` |
| 4 | `eoq.py` | `eoq_results.csv` |
| 5 | `reorder_point.py` | `reorder_point_results.csv` |
| 6 | `newsvendor.py` | `newsvendor_results.csv` (FOODS only) |
| 7 | `pulp_optimization.py` | `pulp_optimization_results.csv` + chart |
| 8 | `policy_comparison.py` | `policy_comparison.csv` + 2 charts |
| 9 | `save_inventory_to_db.py` | SQLite `fact_inventory_policy / fact_newsvendor / fact_pulp_optimization` |
| 10 | `export_for_powerbi.py` | `powerbi_*.csv` (4 tables + dim_date) |
| — | `week3_verification.py` | final acceptance checks |

## Headline results (CA_1 sample — single-store memory fallback, 3,049 item-stores)

- **EOQ vs naive fixed-30 policy:** 49.1% annual cost reduction.
- **Full policy (EOQ + safety stock + stockout cost) vs naive:** $358,804 → $163,704,
  **$195,100 saved (54.4%)**; service level 85% → 94.2% (item-weighted).
- **Newsvendor (FOODS, CR = 0.75):** stock at the 75th percentile of demand;
  total expected profit improvement vs naive-EOQ ordering ≈ $38,096.
- **PuLP LP:** at single-store scope the A+B daily demand costs only ~$12,636 to
  fully procure, so the budget constraint is **non-binding** — all three
  scenarios reach 100% fill. (Constraint would bite at full 10-store scope or a
  multi-day procurement horizon.)

## Scope note

Forecasts cover **CA_1 only** (memory fallback — see `config.DATA_SCOPE`). The
`abc_xyz_classification.csv` covers all 10 stores; everything here is filtered to
the CA_1 item-stores that have forecasts.
