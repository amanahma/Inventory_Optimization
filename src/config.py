"""
config.py  --  M5 Inventory Optimizer (Week 2)

Shared constants + helpers imported by every Week-2 script.

GOLDEN RULE: forecasting here is ALWAYS split by time, never randomly.
  - Train      = rows with date <= cutoff date (the date of TRAIN_END_DAY)
  - Validation = rows with date >  cutoff date  (= the 28-day horizon)
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
PROJECT = r"C:\Users\AMAN AHMAD\Documents\m5-inventory-optimizer"
RAW = os.path.join(PROJECT, "data", "raw")
PROC = os.path.join(PROJECT, "data", "processed")
OUT = os.path.join(PROJECT, "outputs")
DB_PATH = os.path.join(PROJECT, "data", "m5_database.db")

CALENDAR_CSV = os.path.join(RAW, "calendar.csv")
ABC_XYZ_CSV = os.path.join(OUT, "abc_xyz_classification.csv")
SALES_LONG = os.path.join(PROC, "sales_long.parquet")
SALES_CA1 = os.path.join(PROC, "sales_CA1_sample.parquet")
TRAIN_PARQUET = os.path.join(PROC, "train_prepared.parquet")
VAL_PARQUET = os.path.join(PROC, "val_prepared.parquet")

# ---------------------------------------------------------------- split config
TRAIN_END_DAY = "d_1885"     # train on d_1 .. d_1885
VAL_START_DAY = "d_1886"     # validate on d_1886 .. d_1913
FORECAST_HORIZON = 28        # always 28 days ahead
RANDOM_SEED = 42

CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]
ALL_STORES = ["CA_1", "CA_2", "CA_3", "CA_4",
              "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]

# ---------------------------------------------------------------- memory fallback
# This machine has ~7.3 GB RAM. The full 58M-row long dataset plus 28-day lag /
# rolling feature engineering (LightGBM step) needs ~6-8 GB and OOMs reliably, so
# the Week-2 pipeline runs on the single-store CA_1 sample. Every script prints a
# clear note when this fallback base is in use. Flip USE_FULL_DATASET to True only
# on a larger machine.
USE_FULL_DATASET = False
BASE_PARQUET = SALES_LONG if USE_FULL_DATASET else SALES_CA1
DATA_SCOPE = "FULL (all 10 stores)" if USE_FULL_DATASET else "CA_1 SAMPLE (memory fallback)"


def get_cutoff_date():
    """Return (cutoff_date as pd.Timestamp, calendar df) by looking up TRAIN_END_DAY."""
    cal = pd.read_csv(CALENDAR_CSV, parse_dates=["date"])
    cutoff = cal.loc[cal["d"] == TRAIN_END_DAY, "date"].iloc[0]
    return cutoff, cal


# ---------------------------------------------------------------- metrics
def rmse(actual, forecast):
    actual = np.asarray(actual, dtype="float64")
    forecast = np.asarray(forecast, dtype="float64")
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual, forecast):
    actual = np.asarray(actual, dtype="float64")
    forecast = np.asarray(forecast, dtype="float64")
    return float(np.mean(np.abs(actual - forecast)))


def mape(actual, forecast):
    """MAPE over rows where actual > 0 only (in %)."""
    actual = np.asarray(actual, dtype="float64")
    forecast = np.asarray(forecast, dtype="float64")
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - forecast[mask]) / actual[mask]) * 100.0)


def bias(actual, forecast):
    """mean(forecast - actual); positive => over-forecasting."""
    actual = np.asarray(actual, dtype="float64")
    forecast = np.asarray(forecast, dtype="float64")
    return float(np.mean(forecast - actual))


def metrics_by_category(df, forecast_col, cat_col="cat_id", actual_col="actual"):
    """Return a tidy DataFrame: cat_id, RMSE, MAE, MAPE, Bias for one forecast column."""
    rows = []
    for cat in CATEGORIES:
        sub = df[df[cat_col] == cat]
        if len(sub) == 0:
            continue
        rows.append({
            "cat_id": cat,
            "RMSE": rmse(sub[actual_col], sub[forecast_col]),
            "MAE": mae(sub[actual_col], sub[forecast_col]),
            "MAPE": mape(sub[actual_col], sub[forecast_col]),
            "Bias": bias(sub[actual_col], sub[forecast_col]),
        })
    return pd.DataFrame(rows)


# ── Inventory optimization parameters (Week 3) ─────────────────────────────
# M5 dataset has no cost data. These are standard industry assumptions.
# Always state these assumptions explicitly in any output or README.

HOLDING_COST_RATE   = 0.20   # 20% of item value per year (standard assumption)
ORDERING_COST       = 5.00   # $5 per order placed
LEAD_TIME_DAYS      = 7      # 7 days supplier lead time (1 week)
WORKING_DAYS_YEAR   = 365    # Use 365 for daily demand data

# Service level targets by ABC class
# Higher value items get higher service level targets
SERVICE_LEVEL = {
    'A': 0.98,   # 98% fill rate — A items are high revenue, stockout is costly
    'B': 0.95,   # 95% fill rate
    'C': 0.90,   # 90% fill rate — C items are low revenue, less critical
}

# Newsvendor parameters (for FOODS category — perishable items)
MARGIN_RATE       = 0.30   # 30% profit margin on sell price
DISPOSAL_COST_RATE= 0.10   # 10% of sell price as waste/markdown cost

# Budget scenarios for PuLP optimization (Task 7)
BUDGET_TIGHT   = 50_000   # $50,000 procurement budget (constrained scenario)
BUDGET_NORMAL  = 100_000  # $100,000 (base scenario)
BUDGET_RELAXED = 200_000  # $200,000 (unconstrained scenario)


if __name__ == "__main__":
    cutoff, _ = get_cutoff_date()
    print("=" * 70)
    print("Week 2 configuration")
    print("=" * 70)
    print(f"TRAIN_END_DAY     : {TRAIN_END_DAY}")
    print(f"VAL_START_DAY     : {VAL_START_DAY}")
    print(f"FORECAST_HORIZON  : {FORECAST_HORIZON}")
    print(f"RANDOM_SEED       : {RANDOM_SEED}")
    print(f"CUTOFF DATE       : {cutoff.date()}   "
          f"(train: date <= this; val: date > this)")
    print(f"DATA SCOPE        : {DATA_SCOPE}")
    print(f"BASE PARQUET      : {os.path.basename(BASE_PARQUET)}")

    print("\n" + "=" * 70)
    print("Week 3 inventory optimization parameters")
    print("=" * 70)
    print(f"HOLDING_COST_RATE : {HOLDING_COST_RATE}   (20% of item value / year)")
    print(f"ORDERING_COST     : ${ORDERING_COST:.2f}   per order placed")
    print(f"LEAD_TIME_DAYS    : {LEAD_TIME_DAYS}      days supplier lead time")
    print(f"WORKING_DAYS_YEAR : {WORKING_DAYS_YEAR}")
    print(f"SERVICE_LEVEL     : {SERVICE_LEVEL}")
    print(f"MARGIN_RATE       : {MARGIN_RATE}   (30% profit margin)")
    print(f"DISPOSAL_COST_RATE: {DISPOSAL_COST_RATE}   (10% waste/markdown cost)")
    print(f"BUDGET_TIGHT      : ${BUDGET_TIGHT:,}")
    print(f"BUDGET_NORMAL     : ${BUDGET_NORMAL:,}")
    print(f"BUDGET_RELAXED    : ${BUDGET_RELAXED:,}")
    print("\nASSUMPTION NOTE: M5 has no cost data; the rates above are standard")
    print("industry assumptions and are stated explicitly in every output/README.")
