"""
export_for_powerbi.py  --  M5 Inventory Optimizer (Week 3, Task 10)

Exports clean, Power-BI-ready CSVs (no NaN, underscore column names) from the
SQLite inventory tables plus a date dimension for time intelligence.
"""

import os
import sqlite3
import numpy as np
import pandas as pd

import config as cfg


def clean(df):
    """Fill NaN: 0 for numeric, 'UNKNOWN' for text; strip spaces in col names."""
    df = df.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0)
        else:
            df[c] = df[c].fillna("UNKNOWN")
    return df


def main():
    print("=" * 72)
    print("TASK 10 — Export final tables for Power BI")
    print("=" * 72)
    con = sqlite3.connect(cfg.DB_PATH)

    # ---------------------------------------------------------------- Step 1
    inv = clean(pd.read_sql_query("SELECT * FROM fact_inventory_policy", con))
    p1 = os.path.join(cfg.OUT, "powerbi_inventory_policy.csv")
    inv.to_csv(p1, index=False)
    print(f"[Step 1] {os.path.basename(p1)}: {len(inv):,} rows")
    print(f"         columns: {list(inv.columns)}")

    # ---------------------------------------------------------------- Step 2
    nv = clean(pd.read_sql_query("SELECT * FROM fact_newsvendor", con))
    p2 = os.path.join(cfg.OUT, "powerbi_newsvendor.csv")
    nv.to_csv(p2, index=False)
    print(f"[Step 2] {os.path.basename(p2)}: {len(nv):,} rows")

    # ---------------------------------------------------------------- Step 3
    pu = clean(pd.read_sql_query("SELECT * FROM fact_pulp_optimization", con))
    p3 = os.path.join(cfg.OUT, "powerbi_pulp_scenarios.csv")
    pu.to_csv(p3, index=False)
    print(f"[Step 3] {os.path.basename(p3)}: {len(pu):,} rows")

    # ---------------------------------------------------------------- Step 4
    # Main dashboard table: inventory policy LEFT JOIN forecast stats.
    # item_forecast_stats is not in SQLite -> load CSV and merge in pandas.
    inv_db = pd.read_sql_query("""
        SELECT item_id, store_id, cat_id, dept_id, state_id,
               abc_class, xyz_class, abc_xyz,
               sell_price, forecast_mean,
               safety_stock, EOQ, ROP,
               DOH, turnover, stockout_risk,
               total_annual_cost_EOQ, total_annual_cost_naive, annual_saving,
               annual_saving_pct, best_model_used
        FROM fact_inventory_policy
    """, con)
    stats = pd.read_csv(os.path.join(cfg.OUT, "item_forecast_stats.csv"))[
        ["item_id", "store_id", "forecast_rmse", "forecast_bias"]]
    dash = inv_db.merge(stats, on=["item_id", "store_id"], how="left")
    dash = clean(dash)
    p4 = os.path.join(cfg.OUT, "powerbi_main_dashboard.csv")
    dash.to_csv(p4, index=False)
    print(f"[Step 4] {os.path.basename(p4)}: {len(dash):,} rows "
          f"(should match item-store count = {len(inv_db):,})")

    con.close()

    # ---------------------------------------------------------------- Step 5
    cal = pd.read_csv(cfg.CALENDAR_CSV, parse_dates=["date"])
    cal["is_weekend"] = cal["date"].dt.dayofweek.isin([5, 6]).astype(int)
    cal["quarter"] = cal["date"].dt.quarter
    cal["fiscal_year"] = cal["date"].dt.year  # fiscal year = calendar year
    cal = clean(cal)
    p5 = os.path.join(cfg.OUT, "powerbi_dim_date.csv")
    cal.to_csv(p5, index=False)
    print(f"[Step 5] {os.path.basename(p5)}: {len(cal):,} rows")

    # ---------------------------------------------------------------- Step 6
    print("\n" + "-" * 72)
    print("POWER BI EXPORT FILES SUMMARY")
    print("-" * 72)
    print(f"{'filename':<38}{'rows':>10}{'cols':>8}{'size_KB':>12}")
    for f in sorted(os.listdir(cfg.OUT)):
        if f.startswith("powerbi_") and f.endswith(".csv"):
            path = os.path.join(cfg.OUT, f)
            d = pd.read_csv(path)
            kb = os.path.getsize(path) / 1024
            print(f"{f:<38}{len(d):>10,}{d.shape[1]:>8}{kb:>11.1f}K")
    print("=" * 72)


if __name__ == "__main__":
    main()
