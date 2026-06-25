"""
forecast_error_stats.py  --  M5 Inventory Optimizer (Week 3, Task 2)

CRITICAL CONCEPT
----------------
Safety stock is driven by the standard deviation of FORECAST ERROR
(residual = actual - forecast), NOT the raw demand standard deviation.

Why this matters financially:
  * An ACCURATE model has small residuals  -> small forecast_error_std
    -> small safety stock -> less capital tied up -> lower holding cost.
  * A POOR model has large residuals -> large forecast_error_std
    -> large safety stock -> more capital frozen -> higher holding cost.

So forecast_error_std is the *direct financial link* between forecasting
accuracy (Week 2) and inventory cost (Week 3). Every downstream safety-stock,
ROP and policy-cost number flows from this single statistic.
"""

import os
import sqlite3
import numpy as np
import pandas as pd

import config as cfg


def main():
    print("=" * 72)
    print("TASK 2 — Forecast error statistics per item-store")
    print(f"DATA SCOPE: {cfg.DATA_SCOPE}")
    print("=" * 72)

    # ---------------------------------------------------------------- Step 1
    # LightGBM forecasts (the workhorse model for X/Y class items).
    lg = pd.read_csv(os.path.join(cfg.OUT, "lgbm_forecasts.csv"))
    # standardise the actual column name -> 'actual'
    if "actual_units" in lg.columns and "actual" not in lg.columns:
        lg = lg.rename(columns={"actual_units": "actual"})
    lg = lg[["item_id", "store_id", "date", "actual", "lgbm_forecast"]].copy()
    print(f"\n[Step 1] lgbm_forecasts.csv loaded: {len(lg):,} rows, "
          f"{lg.groupby(['item_id', 'store_id']).ngroups:,} item-stores")

    # ---------------------------------------------------------------- Step 2
    # Croston SBA forecasts (best for intermittent Z-class items).
    cr = pd.read_csv(os.path.join(cfg.OUT, "croston_forecasts.csv"))
    cr = cr[["item_id", "store_id", "date", "croston_sba_forecast"]].copy()
    print(f"[Step 2] croston_forecasts.csv loaded: {len(cr):,} rows, "
          f"{cr.groupby(['item_id', 'store_id']).ngroups:,} item-stores")

    # ---------------------------------------------------------------- Step 7 (early)
    # We need xyz_class to pick the best model, so load the classification now.
    abc = pd.read_csv(os.path.join(cfg.OUT, "abc_xyz_classification.csv"))
    abc = abc[["item_id", "store_id", "abc_class", "xyz_class", "abc_xyz"]].copy()

    # ---------------------------------------------------------------- Step 3
    # Unified per-row forecast frame.
    # Merge croston onto the lgbm rows (lgbm has the full 28-day x item-store grid).
    df = lg.merge(cr, on=["item_id", "store_id", "date"], how="left")
    df = df.merge(abc, on=["item_id", "store_id"], how="left")

    # Rule: xyz_class == 'Z' -> use croston_sba_forecast; otherwise lgbm_forecast.
    # If a Z item has no croston value (some very sparse series), fall back to lgbm.
    is_z = (df["xyz_class"] == "Z")
    has_croston = df["croston_sba_forecast"].notna()
    use_croston = is_z & has_croston

    df["best_forecast"] = np.where(use_croston,
                                   df["croston_sba_forecast"],
                                   df["lgbm_forecast"])
    df["best_model_used"] = np.where(use_croston, "Croston_SBA", "LightGBM")
    print(f"[Step 3] unified frame: {len(df):,} rows  "
          f"(Croston rows={int(use_croston.sum()):,}, "
          f"LightGBM rows={int((~use_croston).sum()):,})")

    # ---------------------------------------------------------------- Step 4
    # Residual per row. Positive = under-forecast (stockout risk),
    # negative = over-forecast (overstock risk).
    df["actual"] = df["actual"].astype("float64")
    df["best_forecast"] = df["best_forecast"].astype("float64")
    df["residual"] = df["actual"] - df["best_forecast"]

    # ---------------------------------------------------------------- Step 5
    # Per item-store error statistics across the 28 validation days.
    def stats(g):
        actual = g["actual"].to_numpy()
        fc = g["best_forecast"].to_numpy()
        resid = g["residual"].to_numpy()
        a_mean = actual.mean()
        return pd.Series({
            "forecast_mean": fc.mean(),                         # avg predicted daily demand
            "actual_mean": a_mean,                              # true avg daily demand
            "forecast_bias": resid.mean(),                      # +ve = systematic under-forecast
            "forecast_error_std": resid.std(ddof=1) if len(resid) > 1 else 0.0,  # DRIVES safety stock
            "forecast_rmse": np.sqrt(np.mean(resid ** 2)),
            "cv_actual": (actual.std(ddof=1) / a_mean) if a_mean > 0 else 0.0,
            "best_model_used": g["best_model_used"].iloc[0],
        })

    grp = df.groupby(["item_id", "store_id"], observed=True)
    n_groups = grp.ngroups
    print(f"[Step 5] computing residual stats for {n_groups:,} item-stores "
          f"(loop > 100 items -> progress below)...")
    stat = grp.apply(stats, include_groups=False).reset_index()
    # NaN std (single-row groups) -> 0
    stat["forecast_error_std"] = stat["forecast_error_std"].fillna(0.0)
    print(f"         done: {len(stat):,} item-store stat rows")

    # ---------------------------------------------------------------- Step 6
    # Median sell_price per item-store from fact_prices; fall back to category median.
    print("[Step 6] pulling sell_price (median per item-store) from SQLite...")
    con = sqlite3.connect(cfg.DB_PATH)
    stores = "','".join(sorted(stat["store_id"].unique()))
    price = pd.read_sql_query(
        f"SELECT item_id, store_id, sell_price FROM fact_prices "
        f"WHERE store_id IN ('{stores}')", con)
    con.close()
    med_price = (price.groupby(["item_id", "store_id"])["sell_price"]
                 .median().reset_index().rename(columns={"sell_price": "sell_price"}))
    stat = stat.merge(med_price, on=["item_id", "store_id"], how="left")

    # ---------------------------------------------------------------- Step 7
    # Merge cat_id / dept_id / state_id. abc_xyz_classification.csv does NOT carry
    # these, so derive from the dim tables (authoritative) and keep abc/xyz/abc_xyz.
    con = sqlite3.connect(cfg.DB_PATH)
    dim_item = pd.read_sql_query("SELECT item_id, dept_id, cat_id FROM dim_item", con)
    dim_store = pd.read_sql_query("SELECT store_id, state_id FROM dim_store", con)
    con.close()
    stat = stat.merge(abc, on=["item_id", "store_id"], how="left")
    stat = stat.merge(dim_item, on="item_id", how="left")
    stat = stat.merge(dim_store, on="store_id", how="left")

    # category-median price fill for missing/zero prices (never divide by zero later)
    cat_med = stat.groupby("cat_id")["sell_price"].transform("median")
    global_med = stat["sell_price"].median()
    bad_price = stat["sell_price"].isna() | (stat["sell_price"] <= 0)
    n_bad = int(bad_price.sum())
    stat.loc[bad_price, "sell_price"] = cat_med[bad_price]
    stat["sell_price"] = stat["sell_price"].fillna(global_med).clip(lower=0.01)
    print(f"         sell_price merged; {n_bad} item-stores fell back to category median")

    # ---------------------------------------------------------------- Step 8
    cols = ["item_id", "store_id", "cat_id", "dept_id", "state_id",
            "abc_class", "xyz_class", "abc_xyz",
            "forecast_mean", "actual_mean", "forecast_bias",
            "forecast_error_std", "forecast_rmse", "cv_actual",
            "sell_price", "best_model_used"]
    stat = stat[cols]
    out_path = os.path.join(cfg.OUT, "item_forecast_stats.csv")
    stat.to_csv(out_path, index=False)
    print(f"[Step 8] saved -> {out_path}")

    # ---------------------------------------------------------------- prints
    print("\n" + "-" * 72)
    print("ANALYSIS")
    print("-" * 72)
    print(f"Total item-store combinations : {len(stat):,}")
    mc = stat["best_model_used"].value_counts()
    print(f"Best model used               : "
          f"LightGBM={mc.get('LightGBM', 0):,}, Croston_SBA={mc.get('Croston_SBA', 0):,}")

    q = stat["forecast_error_std"].quantile([0, .25, .5, .75, 1.0])
    print("\nforecast_error_std distribution (drives safety stock):")
    print(f"  min   = {q[0.0]:.4f}")
    print(f"  25th  = {q[0.25]:.4f}")
    print(f"  median= {q[0.5]:.4f}")
    print(f"  75th  = {q[0.75]:.4f}")
    print(f"  max   = {q[1.0]:.4f}")

    n_under = int((stat["forecast_bias"] > 0).sum())
    print(f"\nItems with forecast_bias > 0 (systematic under-forecasting): "
          f"{n_under:,} ({100*n_under/len(stat):.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
