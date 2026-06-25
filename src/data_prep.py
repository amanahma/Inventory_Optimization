"""
data_prep.py  --  M5 Inventory Optimizer (Week 2, Task 2)

Builds the modelling base table: loads the long sales data, downcasts, merges
ABC-XYZ labels + calendar events/SNAP, derives calendar & price features, then
splits by TIME (never randomly) into train/val parquet files.

NOTE: runs on the CA_1 sample (config.BASE_PARQUET) because the full 58M-row set
OOMs on this ~7.3 GB machine. See config.DATA_SCOPE.
"""

import os
import numpy as np
import pandas as pd

import config as C


def mem_mb(df):
    return df.memory_usage(deep=True).sum() / 1024 / 1024


def main():
    print("=" * 78)
    print(f"DATA SCOPE: {C.DATA_SCOPE}  ({os.path.basename(C.BASE_PARQUET)})")
    print("=" * 78)

    # ---- Step 1: load + downcast ------------------------------------------
    cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id",
            "date", "wm_yr_wk", "units_sold", "sell_price"]
    df = pd.read_parquet(C.BASE_PARQUET, columns=cols)
    print("STEP 1: loaded base parquet")
    print(f"  shape: {df.shape}")
    print(f"  memory BEFORE downcast: {mem_mb(df):,.1f} MB")

    df["units_sold"] = df["units_sold"].astype("int16")
    for c in df.select_dtypes(include=["float64", "float32"]).columns:
        df[c] = df[c].astype("float32")
    for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        df[c] = df[c].astype("category")
    df["date"] = pd.to_datetime(df["date"])
    print(f"  memory AFTER  downcast: {mem_mb(df):,.1f} MB")

    # ---- Step 2: merge ABC-XYZ labels -------------------------------------
    abc = pd.read_csv(C.ABC_XYZ_CSV,
                      usecols=["item_id", "store_id", "abc_class", "xyz_class", "abc_xyz"])
    df = df.merge(abc, on=["item_id", "store_id"], how="left")
    print("STEP 2: merged ABC-XYZ labels  "
          f"(non-null abc_class: {df['abc_class'].notna().mean()*100:.1f}%)")

    # ---- Step 3: merge calendar events + SNAP -----------------------------
    cal = pd.read_csv(C.CALENDAR_CSV, parse_dates=["date"])
    cal_cols = ["date", "event_name_1", "event_type_1", "event_name_2",
                "event_type_2", "snap_CA", "snap_TX", "snap_WI"]
    df = df.merge(cal[cal_cols], on="date", how="left")
    for c in ["snap_CA", "snap_TX", "snap_WI"]:
        df[c] = df[c].fillna(0).astype("int8")
    print("STEP 3: merged calendar event + SNAP columns")

    # ---- Step 4: derived features -----------------------------------------
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")          # Mon=0..Sun=6
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["year"] = df["date"].dt.year.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_event"] = df["event_name_1"].notna().astype("int8")

    # is_snap = the SNAP flag for the store's own state
    state = df["state_id"].astype(str)
    snap_self = np.where(state == "CA", df["snap_CA"],
                 np.where(state == "TX", df["snap_TX"], df["snap_WI"]))
    df["is_snap"] = pd.Series(snap_self, index=df.index).astype("int8")

    # price_change = sell_price minus previous week's price for that item-store.
    # Computed at the weekly grain (price is constant within wm_yr_wk) then mapped
    # back to daily rows, so it is a true week-over-week change, not a daily diff.
    wk = (df[["item_id", "store_id", "wm_yr_wk", "sell_price"]]
          .drop_duplicates(["item_id", "store_id", "wm_yr_wk"])
          .sort_values(["item_id", "store_id", "wm_yr_wk"]))
    wk["price_change"] = (wk.groupby(["item_id", "store_id"], observed=True)["sell_price"]
                            .diff())
    df = df.merge(wk[["item_id", "store_id", "wm_yr_wk", "price_change"]],
                  on=["item_id", "store_id", "wm_yr_wk"], how="left")
    df["price_change"] = df["price_change"].fillna(0).astype("float32")
    print("STEP 4: derived day_of_week, week_of_year, month, year, "
          "is_weekend, is_event, is_snap, price_change")

    # ---- Step 5: TIME-based split -----------------------------------------
    cutoff, _ = C.get_cutoff_date()
    train = df[df["date"] <= cutoff].copy()
    val = df[df["date"] > cutoff].copy()
    print("STEP 5: time-based split  (cutoff =", cutoff.date(), ")")
    print(f"  train shape: {train.shape}  (date <= {cutoff.date()})")
    print(f"  val   shape: {val.shape}  (date >  {cutoff.date()})")
    print(f"  val date range: {val['date'].min().date()} .. {val['date'].max().date()}")

    # ---- Step 6: save ------------------------------------------------------
    train.to_parquet(C.TRAIN_PARQUET, index=False)
    val.to_parquet(C.VAL_PARQUET, index=False)
    print("STEP 6: saved")
    print(f"  {C.TRAIN_PARQUET}  ({os.path.getsize(C.TRAIN_PARQUET)/1024/1024:.1f} MB)")
    print(f"  {C.VAL_PARQUET}  ({os.path.getsize(C.VAL_PARQUET)/1024/1024:.1f} MB)")
    print("DONE.")


if __name__ == "__main__":
    main()
