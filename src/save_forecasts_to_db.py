"""
save_forecasts_to_db.py  --  M5 Inventory Optimizer (Week 2, Task 7)

Merges every model's validation forecasts into one fact_forecasts table in the
SQLite warehouse, picks the best model per SKU (Croston SBA for intermittent
Z-class items, LightGBM otherwise), records the error, and runs verification
queries.

NOTE: forecasts were produced on the CA_1 sample (see config.DATA_SCOPE), so
fact_forecasts covers CA_1 item-stores over the 28-day validation horizon.
"""

import os
import sqlite3
import numpy as np
import pandas as pd

import config as C

DDL = """
CREATE TABLE fact_forecasts (
    item_id                  TEXT,
    store_id                 TEXT,
    date                     TEXT,
    actual_units             INTEGER,
    seasonal_naive_forecast  REAL,
    ets_forecast             REAL,
    croston_forecast         REAL,
    lgbm_forecast            REAL,
    best_forecast            REAL,
    forecast_error           REAL,
    abs_error                REAL,
    abc_class                TEXT,
    xyz_class                TEXT,
    abc_xyz                  TEXT
);
"""


def main():
    print("=" * 78)
    print(f"DATA SCOPE: {C.DATA_SCOPE}")
    print("TASK 7 - Save forecasts to SQLite (fact_forecasts)")
    print("=" * 78)

    # ---- Step 2: load + merge all forecast CSVs ---------------------------
    base = pd.read_csv(os.path.join(C.OUT, "baseline_forecasts.csv"), parse_dates=["date"])
    cros = pd.read_csv(os.path.join(C.OUT, "croston_forecasts.csv"), parse_dates=["date"])
    lgbm = pd.read_csv(os.path.join(C.OUT, "lgbm_forecasts.csv"), parse_dates=["date"])
    for d in (base, cros, lgbm):
        for c in ["item_id", "store_id"]:
            d[c] = d[c].astype(str)

    df = base[["item_id", "store_id", "date", "actual",
               "seasonal_naive_forecast", "ets_forecast"]].copy()
    df = df.merge(cros[["item_id", "store_id", "date", "croston_sba_forecast"]],
                  on=["item_id", "store_id", "date"], how="left")
    df = df.merge(lgbm[["item_id", "store_id", "date", "lgbm_forecast"]],
                  on=["item_id", "store_id", "date"], how="left")
    df = df.rename(columns={"actual": "actual_units",
                            "croston_sba_forecast": "croston_forecast"})
    print(f"STEP 2: merged forecasts -> {len(df):,} rows")

    # ---- Step 4: attach abc/xyz labels ------------------------------------
    abc = pd.read_csv(C.ABC_XYZ_CSV,
                      usecols=["item_id", "store_id", "abc_class", "xyz_class", "abc_xyz"])
    abc["item_id"] = abc["item_id"].astype(str)
    abc["store_id"] = abc["store_id"].astype(str)
    df = df.merge(abc, on=["item_id", "store_id"], how="left")

    # ---- Step 3: best_forecast (Croston for Z, LightGBM otherwise) --------
    is_z = df["xyz_class"].eq("Z")
    df["best_forecast"] = np.where(is_z & df["croston_forecast"].notna(),
                                   df["croston_forecast"], df["lgbm_forecast"])
    # safety: if a best is still missing, fall back to seasonal naive
    df["best_forecast"] = df["best_forecast"].fillna(df["seasonal_naive_forecast"])
    df["forecast_error"] = df["best_forecast"] - df["actual_units"]
    df["abs_error"] = df["forecast_error"].abs()
    print(f"STEP 3/4: best_forecast set (Z-class via Croston: {int((is_z & df['croston_forecast'].notna()).sum()):,} rows; "
          f"others via LightGBM), abc/xyz merged")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["actual_units"] = df["actual_units"].astype("int64")
    col_order = ["item_id", "store_id", "date", "actual_units",
                 "seasonal_naive_forecast", "ets_forecast", "croston_forecast",
                 "lgbm_forecast", "best_forecast", "forecast_error", "abs_error",
                 "abc_class", "xyz_class", "abc_xyz"]
    df = df[col_order]

    # ---- Step 1 + 5: (re)create table and insert --------------------------
    con = sqlite3.connect(C.DB_PATH)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS fact_forecasts")
    cur.execute(DDL)
    con.commit()
    df.to_sql("fact_forecasts", con, if_exists="append", index=False)
    con.commit()
    total = cur.execute("SELECT COUNT(*) FROM fact_forecasts").fetchone()[0]
    print(f"STEP 5: inserted {total:,} rows into fact_forecasts")

    # ---- Step 6: verification queries -------------------------------------
    print("\n" + "=" * 60)
    print("QUERY 1: mean absolute error by category (join dim_item)")
    print("=" * 60)
    q1 = """
        SELECT di.cat_id AS category, AVG(f.abs_error) AS mean_abs_error,
               COUNT(*) AS n
        FROM fact_forecasts f
        JOIN dim_item di ON f.item_id = di.item_id
        GROUP BY di.cat_id ORDER BY mean_abs_error DESC
    """
    for r in cur.execute(q1).fetchall():
        print(f"  {r[0]:<12} mean_abs_error={r[1]:.4f}  (n={r[2]:,})")

    print("\n" + "=" * 60)
    print("QUERY 2: row count by abc_xyz class")
    print("=" * 60)
    q2 = "SELECT abc_xyz, COUNT(*) AS count FROM fact_forecasts GROUP BY abc_xyz ORDER BY count DESC"
    for r in cur.execute(q2).fetchall():
        print(f"  {str(r[0]):<6} {r[1]:,}")

    print("\n" + "=" * 60)
    print("QUERY 3: top 10 item-stores by avg abs forecast error")
    print("=" * 60)
    q3 = """
        SELECT item_id, store_id, AVG(abs_error) AS avg_error
        FROM fact_forecasts GROUP BY item_id, store_id
        ORDER BY avg_error DESC LIMIT 10
    """
    for r in cur.execute(q3).fetchall():
        print(f"  {r[0]:<18} {r[1]:<6} avg_error={r[2]:.4f}")

    con.close()
    print("\nDONE.")


if __name__ == "__main__":
    main()
