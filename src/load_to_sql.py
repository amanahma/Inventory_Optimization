"""
load_to_sql.py  --  M5 Inventory Optimizer (Week 1, Task 4)

Builds a SQLite star-schema database (data/m5_database.db) from the raw CSVs
and the processed Parquet, loading large fact tables in chunks of 500,000.

Indexes (defined in schema.sql) are created AFTER the bulk load -- maintaining
an index during 58M inserts is dramatically slower than building it once at the
end, so the loader runs CREATE TABLE statements first, loads, then CREATE INDEX.
"""

import os
import re
import time
import sqlite3
import pandas as pd
import pyarrow.parquet as pq

PROJECT = r"C:\Users\AMAN AHMAD\Documents\m5-inventory-optimizer"
RAW = os.path.join(PROJECT, "data", "raw")
PROC = os.path.join(PROJECT, "data", "processed")

DB_PATH = os.path.join(PROJECT, "data", "m5_database.db")
SCHEMA_SQL = os.path.join(PROJECT, "sql", "schema.sql")
CAL_CSV = os.path.join(RAW, "calendar.csv")
SALES_CSV = os.path.join(RAW, "sales_train_validation.csv")
PRICE_CSV = os.path.join(RAW, "sell_prices.csv")
LONG_PARQUET = os.path.join(PROC, "sales_long.parquet")

CHUNK = 500_000


def split_schema(path):
    """Return (table_statements, index_statements) from schema.sql."""
    sql = open(path, "r", encoding="utf-8").read()
    # strip line comments
    sql = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    stmts = [s.strip() for s in sql.split(";") if s.strip()]
    tables = [s for s in stmts if not s.upper().startswith("CREATE INDEX")]
    indexes = [s for s in stmts if s.upper().startswith("CREATE INDEX")]
    return tables, indexes


def main():
    t0 = time.time()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # fresh build

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    # speed pragmas for bulk loading
    cur.execute("PRAGMA journal_mode = OFF")
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA temp_store = MEMORY")
    cur.execute("PRAGMA cache_size = -200000")  # ~200MB page cache

    tables, indexes = split_schema(SCHEMA_SQL)
    for stmt in tables:
        cur.execute(stmt)
    con.commit()
    print(f"Created {len(tables)} tables from schema.sql")

    # ---- dim_date ----------------------------------------------------------
    cal = pd.read_csv(CAL_CSV)
    dt = pd.to_datetime(cal["date"])
    dim_date = pd.DataFrame({
        "date": cal["date"].astype(str),
        "d": cal["d"],
        "wm_yr_wk": cal["wm_yr_wk"].astype(int),
        "day_of_week": dt.dt.dayofweek.astype(int),          # Mon=0 .. Sun=6
        "week_of_year": dt.dt.isocalendar().week.astype(int),
        "month": cal["month"].astype(int),
        "year": cal["year"].astype(int),
        "event_name_1": cal["event_name_1"],
        "event_type_1": cal["event_type_1"],
        "event_name_2": cal["event_name_2"],
        "event_type_2": cal["event_type_2"],
        "snap_CA": cal["snap_CA"].astype(int),
        "snap_TX": cal["snap_TX"].astype(int),
        "snap_WI": cal["snap_WI"].astype(int),
        "is_weekend": dt.dt.dayofweek.isin([5, 6]).astype(int),
    })
    cur.executemany(
        "INSERT INTO dim_date VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        list(dim_date.itertuples(index=False, name=None)),
    )
    con.commit()
    print(f"Loaded dim_date: {cur.execute('SELECT COUNT(*) FROM dim_date').fetchone()[0]:,} rows")

    # ---- dim_item / dim_store (from unique sales rows) ---------------------
    meta = pd.read_csv(SALES_CSV, usecols=["item_id", "dept_id", "cat_id", "store_id", "state_id"])
    dim_item = meta[["item_id", "dept_id", "cat_id"]].drop_duplicates("item_id")
    cur.executemany("INSERT INTO dim_item VALUES (?,?,?)",
                    list(dim_item.itertuples(index=False, name=None)))
    con.commit()
    print(f"Loaded dim_item: {cur.execute('SELECT COUNT(*) FROM dim_item').fetchone()[0]:,} rows")

    dim_store = meta[["store_id", "state_id"]].drop_duplicates("store_id")
    cur.executemany("INSERT INTO dim_store VALUES (?,?)",
                    list(dim_store.itertuples(index=False, name=None)))
    con.commit()
    print(f"Loaded dim_store: {cur.execute('SELECT COUNT(*) FROM dim_store').fetchone()[0]:,} rows")

    # ---- fact_sales (stream parquet in 500k-row batches) -------------------
    pf = pq.ParquetFile(LONG_PARQUET)
    total = 0
    cols = ["item_id", "store_id", "date", "units_sold"]
    for batch in pf.iter_batches(batch_size=CHUNK, columns=cols):
        df = batch.to_pandas()
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df["units_sold"] = df["units_sold"].astype(int)
        cur.executemany("INSERT INTO fact_sales VALUES (?,?,?,?)",
                        list(df.itertuples(index=False, name=None)))
        total += len(df)
        if total % 5_000_000 < CHUNK:
            con.commit()
            print(f"  fact_sales ... {total:,} rows", flush=True)
    con.commit()
    print(f"Loaded fact_sales: {cur.execute('SELECT COUNT(*) FROM fact_sales').fetchone()[0]:,} rows")

    # ---- fact_prices (stream csv in 500k-row chunks) -----------------------
    total = 0
    for chunk in pd.read_csv(PRICE_CSV, chunksize=CHUNK):
        rows = chunk[["item_id", "store_id", "wm_yr_wk", "sell_price"]]
        cur.executemany("INSERT INTO fact_prices VALUES (?,?,?,?)",
                        list(rows.itertuples(index=False, name=None)))
        total += len(rows)
    con.commit()
    print(f"Loaded fact_prices: {cur.execute('SELECT COUNT(*) FROM fact_prices').fetchone()[0]:,} rows")

    # ---- indexes (after bulk load) -----------------------------------------
    print("Building indexes ...")
    for stmt in indexes:
        cur.execute(stmt)
    con.commit()
    con.close()
    print(f"DB written to {DB_PATH}  ({os.path.getsize(DB_PATH)/1024/1024:.1f} MB)")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
