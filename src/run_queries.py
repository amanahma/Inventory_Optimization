"""
run_queries.py  --  M5 Inventory Optimizer (Week 1, Task 5)

Runs each analytical query in sql/queries/ against data/m5_database.db, writes
the result to outputs/ as CSV, and prints the first 5 rows of each.
"""

import os
import time
import sqlite3
import pandas as pd

PROJECT = r"C:\Users\AMAN AHMAD\Documents\m5-inventory-optimizer"
DB = os.path.join(PROJECT, "data", "m5_database.db")
QDIR = os.path.join(PROJECT, "sql", "queries")
OUT = os.path.join(PROJECT, "outputs")
os.makedirs(OUT, exist_ok=True)

# (sql file, output csv)
JOBS = [
    ("01_sales_by_category_month.sql", "sales_by_category_month.csv"),
    ("02_zero_sales_percentage.sql",   "zero_sales_analysis.csv"),
    ("03_top_20_items_revenue.sql",    "top20_items.csv"),
    ("04_snap_impact.sql",             "snap_impact.csv"),
    ("05_price_trend.sql",             "price_trend.csv"),
]


def main():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA cache_size = -400000")
    con.execute("PRAGMA temp_store = MEMORY")

    for sql_file, csv_file in JOBS:
        sql = open(os.path.join(QDIR, sql_file), "r", encoding="utf-8").read()
        print("=" * 78)
        print(f"RUN {sql_file}  ->  outputs/{csv_file}")
        t = time.time()
        df = pd.read_sql_query(sql, con)
        out_path = os.path.join(OUT, csv_file)
        df.to_csv(out_path, index=False)
        print(f"  {len(df):,} rows  in {time.time() - t:.1f}s")
        print(df.head(5).to_string(index=False))

    con.close()
    print("=" * 78)
    print("All 5 queries complete.")


if __name__ == "__main__":
    main()
