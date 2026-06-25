"""
data_loader.py  --  M5 Inventory Optimizer (Week 1, Task 3)

Loads the raw M5 files, reshapes daily sales from wide to long format, enriches
with calendar + price information, and writes Parquet outputs.

Memory note: this machine has ~7 GB RAM, so a 58M-row long dataframe will NOT
fit in memory all at once. The wide->long melt is therefore done in row-batches
and streamed straight to Parquet via pyarrow.ParquetWriter (fixed schema), which
keeps peak memory to a few hundred MB.
"""

import os
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------
PROJECT = r"C:\Users\AMAN AHMAD\Documents\m5-inventory-optimizer"
RAW = os.path.join(PROJECT, "data", "raw")
PROC = os.path.join(PROJECT, "data", "processed")
os.makedirs(PROC, exist_ok=True)

SALES_CSV = os.path.join(RAW, "sales_train_validation.csv")
CAL_CSV = os.path.join(RAW, "calendar.csv")
PRICE_CSV = os.path.join(RAW, "sell_prices.csv")

LONG_PARQUET = os.path.join(PROC, "sales_long.parquet")
CA1_PARQUET = os.path.join(PROC, "sales_CA1_sample.parquet")

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
BATCH_WIDE_ROWS = 1000  # wide rows per batch -> ~1.9M long rows per batch

# Fixed output schema so every batch's Arrow table is identical and appendable.
OUT_SCHEMA = pa.schema([
    ("item_id", pa.string()),
    ("dept_id", pa.string()),
    ("cat_id", pa.string()),
    ("store_id", pa.string()),
    ("state_id", pa.string()),
    ("d", pa.string()),
    ("date", pa.timestamp("us")),
    ("wm_yr_wk", pa.int32()),
    ("weekday", pa.string()),
    ("wday", pa.int8()),
    ("month", pa.int8()),
    ("year", pa.int16()),
    ("event_name_1", pa.string()),
    ("event_type_1", pa.string()),
    ("event_name_2", pa.string()),
    ("event_type_2", pa.string()),
    ("snap_CA", pa.int8()),
    ("snap_TX", pa.int8()),
    ("snap_WI", pa.int8()),
    ("units_sold", pa.int16()),
    ("sell_price", pa.float32()),
])


def main():
    t0 = time.time()

    # ----- Step 1: sales (wide) ---------------------------------------------
    print("=" * 70)
    print("STEP 1: Load sales_train_validation.csv")
    header = pd.read_csv(SALES_CSV, nrows=0).columns.tolist()
    d_cols = [c for c in header if c.startswith("d_")]
    dtypes = {c: "int16" for c in d_cols}
    for c in ID_COLS + ["id"]:
        dtypes[c] = "string"
    sales = pd.read_csv(SALES_CSV, dtype=dtypes)
    print("sales shape:", sales.shape)
    print("first 3 rows (id + first few day columns):")
    print(sales[["id"] + ID_COLS + d_cols[:3]].head(3).to_string())

    # ----- Step 2: calendar --------------------------------------------------
    print("=" * 70)
    print("STEP 2: Load calendar.csv")
    cal = pd.read_csv(CAL_CSV)
    cal["date"] = pd.to_datetime(cal["date"])
    print("calendar shape:", cal.shape)
    print("calendar columns:", cal.columns.tolist())
    cal_keep = cal[[
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]].copy()

    # ----- Step 3: prices ----------------------------------------------------
    print("=" * 70)
    print("STEP 3: Load sell_prices.csv")
    prices = pd.read_csv(
        PRICE_CSV,
        dtype={"store_id": "category", "item_id": "category",
               "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    print("prices shape:", prices.shape)
    # MultiIndex Series for fast, memory-light price lookup per batch.
    price_ser = prices.set_index(["item_id", "store_id", "wm_yr_wk"])["sell_price"]
    price_ser = price_ser[~price_ser.index.duplicated(keep="last")].sort_index()
    del prices

    # ----- Step 4: wide -> long (batched) + Step 5/6 writers ----------------
    print("=" * 70)
    print("STEP 4: Melt wide -> long, merge calendar + prices (batched)")
    long_writer = pq.ParquetWriter(LONG_PARQUET, OUT_SCHEMA, compression="snappy")
    ca1_writer = pq.ParquetWriter(CA1_PARQUET, OUT_SCHEMA, compression="snappy")

    total_rows = 0
    ca1_rows = 0
    n = len(sales)
    for start in range(0, n, BATCH_WIDE_ROWS):
        chunk = sales.iloc[start:start + BATCH_WIDE_ROWS]
        long = chunk.melt(
            id_vars=ID_COLS, value_vars=d_cols,
            var_name="d", value_name="units_sold",
        )
        long["units_sold"] = long["units_sold"].astype("int16")
        # calendar merge (every d exists -> no nulls introduced)
        long = long.merge(cal_keep, on="d", how="left")
        # price lookup via reindex on MultiIndex
        idx = pd.MultiIndex.from_arrays(
            [long["item_id"].astype("object"),
             long["store_id"].astype("object"),
             long["wm_yr_wk"].to_numpy()]
        )
        long["sell_price"] = price_ser.reindex(idx).to_numpy(dtype="float32")

        long = long[[f.name for f in OUT_SCHEMA]]
        tbl = pa.Table.from_pandas(long, schema=OUT_SCHEMA, preserve_index=False)
        long_writer.write_table(tbl)
        total_rows += tbl.num_rows

        ca1 = long[long["store_id"] == "CA_1"]
        if len(ca1):
            ca1_tbl = pa.Table.from_pandas(ca1, schema=OUT_SCHEMA, preserve_index=False)
            ca1_writer.write_table(ca1_tbl)
            ca1_rows += ca1_tbl.num_rows

        done = min(start + BATCH_WIDE_ROWS, n)
        print(f"  batch {done:>6}/{n} wide rows  ->  {total_rows:>12,} long rows", flush=True)

    long_writer.close()
    ca1_writer.close()

    # ----- report on final long dataset (read metadata, not full load) ------
    pf = pq.ParquetFile(LONG_PARQUET)
    print("=" * 70)
    print("FINAL long-format dataset:")
    print("shape:", (pf.metadata.num_rows, pf.metadata.num_columns))
    print("columns:", pf.schema_arrow.names)
    sample = pa.Table.from_batches([next(pf.iter_batches(batch_size=5))]).to_pandas()
    print("sample 5 rows:")
    print(sample.to_string())

    print("=" * 70)
    print(f"STEP 5: wrote {LONG_PARQUET}  ({total_rows:,} rows)")
    print(f"STEP 6: wrote {CA1_PARQUET}  (store CA_1, {ca1_rows:,} rows)")
    for p in (LONG_PARQUET, CA1_PARQUET):
        print(f"  {os.path.basename(p)}: {os.path.getsize(p) / 1024 / 1024:.1f} MB")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
