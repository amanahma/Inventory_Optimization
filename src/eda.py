"""
eda.py  --  M5 Inventory Optimizer (Week 1, Task 6)

Exploratory analysis + ABC-XYZ classification over data/processed/sales_long.parquet.

Memory note: the long dataset is 58M rows. Pulling every string column into a
single pandas frame would balloon to several GB of Python str objects on a ~7 GB
machine, so this script makes ONE streaming pass with pyarrow.iter_batches and
accumulates only additive aggregates (sums, sum-of-squares, counts, per-value
counts). Every required statistic -- including exact medians (via integer
value-count tables) and the coefficient of variation -- is recoverable from
those accumulators, keeping peak memory to a few hundred MB.
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT = r"C:\Users\AMAN AHMAD\Documents\m5-inventory-optimizer"
PARQUET = os.path.join(PROJECT, "data", "processed", "sales_long.parquet")
OUT = os.path.join(PROJECT, "outputs")
os.makedirs(OUT, exist_ok=True)

BATCH = 2_000_000
CATS = ["FOODS", "HOBBIES", "HOUSEHOLD"]
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COLS = ["item_id", "store_id", "cat_id", "state_id",
        "date", "units_sold", "sell_price", "snap_CA"]


def merge_add(acc, df):
    """Accumulate a grouped aggregate frame into a running dict-of-arrays store."""
    return acc.add(df, fill_value=0) if acc is not None else df


def main():
    t0 = time.time()
    pf = pq.ParquetFile(PARQUET)

    # ---- accumulators ------------------------------------------------------
    cat_valcounts = {c: defaultdict(int) for c in CATS}   # cat -> {units: count}
    zero_is = None        # index (item_id, store_id) -> [total, zeros]
    dow_cat = None        # index (cat_id, dow)        -> [sum_units, n]
    mon_cat = None        # index (cat_id, month)      -> [sum_units, n]
    snap_acc = None       # index (snap_CA)            -> [sum_units, n]  (FOODS & CA)
    abc = None            # index (item_id, store_id)  -> [sum, sumsq, n, revenue]

    total_records = 0
    n_batches = 0
    for rb in pf.iter_batches(batch_size=BATCH, columns=COLS):
        df = rb.to_pandas()
        n = len(df)
        total_records += n
        n_batches += 1

        u = df["units_sold"].astype("int64")
        df["units_sold"] = u

        # 1) per-category value counts (-> mean/median/std/max/zero%)
        for c in CATS:
            vc = u[df["cat_id"] == c].value_counts()
            d = cat_valcounts[c]
            for val, cnt in vc.items():
                d[int(val)] += int(cnt)

        # 2) zero-demand per (item, store)
        g = df.groupby(["item_id", "store_id"], observed=True)["units_sold"]
        zb = pd.DataFrame({"total": g.size(),
                           "zeros": g.apply(lambda s: (s == 0).sum())})
        zero_is = merge_add(zero_is, zb)

        # 3) seasonality: dow & month per category
        dow = df["date"].dt.dayofweek            # Mon=0 .. Sun=6
        mon = df["date"].dt.month
        gd = df.assign(dow=dow).groupby(["cat_id", "dow"], observed=True)["units_sold"]
        dow_cat = merge_add(dow_cat, pd.DataFrame({"sum": gd.sum(), "n": gd.size()}))
        gm = df.assign(month=mon).groupby(["cat_id", "month"], observed=True)["units_sold"]
        mon_cat = merge_add(mon_cat, pd.DataFrame({"sum": gm.sum(), "n": gm.size()}))

        # 4) SNAP impact (FOODS items in CA stores)
        m = (df["cat_id"] == "FOODS") & (df["state_id"] == "CA")
        if m.any():
            gs = df[m].groupby("snap_CA", observed=True)["units_sold"]
            snap_acc = merge_add(snap_acc, pd.DataFrame({"sum": gs.sum(), "n": gs.size()}))

        # 5) ABC-XYZ accumulators per (item, store)
        rev = (u * df["sell_price"].fillna(0).astype("float64"))
        ab = df.assign(_u=u, _u2=u * u, _rev=rev) \
               .groupby(["item_id", "store_id"], observed=True) \
               .agg(sum=("_u", "sum"), sumsq=("_u2", "sum"),
                    n=("_u", "size"), revenue=("_rev", "sum"))
        abc = merge_add(abc, ab)

        print(f"  batch {n_batches:>2}: {total_records:>12,} rows processed", flush=True)

    # ======================================================================
    # STEP 2 -- basic statistics
    # ======================================================================
    print("=" * 78)
    print("STEP 2: BASIC STATISTICS")
    abc = abc  # noqa
    n_items = zero_is.index.get_level_values("item_id").nunique()
    n_stores = zero_is.index.get_level_values("store_id").nunique()
    # date range from parquet metadata sample
    dr0 = pf.read_row_group(0, columns=["date"]).to_pandas()["date"].min()
    drN = pq.read_table(PARQUET, columns=["date"])  # only date col, compact
    date_min, date_max = drN["date"].to_pandas().min(), drN["date"].to_pandas().max()
    del drN

    total_zeros = sum(cnt for c in CATS for v, cnt in cat_valcounts[c].items() if v == 0)
    print(f"  unique items     : {n_items:,}")
    print(f"  unique stores    : {n_stores}")
    print(f"  date range       : {date_min.date()}  ->  {date_max.date()}")
    print(f"  total records    : {total_records:,}")
    print(f"  units_sold == 0  : {100.0 * total_zeros / total_records:.2f}%")

    rows = []
    for c in CATS:
        d = cat_valcounts[c]
        vals = np.array(sorted(d.keys()), dtype="int64")
        cnts = np.array([d[v] for v in vals], dtype="int64")
        N = cnts.sum()
        mean = (vals * cnts).sum() / N
        var = ((vals - mean) ** 2 * cnts).sum() / N
        std = np.sqrt(var)
        # exact median from cumulative counts
        cum = np.cumsum(cnts)
        half = N / 2.0
        med_idx = np.searchsorted(cum, half, side="left")
        if N % 2 == 0 and cum[med_idx] == half and med_idx + 1 < len(vals):
            median = (vals[med_idx] + vals[med_idx + 1]) / 2.0
        else:
            median = float(vals[med_idx])
        mx = int(vals.max())
        zero_pct = 100.0 * d.get(0, 0) / N
        rows.append({"category": c, "count": N, "mean": round(mean, 4),
                     "median": median, "std": round(std, 4), "max": mx,
                     "pct_zero": round(zero_pct, 2)})
        print(f"  {c:9}  mean={mean:7.4f}  median={median:5.1f}  "
              f"std={std:7.4f}  max={mx}")
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(OUT, "basic_stats.csv"), index=False)
    print(f"  -> outputs/basic_stats.csv")

    # ======================================================================
    # STEP 3 -- zero-demand analysis
    # ======================================================================
    print("=" * 78)
    print("STEP 3: ZERO-DEMAND ANALYSIS")
    zero_is = zero_is.copy()
    zero_is["zero_pct"] = 100.0 * zero_is["zeros"] / zero_is["total"]
    pct_over_40 = 100.0 * (zero_is["zero_pct"] > 40).mean()
    print(f"  item-store pairs                 : {len(zero_is):,}")
    print(f"  pairs with > 40% zero-sale days  : {pct_over_40:.2f}%")

    plt.figure(figsize=(9, 5))
    plt.hist(zero_is["zero_pct"], bins=50, color="#4C72B0", edgecolor="white")
    plt.axvline(40, color="red", ls="--", lw=1.5, label="40% threshold")
    plt.title("Distribution of Zero-Sale Day Percentage across Item-Store Pairs")
    plt.xlabel("% of days with zero sales")
    plt.ylabel("Number of item-store pairs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "zero_demand_histogram.png"), dpi=120)
    plt.close()
    print("  -> outputs/zero_demand_histogram.png")

    # ======================================================================
    # STEP 4 -- seasonality
    # ======================================================================
    print("=" * 78)
    print("STEP 4: SEASONALITY")
    dow_avg = (dow_cat["sum"] / dow_cat["n"]).rename("avg").reset_index()
    dow_pivot = dow_avg.pivot(index="dow", columns="cat_id", values="avg").sort_index()
    mon_avg = (mon_cat["sum"] / mon_cat["n"]).rename("avg").reset_index()
    mon_pivot = mon_avg.pivot(index="month", columns="cat_id", values="avg").sort_index()

    ax = dow_pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_xticklabels([DOW_LABELS[int(i)] for i in dow_pivot.index], rotation=0)
    ax.set_title("Average Daily Units Sold by Day of Week (by Category)")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Average units sold")
    ax.legend(title="Category")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "seasonality_day_of_week.png"), dpi=120)
    plt.close()
    print("  -> outputs/seasonality_day_of_week.png")

    ax = mon_pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Average Daily Units Sold by Month (by Category)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average units sold")
    ax.legend(title="Category")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "seasonality_monthly.png"), dpi=120)
    plt.close()
    print("  -> outputs/seasonality_monthly.png")

    # ======================================================================
    # STEP 5 -- SNAP impact (FOODS in CA)
    # ======================================================================
    print("=" * 78)
    print("STEP 5: SNAP IMPACT (FOODS items, CA stores)")
    snap_avg = (snap_acc["sum"] / snap_acc["n"])
    avg_non = float(snap_avg.loc[0])
    avg_snap = float(snap_avg.loc[1])
    uplift = 100.0 * (avg_snap - avg_non) / avg_non
    print(f"  avg daily sales  NON-SNAP days : {avg_non:.4f}")
    print(f"  avg daily sales  SNAP days     : {avg_snap:.4f}")
    print(f"  SNAP uplift                    : {uplift:+.2f}%")

    plt.figure(figsize=(6, 5))
    bars = plt.bar(["Non-SNAP", "SNAP"], [avg_non, avg_snap],
                   color=["#999999", "#55A868"])
    for b, v in zip(bars, [avg_non, avg_snap]):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                 ha="center", va="bottom")
    plt.title(f"FOODS / CA: Avg Daily Sales on SNAP vs Non-SNAP Days "
              f"({uplift:+.1f}%)")
    plt.xlabel("Day type")
    plt.ylabel("Average units sold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "snap_impact.png"), dpi=120)
    plt.close()
    print("  -> outputs/snap_impact.png")

    # ======================================================================
    # STEP 6 -- ABC-XYZ classification
    # ======================================================================
    print("=" * 78)
    print("STEP 6: ABC-XYZ CLASSIFICATION")
    c = abc.reset_index()
    c.columns = ["item_id", "store_id", "sum", "sumsq", "n", "revenue"]
    c["total_revenue"] = c["revenue"]
    mean = c["sum"] / c["n"]
    # population variance is fine at n~1913; guard against negative rounding
    var = (c["sumsq"] / c["n"]) - mean ** 2
    var = var.clip(lower=0)
    std = np.sqrt(var)
    c["cv"] = np.where(mean > 0, std / mean, np.inf)

    # --- ABC by cumulative revenue share ---
    c = c.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    total_rev = c["total_revenue"].sum()
    cum_share = c["total_revenue"].cumsum() / total_rev
    c["abc_class"] = np.where(cum_share <= 0.70, "A",
                       np.where(cum_share <= 0.90, "B", "C"))

    # --- XYZ by coefficient of variation ---
    c["xyz_class"] = np.where(c["cv"] < 0.5, "X",
                       np.where(c["cv"] <= 1.0, "Y", "Z"))
    c["abc_xyz"] = c["abc_class"] + c["xyz_class"]

    out_cols = ["item_id", "store_id", "total_revenue", "cv",
                "abc_class", "xyz_class", "abc_xyz"]
    c[out_cols].to_csv(os.path.join(OUT, "abc_xyz_classification.csv"), index=False)
    print(f"  classified {len(c):,} item-store pairs")
    print(f"  -> outputs/abc_xyz_classification.csv")

    # --- 9-cell count table ---
    cell = pd.crosstab(c["abc_class"], c["xyz_class"]).reindex(
        index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
    print("\n  ABC-XYZ cell counts:")
    print(cell.to_string())
    print(f"  total: {int(cell.values.sum()):,}")

    # --- forecasting method mapping ---
    methods = {
        "LightGBM (stable, high-value)":      ["AX", "BX", "AY", "BY"],
        "ETS baseline (low-value)":           ["CX", "CY"],
        "Croston/SBA (intermittent demand)":  ["AZ", "BZ", "CZ"],
    }
    vc = c["abc_xyz"].value_counts()
    print("\n  Forecasting-method assignment:")
    for method, cells in methods.items():
        cnt = int(sum(vc.get(k, 0) for k in cells))
        print(f"    {', '.join(cells):16} -> {method:36} {cnt:>8,} pairs")

    # --- heatmap ---
    plt.figure(figsize=(7, 6))
    sns.heatmap(cell, annot=True, fmt=",d", cmap="YlOrRd",
                cbar_kws={"label": "item-store pairs"})
    plt.title("ABC-XYZ Classification Matrix\n(rows = revenue class, cols = demand variability)")
    plt.xlabel("XYZ class  (X stable | Y variable | Z intermittent)")
    plt.ylabel("ABC class  (A high | B mid | C low revenue)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "abc_xyz_heatmap.png"), dpi=120)
    plt.close()
    print("\n  -> outputs/abc_xyz_heatmap.png")

    print("=" * 78)
    print(f"EDA complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
