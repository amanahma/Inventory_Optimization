"""
croston_model.py  --  M5 Inventory Optimizer (Week 2, Task 4)

Croston SBA forecasts for intermittent (XYZ Z-class, CV > 1.0) item-stores.

IMPORTANT FIX (done without asking, per the project rules):
  The task spec imports `statsmodels.tsa.exponential_smoothing.croston.Croston`.
  That module does NOT exist in statsmodels 0.14.6 (Croston has never shipped in
  stable statsmodels). Rather than add a dependency, we implement Croston's method
  with the Syntetos-Boylan Approximation (SBA) variant directly -- the standard,
  textbook algorithm. The SBA bias correction multiplies the classic Croston ratio
  by (1 - alpha/2). alpha = 0.1 (the conventional Croston smoothing constant).

Per-item edge cases:
  - < 30 non-zero observations   -> forecast = simple mean of the series
  - algorithm error / degenerate -> forecast = mean of last 28 training values
  - forecast clipped at 0 (no negative demand)

Outputs:
  outputs/croston_forecasts.csv   item_id, store_id, date, actual, croston_sba_forecast
  outputs/croston_metrics.csv     Category | Metric | Croston SBA | Seasonal Naive

NOTE: runs on the CA_1 sample (config.BASE_PARQUET); see config.DATA_SCOPE.
"""

import os
import numpy as np
import pandas as pd

import config as C

ALPHA = 0.1  # conventional Croston smoothing constant


def croston_sba(ts, alpha=ALPHA, horizon=C.FORECAST_HORIZON):
    """Croston's method, SBA variant. Returns a flat horizon-length forecast array."""
    ts = np.asarray(ts, dtype="float64")
    nz = np.flatnonzero(ts > 0)
    if nz.size == 0:
        return np.zeros(horizon)
    if nz.size == 1:
        # single demand event: classic ratio is just that demand over its interval
        z = ts[nz[0]]
        p = float(nz[0] + 1)
        f = (1.0 - alpha / 2.0) * (z / p)
        return np.full(horizon, max(f, 0.0))

    # initialise level (z) and interval (p)
    first = int(nz[0])
    z = float(ts[first])
    p = float(np.mean(np.diff(nz)))  # mean inter-arrival interval
    q = 1                            # periods since last non-zero demand

    for t in range(first + 1, len(ts)):
        if ts[t] > 0:
            z = alpha * ts[t] + (1.0 - alpha) * z
            p = alpha * q + (1.0 - alpha) * p
            q = 1
        else:
            q += 1

    f_classic = z / p if p > 0 else z
    f_sba = (1.0 - alpha / 2.0) * f_classic
    return np.full(horizon, max(f_sba, 0.0))


def main():
    print("=" * 78)
    print(f"DATA SCOPE: {C.DATA_SCOPE}")
    print("TASK 4 - Croston SBA for intermittent (Z-class) SKUs")
    print("NOTE: statsmodels has no Croston module; SBA implemented directly.")
    print("=" * 78)

    # ---- Step 1: identify Z-class item-stores -----------------------------
    abc = pd.read_csv(C.ABC_XYZ_CSV)
    z = abc[abc["xyz_class"] == "Z"][["item_id", "store_id"]].drop_duplicates()
    print(f"STEP 1: Z-class (CV>1.0) item-store combinations: {len(z):,}")

    # restrict to combos present in the CA_1 sample scope
    train = pd.read_parquet(C.TRAIN_PARQUET,
                            columns=["item_id", "store_id", "cat_id", "date", "units_sold"])
    val = pd.read_parquet(C.VAL_PARQUET,
                          columns=["item_id", "store_id", "cat_id", "date", "units_sold"])
    for d in (train, val):
        for c in ["item_id", "store_id", "cat_id"]:
            d[c] = d[c].astype(str)
    train["date"] = pd.to_datetime(train["date"])
    val["date"] = pd.to_datetime(val["date"])

    z["item_id"] = z["item_id"].astype(str)
    z["store_id"] = z["store_id"].astype(str)
    z_keys = set(map(tuple, z.values))
    present = set(map(tuple, train[["item_id", "store_id"]].drop_duplicates().values))
    z_keys = sorted(z_keys & present)
    print(f"        Z-class combos present in this data scope: {len(z_keys):,}")

    # ---- Step 2: fit Croston SBA per Z-class series -----------------------
    ztrain = train[train.set_index(["item_id", "store_id"]).index.isin(z_keys)]
    grouped = dict(tuple(ztrain.sort_values(["item_id", "store_id", "date"])
                         .groupby(["item_id", "store_id"], observed=True)))

    val_dates = sorted(val["date"].unique())
    forecasts = {}
    n = len(z_keys)
    mean_fallback = 0
    last28_fallback = 0
    for i, key in enumerate(z_keys, 1):
        if i % 200 == 0 or i == n:
            print(f"  Croston progress: {i}/{n}")
        sub = grouped.get(key)
        if sub is None:
            forecasts[key] = np.zeros(C.FORECAST_HORIZON)
            continue
        series = sub["units_sold"].astype("float64").values
        nz = int((series > 0).sum())
        if nz < 30:
            forecasts[key] = np.full(C.FORECAST_HORIZON, max(float(series.mean()), 0.0))
            mean_fallback += 1
            continue
        try:
            fc = croston_sba(series)
            if not np.all(np.isfinite(fc)):
                raise ValueError("non-finite")
            forecasts[key] = fc
        except Exception:
            last28_fallback += 1
            forecasts[key] = np.full(C.FORECAST_HORIZON,
                                     max(float(series[-28:].mean()), 0.0))
    print(f"  fits done. <30 non-zero -> simple mean: {mean_fallback}; "
          f"errors -> last-28 mean: {last28_fallback}")

    # ---- Step 3: build forecast dataframe ---------------------------------
    rows = []
    for key, arr in forecasts.items():
        it, st = key
        for h, d in enumerate(val_dates):
            rows.append((it, st, pd.Timestamp(d), float(arr[h])))
    fc_df = pd.DataFrame(rows, columns=["item_id", "store_id", "date", "croston_sba_forecast"])

    actual = val[["item_id", "store_id", "date", "cat_id", "units_sold"]].rename(
        columns={"units_sold": "actual"})
    fc_df = fc_df.merge(actual, on=["item_id", "store_id", "date"], how="left")
    fc_df["croston_sba_forecast"] = fc_df["croston_sba_forecast"].clip(lower=0)

    out_csv = os.path.join(C.OUT, "croston_forecasts.csv")
    fc_df[["item_id", "store_id", "date", "actual",
           "croston_sba_forecast"]].to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}  ({len(fc_df):,} rows)")

    # ---- Step 4: metrics vs Seasonal Naive on the SAME Z-class items ------
    base = pd.read_csv(os.path.join(C.OUT, "baseline_forecasts.csv"),
                       parse_dates=["date"])
    for c in ["item_id", "store_id"]:
        base[c] = base[c].astype(str)
    comp = fc_df.merge(
        base[["item_id", "store_id", "date", "seasonal_naive_forecast"]],
        on=["item_id", "store_id", "date"], how="left")

    rows = []
    metric_fns = [("RMSE", C.rmse), ("MAE", C.mae), ("Bias", C.bias)]
    for cat in C.CATEGORIES:
        sub = comp[comp["cat_id"] == cat]
        if len(sub) == 0:
            continue
        for mname, fn in metric_fns:
            rows.append({
                "Category": cat,
                "Metric": mname,
                "Croston SBA": fn(sub["actual"], sub["croston_sba_forecast"]),
                "Seasonal Naive": fn(sub["actual"], sub["seasonal_naive_forecast"]),
            })
    mt = pd.DataFrame(rows)
    out_metrics = os.path.join(C.OUT, "croston_metrics.csv")
    mt.to_csv(out_metrics, index=False)

    print("\n" + "=" * 72)
    print("CROSTON SBA vs SEASONAL NAIVE  (Z-class items only)")
    print("=" * 72)
    print(f"{'Category':<10} | {'Metric':<6} | {'Croston SBA':>12} | {'Seasonal Naive':>14}")
    print("-" * 72)
    for _, r in mt.iterrows():
        print(f"{r['Category']:<10} | {r['Metric']:<6} | "
              f"{r['Croston SBA']:>12.4f} | {r['Seasonal Naive']:>14.4f}")

    # overall RMSE comparison
    cr_rmse = C.rmse(comp["actual"], comp["croston_sba_forecast"])
    sn_rmse = C.rmse(comp["actual"], comp["seasonal_naive_forecast"])
    improve = (sn_rmse - cr_rmse) / sn_rmse * 100.0
    verdict = "better" if improve > 0 else "worse"
    print("-" * 72)
    print(f"Overall Z-class RMSE  ->  Croston SBA: {cr_rmse:.4f} | "
          f"Seasonal Naive: {sn_rmse:.4f}")
    print(f"On Z-class items, Croston SBA is {verdict} than Seasonal Naive "
          f"by {abs(improve):.1f}% (RMSE).")
    print(f"\nsaved {out_metrics}")
    print("DONE.")


if __name__ == "__main__":
    main()
