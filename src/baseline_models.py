"""
baseline_models.py  --  M5 Inventory Optimizer (Week 2, Task 3)

Builds two baseline forecasters over the 28-day validation horizon:

  BASELINE 1 - Seasonal Naive
      forecast(date) = actual sales 364 days earlier (same weekday, 52 wks ago);
      if that day is missing, fall back to 28 days earlier. Pure lookup, no fit.

  BASELINE 2 - ETS (additive trend + additive weekly seasonality, period 7)
      fit on the LAST 365 training days per item-store, forecast 28 days ahead.
      To stay fast we only fit ETS on high-value A/B items; for C-class items the
      ETS column simply reuses the Seasonal Naive forecast. On any fit failure we
      fall back to the mean of the last 28 training days.

Outputs:
  outputs/baseline_forecasts.csv   item_id, store_id, date, actual,
                                   seasonal_naive_forecast, ets_forecast
  outputs/baseline_metrics.csv     Category | Metric | Seasonal Naive | ETS

NOTE: runs on the CA_1 sample (config.BASE_PARQUET) because the full 58M-row set
OOMs on this ~7.9 GB machine. See config.DATA_SCOPE.
"""

import os
import warnings

import numpy as np
import pandas as pd

import config as C

warnings.filterwarnings("ignore")  # silence statsmodels convergence chatter


def seasonal_naive(train, val):
    """Vectorised seasonal-naive lookup: 364-day-back, fallback 28-day-back."""
    lookup = train[["item_id", "store_id", "date", "units_sold"]]

    v = val[["item_id", "store_id", "date"]].copy()
    v["d364"] = v["date"] - pd.Timedelta(days=364)
    v["d28"] = v["date"] - pd.Timedelta(days=28)

    l364 = lookup.rename(columns={"date": "d364", "units_sold": "f364"})
    l28 = lookup.rename(columns={"date": "d28", "units_sold": "f28"})

    v = v.merge(l364, on=["item_id", "store_id", "d364"], how="left")
    v = v.merge(l28, on=["item_id", "store_id", "d28"], how="left")

    v["seasonal_naive_forecast"] = v["f364"].fillna(v["f28"]).fillna(0.0)
    v["seasonal_naive_forecast"] = v["seasonal_naive_forecast"].clip(lower=0)
    return v[["item_id", "store_id", "date", "seasonal_naive_forecast"]]


def run_ets(train, ab_items, val_dates):
    """Fit ETS per A/B item-store on last 365 train days; return dict[(item,store)]->np.array(28)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    cutoff, _ = C.get_cutoff_date()
    start_365 = cutoff - pd.Timedelta(days=364)
    recent = train[train["date"] >= start_365][
        ["item_id", "store_id", "date", "units_sold"]
    ].sort_values(["item_id", "store_id", "date"])

    grouped = dict(tuple(recent.groupby(["item_id", "store_id"], observed=True)))

    out = {}
    n = len(ab_items)
    fails = 0
    for i, key in enumerate(ab_items, 1):
        if i % 500 == 0 or i == n:
            print(f"  ETS progress: {i}/{n}  (fallbacks so far: {fails})")
        sub = grouped.get(key)
        if sub is None or len(sub) < 14:
            out[key] = None  # no usable history -> caller uses seasonal naive
            continue
        series = sub["units_sold"].astype("float64").values
        last28_mean = float(series[-28:].mean())
        try:
            model = ExponentialSmoothing(
                series, trend="add", seasonal="add", seasonal_periods=7,
                initialization_method="estimated",
            )
            res = model.fit()
            fc = np.asarray(res.forecast(C.FORECAST_HORIZON), dtype="float64")
            if not np.all(np.isfinite(fc)):
                raise ValueError("non-finite forecast")
        except Exception:
            fails += 1
            fc = np.full(C.FORECAST_HORIZON, last28_mean, dtype="float64")
        out[key] = np.clip(fc, 0, None)
    print(f"  ETS done: {n} A/B item-stores fit, {fails} fell back to last-28 mean.")
    return out


def metrics_table(df):
    """Build the requested Category | Metric | Seasonal Naive | ETS table."""
    rows = []
    metric_fns = [("RMSE", C.rmse), ("MAE", C.mae), ("MAPE", C.mape), ("Bias", C.bias)]
    for cat in C.CATEGORIES:
        sub = df[df["cat_id"] == cat]
        for mname, fn in metric_fns:
            rows.append({
                "Category": cat,
                "Metric": mname,
                "Seasonal Naive": fn(sub["actual"], sub["seasonal_naive_forecast"]),
                "ETS": fn(sub["actual"], sub["ets_forecast"]),
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print(f"DATA SCOPE: {C.DATA_SCOPE}")
    print("TASK 3 - Baseline models (Seasonal Naive + ETS)")
    print("=" * 78)

    train = pd.read_parquet(C.TRAIN_PARQUET,
                            columns=["item_id", "store_id", "cat_id", "date",
                                     "units_sold", "abc_class"])
    val = pd.read_parquet(C.VAL_PARQUET,
                          columns=["item_id", "store_id", "cat_id", "date", "units_sold"])
    train["date"] = pd.to_datetime(train["date"])
    val["date"] = pd.to_datetime(val["date"])
    for c in ["item_id", "store_id", "cat_id"]:
        train[c] = train[c].astype(str)
        val[c] = val[c].astype(str)
    print(f"loaded train {train.shape}, val {val.shape}")

    val_dates = sorted(val["date"].unique())
    print(f"validation horizon: {len(val_dates)} days "
          f"({pd.Timestamp(val_dates[0]).date()} .. {pd.Timestamp(val_dates[-1]).date()})")

    # ---- Baseline 1: Seasonal Naive ---------------------------------------
    print("\n[1/2] Seasonal Naive ...")
    sn = seasonal_naive(train, val)
    print(f"  seasonal-naive forecasts: {len(sn)} rows")

    # ---- Baseline 2: ETS on A/B items -------------------------------------
    abc_map = (train.drop_duplicates(["item_id", "store_id"])
                    .set_index(["item_id", "store_id"])["abc_class"].to_dict())
    ab_items = [k for k, v in abc_map.items() if v in ("A", "B")]
    print(f"\n[2/2] ETS on {len(ab_items)} A/B item-stores "
          f"(C-class items reuse seasonal naive) ...")
    ets_fc = run_ets(train, ab_items, val_dates)

    # build ETS long frame: one row per (item,store,date)
    date_index = {pd.Timestamp(d): h for h, d in enumerate(val_dates)}
    ets_rows = []
    for key, arr in ets_fc.items():
        if arr is None:
            continue
        it, st = key
        for d, h in date_index.items():
            ets_rows.append((it, st, d, float(arr[h])))
    ets_df = pd.DataFrame(ets_rows, columns=["item_id", "store_id", "date", "ets_forecast"])

    # ---- assemble forecast frame ------------------------------------------
    fc = val[["item_id", "store_id", "date", "cat_id", "units_sold"]].rename(
        columns={"units_sold": "actual"})
    fc = fc.merge(sn, on=["item_id", "store_id", "date"], how="left")
    fc = fc.merge(ets_df, on=["item_id", "store_id", "date"], how="left")
    # C-class items (and any ETS gap) reuse seasonal naive as the ETS column
    fc["ets_forecast"] = fc["ets_forecast"].fillna(fc["seasonal_naive_forecast"])
    fc["ets_forecast"] = fc["ets_forecast"].clip(lower=0)

    out_csv = os.path.join(C.OUT, "baseline_forecasts.csv")
    fc[["item_id", "store_id", "date", "actual",
        "seasonal_naive_forecast", "ets_forecast"]].to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}  ({len(fc):,} rows)")

    # ---- metrics ----------------------------------------------------------
    mt = metrics_table(fc)
    out_metrics = os.path.join(C.OUT, "baseline_metrics.csv")
    mt.to_csv(out_metrics, index=False)

    print("\n" + "=" * 70)
    print("BASELINE METRICS BY CATEGORY")
    print("=" * 70)
    print(f"{'Category':<10} | {'Metric':<6} | {'Seasonal Naive':>14} | {'ETS':>10}")
    print("-" * 70)
    for _, r in mt.iterrows():
        print(f"{r['Category']:<10} | {r['Metric']:<6} | "
              f"{r['Seasonal Naive']:>14.4f} | {r['ETS']:>10.4f}")
    print(f"\nsaved {out_metrics}")
    print("DONE.")


if __name__ == "__main__":
    main()
