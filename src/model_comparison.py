"""
model_comparison.py  --  M5 Inventory Optimizer (Week 2, Task 6)

Unifies every model's validation metrics into one table, quantifies LightGBM's
lift over the Seasonal Naive baseline per category, and Croston SBA's lift on the
intermittent Z-class SKUs, then plots a grouped RMSE bar chart.

Outputs:
  outputs/model_comparison_final.csv
  outputs/model_comparison_chart.png

NOTE: built on the CA_1-sample metrics; see config.DATA_SCOPE.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

METRICS = ["RMSE", "MAE", "MAPE", "Bias"]


def pivot_long(long_df, value_col, model_name):
    """baseline/croston metrics are long (Category, Metric, <value_col>) -> wide per model."""
    w = long_df.pivot(index="Category", columns="Metric", values=value_col)
    w = w.reindex(columns=METRICS)  # MAPE may be absent (Croston) -> NaN
    w = w.reset_index()
    w.insert(1, "Model", model_name)
    return w


def main():
    print("=" * 78)
    print(f"DATA SCOPE: {C.DATA_SCOPE}")
    print("TASK 6 - Unified model comparison")
    print("=" * 78)

    # ---- Step 1: load all metrics -----------------------------------------
    base = pd.read_csv(os.path.join(C.OUT, "baseline_metrics.csv"))
    cros = pd.read_csv(os.path.join(C.OUT, "croston_metrics.csv"))
    lgbm = pd.read_csv(os.path.join(C.OUT, "lgbm_metrics.csv"))

    # ---- Step 2: unified table --------------------------------------------
    sn = pivot_long(base, "Seasonal Naive", "Seasonal Naive")
    ets = pivot_long(base, "ETS", "ETS")
    cr = pivot_long(cros, "Croston SBA", "Croston SBA (Z-class)")
    lg = lgbm.copy()
    lg.insert(1, "Model", "LightGBM")
    lg = lg[["Category", "Model"] + METRICS]

    unified = pd.concat([sn, ets, cr, lg], ignore_index=True)
    unified = unified.sort_values(["Category", "Model"]).reset_index(drop=True)
    unified[METRICS] = unified[METRICS].round(4)

    out_csv = os.path.join(C.OUT, "model_comparison_final.csv")
    unified.to_csv(out_csv, index=False)

    print("\nUNIFIED COMPARISON TABLE")
    print("-" * 88)
    print(f"{'Category':<10} | {'Model':<22} | {'RMSE':>8} | {'MAE':>8} | {'MAPE':>9} | {'Bias':>8}")
    print("-" * 88)
    for _, r in unified.iterrows():
        mape = "  n/a" if pd.isna(r["MAPE"]) else f"{r['MAPE']:>9.3f}"
        print(f"{r['Category']:<10} | {r['Model']:<22} | {r['RMSE']:>8.4f} | "
              f"{r['MAE']:>8.4f} | {mape} | {r['Bias']:>8.4f}")

    # ---- Step 3: LightGBM vs Seasonal Naive per category (RMSE) ------------
    print("\n" + "=" * 70)
    sn_rmse = sn.set_index("Category")["RMSE"]
    lg_rmse = lg.set_index("Category")["RMSE"]
    for cat in C.CATEGORIES:
        imp = (sn_rmse[cat] - lg_rmse[cat]) / sn_rmse[cat] * 100.0
        print(f"LightGBM improves over Seasonal Naive by {imp:.1f}% on RMSE for {cat}")

    # ---- Step 4: Croston SBA vs Seasonal Naive on Z-class (RMSE) -----------
    cfc = pd.read_csv(os.path.join(C.OUT, "croston_forecasts.csv"), parse_dates=["date"])
    bfc = pd.read_csv(os.path.join(C.OUT, "baseline_forecasts.csv"), parse_dates=["date"])
    for d in (cfc, bfc):
        for c in ["item_id", "store_id"]:
            d[c] = d[c].astype(str)
    z = cfc.merge(bfc[["item_id", "store_id", "date", "seasonal_naive_forecast"]],
                  on=["item_id", "store_id", "date"], how="left")
    cr_z = C.rmse(z["actual"], z["croston_sba_forecast"])
    sn_z = C.rmse(z["actual"], z["seasonal_naive_forecast"])
    z_imp = (sn_z - cr_z) / sn_z * 100.0
    print(f"\nCroston SBA improves over Seasonal Naive by {z_imp:.1f}% on RMSE "
          f"for intermittent SKUs")

    # ---- Step 6: grouped RMSE bar chart -----------------------------------
    models = ["Seasonal Naive", "ETS", "Croston SBA (Z-class)", "LightGBM"]
    rmse_by = {m: [unified[(unified.Category == c) & (unified.Model == m)]["RMSE"].values
                   for c in C.CATEGORIES] for m in models}
    x = np.arange(len(C.CATEGORIES))
    w = 0.2
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(models):
        vals = [float(v[0]) if len(v) else np.nan for v in rmse_by[m]]
        plt.bar(x + (i - 1.5) * w, vals, w, label=m)
    plt.xticks(x, C.CATEGORIES)
    plt.ylabel("RMSE")
    plt.title("Validation RMSE by Model and Category (CA_1 sample)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT, "model_comparison_chart.png"), dpi=120)
    plt.close()
    print("\nsaved outputs/model_comparison_chart.png")
    print(f"saved {out_csv}")

    # ---- Step 7: summary paragraph ----------------------------------------
    f_lg, f_sn = lg_rmse["FOODS"], sn_rmse["FOODS"]
    f_imp = (f_sn - f_lg) / f_sn * 100.0
    print("\n" + "=" * 78)
    print("SUMMARY: On FOODS items, LightGBM achieves RMSE of "
          f"{f_lg:.2f} vs Seasonal Naive RMSE of {f_sn:.2f} ({f_imp:.1f}% improvement). "
          f"On intermittent Z-class SKUs, Croston SBA achieves RMSE of {cr_z:.2f} vs "
          f"Seasonal Naive RMSE of {sn_z:.2f} ({z_imp:.1f}% improvement). LightGBM will "
          "be used as the primary forecasting method for AX/BX/AY/BY items, and "
          "Croston SBA for Z-class items.")
    print("=" * 78)
    print("DONE.")


if __name__ == "__main__":
    main()
