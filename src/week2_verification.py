"""
week2_verification.py  --  M5 Inventory Optimizer (Week 2, Final Verification)

Confirms all Week-2 deliverables exist and prints the closing summary.
"""

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd

import config as C

REQUIRED = [
    "baseline_forecasts.csv", "baseline_metrics.csv",
    "croston_forecasts.csv", "croston_metrics.csv",
    "lgbm_forecasts.csv", "lgbm_metrics.csv", "lgbm_model.pkl",
    "lgbm_feature_importance.png", "model_comparison_final.csv",
    "model_comparison_chart.png",
]


def main():
    print("=" * 78)
    print(f"WEEK 2 FINAL VERIFICATION   (DATA SCOPE: {C.DATA_SCOPE})")
    print("=" * 78)

    # ---- 1: required output files -----------------------------------------
    print("\n[1] Output files check:")
    all_ok = True
    for fn in REQUIRED:
        p = os.path.join(C.OUT, fn)
        ok = os.path.exists(p)
        all_ok &= ok
        size = f"{os.path.getsize(p)/1024:.1f} KB" if ok else "MISSING"
        print(f"    [{'OK' if ok else 'XX'}] {fn:<32} {size}")
    print(f"    -> all required files present: {all_ok}")

    # ---- 2: full comparison table -----------------------------------------
    print("\n[2] model_comparison_final.csv:")
    comp = pd.read_csv(os.path.join(C.OUT, "model_comparison_final.csv"))
    print(comp.to_string(index=False))

    # ---- 3: fact_forecasts row count --------------------------------------
    con = sqlite3.connect(C.DB_PATH)
    n_rows = con.execute("SELECT COUNT(*) FROM fact_forecasts").fetchone()[0]
    print(f"\n[3] fact_forecasts row count (SELECT COUNT(*)): {n_rows:,}")
    con.close()

    # ---- 4: top 5 LightGBM features ---------------------------------------
    with open(os.path.join(C.OUT, "lgbm_model.pkl"), "rb") as f:
        model = pickle.load(f)
    feats = model.booster_.feature_name()
    gains = model.booster_.feature_importance(importance_type="gain")
    imp = pd.DataFrame({"feature": feats, "gain": gains}).sort_values(
        "gain", ascending=False).head(5)
    print("\n[4] Top 5 LightGBM features by gain:")
    for _, r in imp.iterrows():
        print(f"    {r['feature']:<16} {r['gain']:>14,.0f}")

    # ---- 5: closing statement ---------------------------------------------
    lgbm = pd.read_csv(os.path.join(C.OUT, "lgbm_forecasts.csv"))
    base = pd.read_csv(os.path.join(C.OUT, "baseline_forecasts.csv"))
    lgbm_rmse = C.rmse(lgbm["actual"], lgbm["lgbm_forecast"])
    sn_rmse = C.rmse(base["actual"], base["seasonal_naive_forecast"])
    improvement = (sn_rmse - lgbm_rmse) / sn_rmse * 100.0

    cros = pd.read_csv(os.path.join(C.OUT, "croston_forecasts.csv"))
    z_combos = cros[["item_id", "store_id"]].drop_duplicates().shape[0]

    print("\n[5] " + "-" * 72)
    print(f"Week 2 complete. Models built: Seasonal Naive, ETS, Croston SBA, LightGBM.")
    print(f"LightGBM best validation RMSE: {lgbm_rmse:.4f} "
          f"(overall, all categories; Seasonal Naive {sn_rmse:.4f}).")
    print(f"Improvement over baseline: {improvement:.1f}%.")
    print(f"Intermittent SKUs covered by Croston: {z_combos:,} item-store combinations.")
    print("All forecasts saved to SQL database. Ready for Week 3 OR layer.")
    print("-" * 76)


if __name__ == "__main__":
    main()
