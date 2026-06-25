"""
safety_stock.py  --  M5 Inventory Optimizer (Week 3, Task 3)

FORMULA
-------
    SS = Z * sigma_L
where
    Z        = norm.ppf(service_level_for_this_item)   # from ABC class
    sigma_L  = forecast_error_std * sqrt(LEAD_TIME_DAYS)

sigma_L is the per-day FORECAST-ERROR std scaled up to the lead-time window.
Example: lead time = 7 days, daily forecast_error_std = 2
         -> sigma_L = 2 * sqrt(7) = 5.29 units
         -> buffer needed = Z * 5.29 to cover lead-time demand uncertainty.

NOTE: sigma_L is built from forecast ERROR std, not raw demand std — an
accurate model => small sigma_L => small (cheap) safety stock.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

import config as cfg


def main():
    print("=" * 72)
    print("TASK 3 — Safety stock calculation")
    print("=" * 72)

    # ---------------------------------------------------------------- Step 1
    df = pd.read_csv(os.path.join(cfg.OUT, "item_forecast_stats.csv"))
    print(f"[Step 1] item_forecast_stats.csv loaded: {len(df):,} item-stores")

    # ---------------------------------------------------------------- Step 2
    # Service level + Z-score per item from its ABC class.
    df["service_level"] = df["abc_class"].map(cfg.SERVICE_LEVEL)
    # any unmapped class -> treat as C (lowest tier) so nothing divides/blows up
    df["service_level"] = df["service_level"].fillna(cfg.SERVICE_LEVEL["C"])
    df["Z_score"] = norm.ppf(df["service_level"])

    # ---------------------------------------------------------------- Step 3
    df["sigma_L"] = df["forecast_error_std"] * np.sqrt(cfg.LEAD_TIME_DAYS)
    df["safety_stock"] = (df["Z_score"] * df["sigma_L"]).clip(lower=0).round(1)

    # ---------------------------------------------------------------- Step 4
    df["ss_holding_cost"] = (df["safety_stock"] * df["sell_price"]
                             * cfg.HOLDING_COST_RATE)

    # ---------------------------------------------------------------- Step 5
    # Sensitivity: safety stock + holding cost at 90 / 95 / 98% service.
    z90, z95, z98 = norm.ppf(0.90), norm.ppf(0.95), norm.ppf(0.98)
    for z, lvl in [(z90, "90"), (z95, "95"), (z98, "98")]:
        df[f"safety_stock_{lvl}"] = (z * df["sigma_L"]).clip(lower=0).round(1)
        df[f"ss_cost_{lvl}"] = (df[f"safety_stock_{lvl}"] * df["sell_price"]
                                * cfg.HOLDING_COST_RATE)

    # ---------------------------------------------------------------- Step 6
    out_path = os.path.join(cfg.OUT, "safety_stock.csv")
    df.to_csv(out_path, index=False)
    print(f"[Step 6] saved -> {out_path}")

    # ---------------------------------------------------------------- Step 7
    print("\n" + "-" * 72)
    print("ANALYSIS")
    print("-" * 72)
    print(f"Z-scores used: A(0.98)->{norm.ppf(0.98):.3f}  "
          f"B(0.95)->{norm.ppf(0.95):.3f}  C(0.90)->{norm.ppf(0.90):.3f}")

    print("\nAverage safety stock by ABC class:")
    print(df.groupby("abc_class")["safety_stock"].mean().round(2).to_string())

    print("\nAverage safety stock by category:")
    print(df.groupby("cat_id")["safety_stock"].mean().round(2).to_string())

    total_ss_cost = df["ss_holding_cost"].sum()
    print(f"\nTotal annual safety-stock holding cost (all items): ${total_ss_cost:,.2f}")

    cost_90 = df["ss_cost_90"].sum()
    cost_98 = df["ss_cost_98"].sum()
    diff = cost_98 - cost_90
    print(f"\nCost of service level (90% -> 98%):")
    print(f"  Total SS holding cost @ 90% service: ${cost_90:,.2f}")
    print(f"  Total SS holding cost @ 98% service: ${cost_98:,.2f}")
    print(f"  EXTRA cost to go from 90% to 98%   : ${diff:,.2f} "
          f"(+{100*diff/cost_90:.1f}%)")

    print("\nTop 10 items by safety stock requirement:")
    top = df.nlargest(10, "safety_stock")[
        ["item_id", "store_id", "cat_id", "abc_class", "xyz_class",
         "forecast_error_std", "safety_stock", "ss_holding_cost"]]
    print(top.to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
