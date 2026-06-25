"""
reorder_point.py  --  M5 Inventory Optimizer (Week 3, Task 5)

REORDER POINT
-------------
    ROP = (forecast_mean * LEAD_TIME_DAYS) + safety_stock

When on-hand stock falls to ROP, place an order of EOQ units.
The safety_stock term buffers demand spikes during the lead time.
"""

import os
import numpy as np
import pandas as pd

import config as cfg


def main():
    print("=" * 72)
    print("TASK 5 — Reorder point (ROP)")
    print("=" * 72)

    # ---------------------------------------------------------------- Step 1
    df = pd.read_csv(os.path.join(cfg.OUT, "eoq_results.csv"))
    print(f"[Step 1] eoq_results.csv loaded: {len(df):,} item-stores")

    # ---------------------------------------------------------------- Step 2
    df["demand_during_lead_time"] = df["forecast_mean"] * cfg.LEAD_TIME_DAYS
    df["ROP"] = (df["demand_during_lead_time"] + df["safety_stock"]).round(1)

    # ---------------------------------------------------------------- Step 3
    # Naive current stock = 2 weeks of average demand.
    df["naive_current_stock"] = df["forecast_mean"] * 14
    df["stockout_risk"] = (df["naive_current_stock"] < df["ROP"]).astype(int)

    # ---------------------------------------------------------------- Step 4
    # Days of inventory on hand.
    df["DOH"] = np.where(df["forecast_mean"] > 0,
                         df["naive_current_stock"] / df["forecast_mean"],
                         999.0)

    # ---------------------------------------------------------------- Step 5
    # Inventory turnover ratio (higher = leaner inventory).
    denom = (df["EOQ"] / 2 + df["safety_stock"]).clip(lower=0.01)
    df["turnover"] = df["D_annual"] / denom

    # ---------------------------------------------------------------- Step 6
    out_cols = ["item_id", "store_id", "cat_id", "dept_id", "abc_class", "xyz_class",
                "forecast_mean", "safety_stock", "EOQ",
                "demand_during_lead_time", "ROP",
                "naive_current_stock", "stockout_risk",
                "DOH", "turnover", "total_annual_cost_EOQ"]
    keep_extra = ["state_id", "abc_xyz", "sell_price", "D_annual",
                  "holding_cost_per_unit", "forecast_error_std",
                  "annual_ordering_cost", "annual_holding_cost",
                  "total_annual_cost_naive", "service_level", "Z_score",
                  "sigma_L", "ss_holding_cost"]
    full = df[out_cols + [c for c in keep_extra if c in df.columns]]
    out_path = os.path.join(cfg.OUT, "reorder_point_results.csv")
    full.to_csv(out_path, index=False)
    print(f"[Step 6] saved -> {out_path}")

    # ---------------------------------------------------------------- Step 7
    print("\n" + "-" * 72)
    print("ANALYSIS")
    print("-" * 72)
    pct_risk = 100 * df["stockout_risk"].mean()
    print(f"% of items flagged stockout_risk=1 : {pct_risk:.1f}% "
          f"({int(df['stockout_risk'].sum()):,} of {len(df):,})")

    print("\nStockout risk by ABC class:")
    risk = df.groupby("abc_class").agg(
        total=("stockout_risk", "size"),
        at_risk=("stockout_risk", "sum"))
    risk["pct_at_risk"] = (100 * risk["at_risk"] / risk["total"]).round(1)
    print(risk.to_string())
    a_risk = int(df[(df["abc_class"] == "A")]["stockout_risk"].sum())
    print(f"  ** {a_risk} A-class items at stockout risk — high-value, business critical **")

    print("\nAverage DOH by category:")
    print(df.groupby("cat_id")["DOH"].mean().round(2).to_string())
    print("  (industry targets — FOODS 7-14d, HOBBIES 30-45d, HOUSEHOLD 21-30d)")

    print("\nAverage inventory turnover by category:")
    print(df.groupby("cat_id")["turnover"].mean().round(2).to_string())

    print("\nTop 10 A-class items at stockout_risk=1 (highest priority to fix):")
    a_top = (df[(df["abc_class"] == "A") & (df["stockout_risk"] == 1)]
             .sort_values("forecast_mean", ascending=False)
             .head(10)[["item_id", "store_id", "cat_id", "forecast_mean",
                        "naive_current_stock", "ROP", "DOH"]])
    print(a_top.to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
