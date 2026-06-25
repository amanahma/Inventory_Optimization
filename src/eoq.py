"""
eoq.py  --  M5 Inventory Optimizer (Week 3, Task 4)

ECONOMIC ORDER QUANTITY
-----------------------
    EOQ = sqrt( 2 * D_annual * ordering_cost / holding_cost_per_unit )
where
    D_annual             = forecast_mean * WORKING_DAYS_YEAR
    ordering_cost        = ORDERING_COST                ($5 / order)
    holding_cost_per_unit= sell_price * HOLDING_COST_RATE

EOQ is the order size that minimises:
    annual ordering cost = (D_annual / EOQ) * ordering_cost
  + annual holding cost  = (EOQ / 2)       * holding_cost_per_unit
"""

import os
import numpy as np
import pandas as pd

import config as cfg


def main():
    print("=" * 72)
    print("TASK 4 — Economic Order Quantity (EOQ)")
    print("=" * 72)

    # ---------------------------------------------------------------- Step 1
    df = pd.read_csv(os.path.join(cfg.OUT, "safety_stock.csv"))
    print(f"[Step 1] safety_stock.csv loaded: {len(df):,} item-stores")

    # ---------------------------------------------------------------- Step 2
    df["D_annual"] = df["forecast_mean"] * cfg.WORKING_DAYS_YEAR
    # guard sell_price away from zero so holding_cost_per_unit > 0
    df["sell_price"] = df["sell_price"].clip(lower=0.01)
    df["holding_cost_per_unit"] = df["sell_price"] * cfg.HOLDING_COST_RATE

    # EOQ_raw; edge cases handled afterwards
    safe_hc = df["holding_cost_per_unit"].clip(lower=0.01)
    df["EOQ_raw"] = np.sqrt((2 * df["D_annual"] * cfg.ORDERING_COST) / safe_hc)
    df["EOQ"] = df["EOQ_raw"].round(0)
    # If D_annual==0 or sell_price==0 -> EOQ 1; if EOQ<1 -> 1 (min order 1 unit)
    df.loc[(df["D_annual"] <= 0) | (df["sell_price"] <= 0), "EOQ"] = 1
    df["EOQ"] = df["EOQ"].clip(lower=1)

    # ---------------------------------------------------------------- Step 3
    # Costs under the EOQ policy (holding includes safety stock buffer).
    df["annual_ordering_cost"] = (df["D_annual"] / df["EOQ"]) * cfg.ORDERING_COST
    df["annual_holding_cost"] = ((df["EOQ"] / 2 + df["safety_stock"])
                                 * df["holding_cost_per_unit"])
    df["total_annual_cost_EOQ"] = (df["annual_ordering_cost"]
                                   + df["annual_holding_cost"])

    # ---------------------------------------------------------------- Step 4
    # Naive policy = fixed order of 30 units every time.
    naive_q = 30
    df["annual_ordering_cost_naive"] = (df["D_annual"] / naive_q) * cfg.ORDERING_COST
    df["annual_holding_cost_naive"] = ((naive_q / 2 + df["safety_stock"])
                                       * df["holding_cost_per_unit"])
    df["total_annual_cost_naive"] = (df["annual_ordering_cost_naive"]
                                     + df["annual_holding_cost_naive"])

    df["cost_saving_vs_naive"] = (df["total_annual_cost_naive"]
                                  - df["total_annual_cost_EOQ"])
    df["cost_saving_pct"] = np.where(
        df["total_annual_cost_naive"] > 0,
        df["cost_saving_vs_naive"] / df["total_annual_cost_naive"] * 100,
        0.0)

    # ---------------------------------------------------------------- Step 5
    out_cols = ["item_id", "store_id", "cat_id", "abc_class", "xyz_class",
                "forecast_mean", "D_annual", "sell_price",
                "EOQ", "annual_ordering_cost", "annual_holding_cost",
                "total_annual_cost_EOQ", "total_annual_cost_naive",
                "cost_saving_vs_naive", "cost_saving_pct"]
    # carry safety_stock + holding_cost_per_unit forward for later tasks
    keep_extra = ["safety_stock", "holding_cost_per_unit",
                  "annual_ordering_cost_naive", "annual_holding_cost_naive",
                  "forecast_error_std", "dept_id", "state_id", "abc_xyz",
                  "service_level", "Z_score", "sigma_L", "ss_holding_cost"]
    full = df[out_cols + [c for c in keep_extra if c in df.columns]]
    out_path = os.path.join(cfg.OUT, "eoq_results.csv")
    full.to_csv(out_path, index=False)
    print(f"[Step 5] saved -> {out_path}")

    # ---------------------------------------------------------------- Step 6
    print("\n" + "-" * 72)
    print("ANALYSIS")
    print("-" * 72)
    print("Average EOQ by category:")
    print(df.groupby("cat_id")["EOQ"].mean().round(2).to_string())

    tot_eoq = df["total_annual_cost_EOQ"].sum()
    tot_naive = df["total_annual_cost_naive"].sum()
    saving = tot_naive - tot_eoq
    print(f"\nTotal annual cost — naive (fixed 30): ${tot_naive:,.2f}")
    print(f"Total annual cost — EOQ policy      : ${tot_eoq:,.2f}")
    print(f"Total $ saving from EOQ policy      : ${saving:,.2f}")
    print(f"% cost reduction from EOQ policy    : {100*saving/tot_naive:.1f}%")

    print("\nTop 10 items where EOQ saves the most money:")
    top = df.nlargest(10, "cost_saving_vs_naive")[
        ["item_id", "store_id", "cat_id", "abc_class", "forecast_mean",
         "EOQ", "cost_saving_vs_naive", "cost_saving_pct"]]
    print(top.to_string(index=False))

    print("\nNOTE: EOQ assumes deterministic, constant demand. For intermittent")
    print("Z-class items, EOQ is an approximation — the Newsvendor model (Task 5)")
    print("is more appropriate for single-period or perishable decisions.")
    print("=" * 72)


if __name__ == "__main__":
    main()
