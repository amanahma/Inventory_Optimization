"""
newsvendor.py  --  M5 Inventory Optimizer (Week 3, Task 6)

NEWSVENDOR vs EOQ
-----------------
EOQ        = repeat-purchase items, stable demand, reorder anytime.
Newsvendor = single-period / perishable: order once BEFORE demand is revealed
             (e.g. fresh food ordered in the morning for that day).
FOODS items in M5 include perishables -> Newsvendor is the right OR model.

FORMULA
-------
    Critical ratio  CR = Cu / (Cu + Co)
    Q*             = F_inv(CR; mean=forecast_mean, std=demand_std)
where
    Cu = underage cost = lost profit per stockout unit = sell_price * MARGIN_RATE
    Co = overage  cost = waste/markdown per unsold unit = sell_price * DISPOSAL_COST_RATE
    demand_std = max(forecast_error_std, 1)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

import config as cfg


def expected_profit(Q, mu, sigma, Cu, Co):
    """Expected profit for order qty Q under N(mu, sigma) demand.

    expected_sales    = mu*F(Q) + Q*(1-F(Q))   (per the task spec)
    expected_leftover = Q - expected_sales
    profit            = expected_sales*Cu - expected_leftover*Co
    """
    F = norm.cdf(Q, loc=mu, scale=sigma)
    exp_sales = mu * F + Q * (1 - F)
    exp_leftover = Q - exp_sales
    return exp_sales * Cu - exp_leftover * Co, exp_sales, exp_leftover, F


def main():
    print("=" * 72)
    print("TASK 6 — Newsvendor model (FOODS category)")
    print("=" * 72)

    # ---------------------------------------------------------------- Step 1
    df = pd.read_csv(os.path.join(cfg.OUT, "reorder_point_results.csv"))
    foods = df[df["cat_id"] == "FOODS"].copy()
    print(f"[Step 1] FOODS item-store combinations: {len(foods):,}")

    # ---------------------------------------------------------------- Step 2
    foods["sell_price"] = foods["sell_price"].clip(lower=0.01)
    foods["Cu"] = foods["sell_price"] * cfg.MARGIN_RATE
    foods["Co"] = foods["sell_price"] * cfg.DISPOSAL_COST_RATE
    foods["CR"] = foods["Cu"] / (foods["Cu"] + foods["Co"])
    foods["demand_std"] = foods["forecast_error_std"].clip(lower=1.0)

    foods["Q_star"] = norm.ppf(foods["CR"], loc=foods["forecast_mean"],
                               scale=foods["demand_std"])
    foods["Q_star"] = foods["Q_star"].clip(lower=0).round(1)

    # ---------------------------------------------------------------- Step 3
    foods["q_difference"] = foods["Q_star"] - foods["EOQ"]
    foods["q_difference_pct"] = np.where(
        foods["EOQ"] > 0,
        (foods["Q_star"] - foods["EOQ"]) / foods["EOQ"] * 100, 0.0)

    # ---------------------------------------------------------------- Step 4
    pf_ns, es, el, _ = expected_profit(
        foods["Q_star"].to_numpy(), foods["forecast_mean"].to_numpy(),
        foods["demand_std"].to_numpy(), foods["Cu"].to_numpy(),
        foods["Co"].to_numpy())
    foods["expected_sales"] = es
    foods["expected_leftover"] = el
    foods["expected_stockout"] = foods["forecast_mean"] - foods["expected_sales"]
    foods["expected_profit_newsvendor"] = pf_ns

    # ---------------------------------------------------------------- Step 5
    pf_eoq, _, _, _ = expected_profit(
        foods["EOQ"].to_numpy(), foods["forecast_mean"].to_numpy(),
        foods["demand_std"].to_numpy(), foods["Cu"].to_numpy(),
        foods["Co"].to_numpy())
    foods["expected_profit_eoq"] = pf_eoq
    foods["profit_improvement"] = (foods["expected_profit_newsvendor"]
                                   - foods["expected_profit_eoq"])
    foods["profit_improvement_pct"] = np.where(
        foods["expected_profit_eoq"].abs() > 0,
        foods["profit_improvement"] / foods["expected_profit_eoq"].abs() * 100,
        0.0)

    # ---------------------------------------------------------------- Step 6
    out_cols = ["item_id", "store_id", "cat_id", "dept_id", "abc_class",
                "sell_price", "Cu", "Co", "CR",
                "forecast_mean", "demand_std",
                "Q_star", "EOQ", "q_difference", "q_difference_pct",
                "expected_sales", "expected_leftover", "expected_stockout",
                "expected_profit_newsvendor", "expected_profit_eoq",
                "profit_improvement", "profit_improvement_pct"]
    out_path = os.path.join(cfg.OUT, "newsvendor_results.csv")
    foods[out_cols].to_csv(out_path, index=False)
    print(f"[Step 6] saved -> {out_path}")

    # ---------------------------------------------------------------- Step 7
    print("\n" + "-" * 72)
    print("ANALYSIS")
    print("-" * 72)
    avg_cr = foods["CR"].mean()
    print(f"Average CR across FOODS items: {avg_cr:.3f}  "
          f"({'underage dominates -> order ABOVE mean' if avg_cr > 0.5 else 'overage dominates -> order BELOW mean'})")
    print(f"Average Q_star : {foods['Q_star'].mean():.2f}")
    print(f"Average EOQ    : {foods['EOQ'].mean():.2f}")

    tot_impr = foods["profit_improvement"].sum()
    print(f"\nTotal expected profit improvement (Newsvendor vs naive EOQ): "
          f"${tot_impr:,.2f}")

    pct_less = 100 * (foods["Q_star"] < foods["EOQ"]).mean()
    pct_more = 100 * (foods["Q_star"] > foods["EOQ"]).mean()
    print(f"% FOODS items where Newsvendor orders LESS than EOQ: {pct_less:.1f}%")
    print(f"% FOODS items where Newsvendor orders MORE than EOQ: {pct_more:.1f}%")

    print("\nTop 10 items by profit improvement from Newsvendor:")
    top = foods.nlargest(10, "profit_improvement")[
        ["item_id", "store_id", "abc_class", "CR", "Q_star", "EOQ",
         "profit_improvement", "profit_improvement_pct"]]
    print(top.to_string(index=False))

    print(f"\nINSIGHT: For FOODS items with CR = {avg_cr:.2f}, the optimal strategy is")
    print(f"to stock at the {avg_cr*100:.0f}th percentile of demand, meaning we accept")
    print(f"a {(1-avg_cr)*100:.0f}% chance of stocking out to avoid excessive waste cost.")
    print("=" * 72)


if __name__ == "__main__":
    main()
