"""
policy_comparison.py  --  M5 Inventory Optimizer (Week 3, Task 8)

Business-impact calculation: optimized inventory policy vs a naive "do nothing
smart" policy.

NAIVE policy:
  * No safety stock (SS = 0)
  * Fixed order qty of 30 units every time, regardless of demand
  * Order every 14 days regardless of inventory level
  * No differentiation by item class

OPTIMIZED policy:
  * EOQ order quantity + class-based safety stock + service-level targets
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg


def main():
    print("=" * 72)
    print("TASK 8 — Policy comparison: naive vs optimized")
    print("=" * 72)

    # ---------------------------------------------------------------- Step 1
    df = pd.read_csv(os.path.join(cfg.OUT, "reorder_point_results.csv"))
    print(f"[Step 1] reorder_point_results.csv loaded: {len(df):,} item-stores")
    df["sell_price"] = df["sell_price"].clip(lower=0.01)

    # ---------------------------------------------------------------- Step 2
    # Naive policy annual cost.
    naive_q = 30
    df["naive_order_frequency"] = df["D_annual"] / naive_q
    df["naive_annual_ordering_cost"] = df["naive_order_frequency"] * cfg.ORDERING_COST
    df["naive_annual_holding_cost"] = ((naive_q / 2) * df["sell_price"]
                                       * cfg.HOLDING_COST_RATE)
    # rough stockout proxy: ~1 std dev of demand unmet per month, all 12 months
    df["naive_stockout_cost"] = (df["forecast_error_std"] * df["sell_price"]
                                 * cfg.MARGIN_RATE * 12)
    df["naive_total_cost"] = (df["naive_annual_ordering_cost"]
                              + df["naive_annual_holding_cost"]
                              + df["naive_stockout_cost"])

    # ---------------------------------------------------------------- Step 3
    # Optimized policy annual cost (EOQ total already includes SS holding) +
    # residual stockout cost scaled by (1 - service_level).
    df["optimized_stockout_cost"] = (df["forecast_error_std"] * df["sell_price"]
                                     * cfg.MARGIN_RATE * (1 - df["service_level"]))
    df["optimized_total_cost"] = (df["total_annual_cost_EOQ"]
                                  + df["optimized_stockout_cost"])

    # ---------------------------------------------------------------- Step 4
    df["annual_saving"] = df["naive_total_cost"] - df["optimized_total_cost"]
    df["annual_saving_pct"] = np.where(
        df["naive_total_cost"] > 0,
        df["annual_saving"] / df["naive_total_cost"] * 100, 0.0)

    # ---------------------------------------------------------------- Step 5
    print("\n" + "-" * 72)
    print("BY CATEGORY")
    print("-" * 72)
    cat = df.groupby("cat_id").agg(
        naive_cost=("naive_total_cost", "sum"),
        optimized_cost=("optimized_total_cost", "sum"),
        total_saving=("annual_saving", "sum"),
        avg_saving_pct=("annual_saving_pct", "mean"))
    cat["pct_reduction"] = 100 * cat["total_saving"] / cat["naive_cost"]
    print(cat.round(2).to_string())

    # ---------------------------------------------------------------- Step 6
    print("\n" + "-" * 72)
    print("BY ABC CLASS")
    print("-" * 72)
    abcg = df.groupby("abc_class").agg(
        naive_cost=("naive_total_cost", "sum"),
        optimized_cost=("optimized_total_cost", "sum"),
        total_saving=("annual_saving", "sum"),
        avg_saving_pct=("annual_saving_pct", "mean"))
    abcg["pct_reduction"] = 100 * abcg["total_saving"] / abcg["naive_cost"]
    print(abcg.round(2).to_string())

    # ---------------------------------------------------------------- Step 7
    # Service level: naive ~85%; optimized = item-count-weighted SERVICE_LEVEL.
    naive_fill = 85.0
    w = df["abc_class"].map(cfg.SERVICE_LEVEL).fillna(cfg.SERVICE_LEVEL["C"])
    opt_fill = 100 * w.mean()
    print(f"\nEstimated fill rate improvement: naive {naive_fill:.0f}% vs "
          f"optimized {opt_fill:.1f}%")

    # ---------------------------------------------------------------- Step 8
    out_path = os.path.join(cfg.OUT, "policy_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"[Step 8] saved -> {out_path}")

    # ---------------------------------------------------------------- Step 9
    # Chart 1: stacked bar of cost components, naive vs optimized, per category.
    cats = cfg.CATEGORIES
    comp = {}
    for c in cats:
        s = df[df["cat_id"] == c]
        comp[c] = {
            "naive_order": s["naive_annual_ordering_cost"].sum(),
            "naive_hold": s["naive_annual_holding_cost"].sum(),
            "naive_stockout": s["naive_stockout_cost"].sum(),
            "opt_order": s["annual_ordering_cost"].sum(),
            "opt_hold": s["annual_holding_cost"].sum(),
            "opt_stockout": s["optimized_stockout_cost"].sum(),
        }
    x = np.arange(len(cats))
    w_ = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    n_ord = [comp[c]["naive_order"] for c in cats]
    n_hold = [comp[c]["naive_hold"] for c in cats]
    n_so = [comp[c]["naive_stockout"] for c in cats]
    o_ord = [comp[c]["opt_order"] for c in cats]
    o_hold = [comp[c]["opt_hold"] for c in cats]
    o_so = [comp[c]["opt_stockout"] for c in cats]
    # naive stack (left bars)
    ax.bar(x - w_/2, n_ord, w_, label="Ordering", color="#4C72B0")
    ax.bar(x - w_/2, n_hold, w_, bottom=n_ord, label="Holding", color="#DD8452")
    ax.bar(x - w_/2, n_so, w_, bottom=np.array(n_ord)+np.array(n_hold),
           label="Stockout", color="#C44E52")
    # optimized stack (right bars) — same colors, no dup legend
    ax.bar(x + w_/2, o_ord, w_, color="#4C72B0")
    ax.bar(x + w_/2, o_hold, w_, bottom=o_ord, color="#DD8452")
    ax.bar(x + w_/2, o_so, w_, bottom=np.array(o_ord)+np.array(o_hold), color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(L=naive, R=optimized)" for c in cats])
    ax.set_ylabel("Annual cost ($)")
    ax.set_title("Naive vs Optimized annual cost breakdown by category")
    ax.legend(title="Cost component")
    fig.tight_layout()
    p1 = os.path.join(cfg.OUT, "policy_cost_comparison.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print(f"[Step 9] chart 1 saved -> {p1}")

    # Chart 2: top 20 items by annual_saving_pct.
    top20 = df.nlargest(20, "annual_saving_pct").iloc[::-1]
    labels = top20["item_id"] + " @" + top20["store_id"]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(labels, top20["annual_saving_pct"], color="#55A868")
    ax.set_xlabel("Annual saving (%)")
    ax.set_title("Top 20 items by % cost saving (optimized vs naive)")
    fig.tight_layout()
    p2 = os.path.join(cfg.OUT, "top_savings_items.png")
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    print(f"         chart 2 saved -> {p2}")

    # ---------------------------------------------------------------- Step 10
    tot_naive = df["naive_total_cost"].sum()
    tot_opt = df["optimized_total_cost"].sum()
    tot_save = tot_naive - tot_opt
    pct = 100 * tot_save / tot_naive
    # driver decomposition
    stockout_reduction = (df["naive_stockout_cost"].sum()
                          - df["optimized_stockout_cost"].sum())
    holding_increase = (df["annual_holding_cost"].sum()
                        - df["naive_annual_holding_cost"].sum())
    net = stockout_reduction - holding_increase

    def line(c):
        s = df[df["cat_id"] == c]
        sv = s["annual_saving"].sum()
        nv = s["naive_total_cost"].sum()
        return sv, (100 * sv / nv if nv else 0)

    fa, fap = line("FOODS")
    ha, hap = line("HOBBIES")
    hua, huap = line("HOUSEHOLD")

    print("\n" + "=" * 72)
    print("POLICY COMPARISON SUMMARY")
    print("=" * 72)
    print(f"Total items analyzed:    {len(df):,}")
    print(f"Naive policy total cost: ${tot_naive:,.2f} per year")
    print(f"Optimized policy cost:   ${tot_opt:,.2f} per year")
    print(f"Total annual saving:     ${tot_save:,.2f} ({pct:.1f}% reduction)")
    print("\nBy category:")
    print(f"FOODS:      ${fa:,.2f} saved ({fap:.1f}%)")
    print(f"HOBBIES:    ${ha:,.2f} saved ({hap:.1f}%)")
    print(f"HOUSEHOLD:  ${hua:,.2f} saved ({huap:.1f}%)")
    print(f"\nService level: 85% (naive) -> {opt_fill:.1f}% (optimized)")
    print(f"Key driver: Safety stock reduces stockout cost by ${stockout_reduction:,.2f},")
    print(f"offsetting holding cost increase of ${holding_increase:,.2f}, "
          f"net saving ${net:,.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
