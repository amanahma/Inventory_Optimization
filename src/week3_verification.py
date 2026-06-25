"""
week3_verification.py  --  M5 Inventory Optimizer (Week 3, final verification)

Runs the five Week-3 acceptance checks and prints the closing summary.
"""

import os
import sqlite3
import pandas as pd

import config as cfg


def main():
    print("=" * 72)
    print("WEEK 3 — FINAL VERIFICATION")
    print(f"DATA SCOPE: {cfg.DATA_SCOPE}")
    print("=" * 72)

    # ---------------------------------------------------------------- Check 1
    print("\n[1] Files in outputs/ and their sizes")
    print("-" * 72)
    for f in sorted(os.listdir(cfg.OUT)):
        path = os.path.join(cfg.OUT, f)
        if os.path.isfile(path):
            kb = os.path.getsize(path) / 1024
            print(f"  {f:<46}{kb:>10.1f} KB")

    # ---------------------------------------------------------------- Check 2
    print("\n[2] Row counts across fact tables")
    print("-" * 72)
    con = sqlite3.connect(cfg.DB_PATH)
    q = """
        SELECT 'fact_inventory_policy' as table_name, COUNT(*) as rows
        FROM fact_inventory_policy
        UNION ALL
        SELECT 'fact_newsvendor', COUNT(*) FROM fact_newsvendor
        UNION ALL
        SELECT 'fact_pulp_optimization', COUNT(*) FROM fact_pulp_optimization
        UNION ALL
        SELECT 'fact_forecasts', COUNT(*) FROM fact_forecasts
    """
    print(pd.read_sql_query(q, con).to_string(index=False))
    con.close()

    # ---------------------------------------------------------------- Check 3
    print("\n[3] First 5 rows of powerbi_main_dashboard.csv")
    print("-" * 72)
    dash = pd.read_csv(os.path.join(cfg.OUT, "powerbi_main_dashboard.csv"))
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(dash.head(5).to_string(index=False))

    # ---------------------------------------------------------------- Check 4
    print("\n[4] Policy comparison summary")
    print("-" * 72)
    pol = pd.read_csv(os.path.join(cfg.OUT, "policy_comparison.csv"))
    tot_naive = pol["naive_total_cost"].sum()
    tot_opt = pol["optimized_total_cost"].sum()
    tot_save = tot_naive - tot_opt
    pct = 100 * tot_save / tot_naive
    print(f"  Total naive cost      : ${tot_naive:,.2f}")
    print(f"  Total optimized cost  : ${tot_opt:,.2f}")
    print(f"  Total saving          : ${tot_save:,.2f}")
    print(f"  % reduction           : {pct:.1f}%")

    # ---------------------------------------------------------------- Check 5
    n_items = len(pol)
    eoq = pd.read_csv(os.path.join(cfg.OUT, "eoq_results.csv"))
    eoq_naive = eoq["total_annual_cost_naive"].sum()
    eoq_opt = eoq["total_annual_cost_EOQ"].sum()
    eoq_pct = 100 * (eoq_naive - eoq_opt) / eoq_naive

    nv = pd.read_csv(os.path.join(cfg.OUT, "newsvendor_results.csv"))
    n_foods = len(nv)
    avg_qstar = nv["Q_star"].mean()
    avg_eoq_f = nv["EOQ"].mean()

    pu = pd.read_csv(os.path.join(cfg.OUT, "pulp_optimization_results.csv"))
    fr = (pu.groupby("scenario").apply(
        lambda s: s["fulfilled"].sum() / s["demand_mean"].sum() * 100,
        include_groups=False))
    a = fr.get("Tight_50K", 0)
    b = fr.get("Normal_100K", 0)
    c = fr.get("Relaxed_200K", 0)

    print("\n[5] FINAL STATEMENT")
    print("=" * 72)
    print(f"Week 3 complete. OR inventory optimization layer built.")
    print(f" Safety stock calculated for {n_items:,} item-store combinations")
    print(f" using forecast-error-based sigma (not raw demand sigma).")
    print(f" EOQ policy reduces annual cost by {eoq_pct:.1f}% vs naive fixed-order policy.")
    print(f" Newsvendor model applied to {n_foods:,} FOODS items:")
    print(f" average Q_star = {avg_qstar:.1f} units vs EOQ = {avg_eoq_f:.1f} units.")
    print(f" PuLP LP solved for 3 budget scenarios: fill rates of "
          f"{a:.1f}%, {b:.1f}%, {c:.1f}%.")
    print(f" All results saved to SQLite and exported as Power BI CSVs.")
    print(f" Ready for Week 4 dashboard.")
    print("=" * 72)
    print("\nASSUMPTIONS (M5 has no cost data — standard industry values used):")
    print(f"  Holding cost rate {cfg.HOLDING_COST_RATE:.0%}/yr | Ordering cost "
          f"${cfg.ORDERING_COST:.0f}/order | Lead time {cfg.LEAD_TIME_DAYS}d | "
          f"Margin {cfg.MARGIN_RATE:.0%} | Disposal {cfg.DISPOSAL_COST_RATE:.0%}")
    print(f"  Service levels: {cfg.SERVICE_LEVEL}")


if __name__ == "__main__":
    main()
