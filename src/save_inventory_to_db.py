"""
save_inventory_to_db.py  --  M5 Inventory Optimizer (Week 3, Task 9)

Creates and populates the Week-3 inventory tables in data/m5_database.db:
  * fact_inventory_policy   (one row per item-store: SS, EOQ, ROP, costs, savings)
  * fact_newsvendor         (FOODS items: Cu/Co/CR, Q*, expected profit)
  * fact_pulp_optimization  (LP order qty / fulfilled per budget scenario)
Then runs 5 verification queries.
"""

import os
import sqlite3
import pandas as pd

import config as cfg

CREATE_INVENTORY = """
CREATE TABLE fact_inventory_policy (
    item_id TEXT, store_id TEXT, cat_id TEXT, dept_id TEXT, state_id TEXT,
    abc_class TEXT, xyz_class TEXT, abc_xyz TEXT,
    sell_price REAL, forecast_mean REAL, forecast_error_std REAL,
    service_level REAL, Z_score REAL, sigma_L REAL,
    safety_stock REAL, ss_holding_cost REAL,
    EOQ REAL, ROP REAL,
    demand_during_lead_time REAL, stockout_risk INTEGER,
    DOH REAL, turnover REAL,
    annual_ordering_cost REAL, annual_holding_cost REAL,
    total_annual_cost_EOQ REAL, total_annual_cost_naive REAL,
    annual_saving REAL, annual_saving_pct REAL,
    best_model_used TEXT
)
"""

CREATE_NEWSVENDOR = """
CREATE TABLE fact_newsvendor (
    item_id TEXT, store_id TEXT, cat_id TEXT,
    sell_price REAL, Cu REAL, Co REAL, CR REAL,
    forecast_mean REAL, demand_std REAL,
    Q_star REAL, EOQ REAL,
    expected_profit_newsvendor REAL, expected_profit_eoq REAL,
    profit_improvement REAL
)
"""

CREATE_PULP = """
CREATE TABLE fact_pulp_optimization (
    item_id TEXT, store_id TEXT, abc_class TEXT,
    scenario TEXT, order_qty REAL,
    fulfilled REAL, demand_mean REAL,
    sell_price REAL, budget_spent REAL
)
"""


def main():
    print("=" * 72)
    print("TASK 9 — Save inventory results to SQLite")
    print("=" * 72)
    con = sqlite3.connect(cfg.DB_PATH)
    cur = con.cursor()

    # ---------------------------------------------------------------- Step 2
    for t, ddl in [("fact_inventory_policy", CREATE_INVENTORY),
                   ("fact_newsvendor", CREATE_NEWSVENDOR),
                   ("fact_pulp_optimization", CREATE_PULP)]:
        cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute(ddl)
    con.commit()
    print("[Step 2] tables created (dropped if existed)")

    # ---------------------------------------------------------------- Step 3
    # fact_inventory_policy = reorder_point_results + policy_comparison (merged).
    rop = pd.read_csv(os.path.join(cfg.OUT, "reorder_point_results.csv"))
    pol = pd.read_csv(os.path.join(cfg.OUT, "policy_comparison.csv"))
    # need best_model_used (from item_forecast_stats); merge it in
    stats = pd.read_csv(os.path.join(cfg.OUT, "item_forecast_stats.csv"))[
        ["item_id", "store_id", "best_model_used"]]

    pol_cols = ["item_id", "store_id", "annual_saving", "annual_saving_pct",
                "total_annual_cost_naive"]
    inv = rop.merge(pol[pol_cols], on=["item_id", "store_id"],
                    how="left", suffixes=("", "_pol"))
    inv = inv.merge(stats, on=["item_id", "store_id"], how="left")
    # prefer policy_comparison's naive cost (it is the full naive incl stockout)
    if "total_annual_cost_naive_pol" in inv.columns:
        inv["total_annual_cost_naive"] = inv["total_annual_cost_naive_pol"]

    inv_cols = ["item_id", "store_id", "cat_id", "dept_id", "state_id",
                "abc_class", "xyz_class", "abc_xyz",
                "sell_price", "forecast_mean", "forecast_error_std",
                "service_level", "Z_score", "sigma_L",
                "safety_stock", "ss_holding_cost",
                "EOQ", "ROP",
                "demand_during_lead_time", "stockout_risk",
                "DOH", "turnover",
                "annual_ordering_cost", "annual_holding_cost",
                "total_annual_cost_EOQ", "total_annual_cost_naive",
                "annual_saving", "annual_saving_pct",
                "best_model_used"]
    inv_out = inv[inv_cols]
    inv_out.to_sql("fact_inventory_policy", con, if_exists="append", index=False)
    print(f"[Step 3] fact_inventory_policy inserted: "
          f"{pd.read_sql_query('SELECT COUNT(*) c FROM fact_inventory_policy', con).c[0]:,} rows")

    nv = pd.read_csv(os.path.join(cfg.OUT, "newsvendor_results.csv"))
    nv_cols = ["item_id", "store_id", "cat_id",
               "sell_price", "Cu", "Co", "CR",
               "forecast_mean", "demand_std",
               "Q_star", "EOQ",
               "expected_profit_newsvendor", "expected_profit_eoq",
               "profit_improvement"]
    nv[nv_cols].to_sql("fact_newsvendor", con, if_exists="append", index=False)
    print(f"         fact_newsvendor inserted: "
          f"{pd.read_sql_query('SELECT COUNT(*) c FROM fact_newsvendor', con).c[0]:,} rows")

    pu = pd.read_csv(os.path.join(cfg.OUT, "pulp_optimization_results.csv"))
    pu_cols = ["item_id", "store_id", "abc_class", "scenario", "order_qty",
               "fulfilled", "demand_mean", "sell_price", "budget_spent"]
    # abc_class not in pulp csv -> merge from inv
    if "abc_class" not in pu.columns:
        pu = pu.merge(rop[["item_id", "store_id", "abc_class"]],
                      on=["item_id", "store_id"], how="left")
    pu[pu_cols].to_sql("fact_pulp_optimization", con, if_exists="append", index=False)
    print(f"         fact_pulp_optimization inserted: "
          f"{pd.read_sql_query('SELECT COUNT(*) c FROM fact_pulp_optimization', con).c[0]:,} rows")

    # ---------------------------------------------------------------- Step 4
    queries = {
        "Query 1 — Avg safety stock & ROP by ABC class": """
            SELECT abc_class,
                   ROUND(AVG(safety_stock),2) as avg_safety_stock,
                   ROUND(AVG(ROP),2) as avg_ROP,
                   ROUND(AVG(turnover),2) as avg_turnover,
                   ROUND(AVG(annual_saving_pct),2) as avg_saving_pct
            FROM fact_inventory_policy
            GROUP BY abc_class
            ORDER BY abc_class""",
        "Query 2 — Stockout risk summary": """
            SELECT cat_id,
                   COUNT(*) as total_items,
                   SUM(stockout_risk) as items_at_risk,
                   ROUND(100.0 * SUM(stockout_risk) / COUNT(*), 1) as pct_at_risk
            FROM fact_inventory_policy
            GROUP BY cat_id""",
        "Query 3 — Total cost comparison": """
            SELECT cat_id,
                   ROUND(SUM(total_annual_cost_naive), 2) as naive_total,
                   ROUND(SUM(total_annual_cost_EOQ), 2) as optimized_total,
                   ROUND(SUM(annual_saving), 2) as total_saving,
                   ROUND(100.0 * SUM(annual_saving) / SUM(total_annual_cost_naive), 1) as saving_pct
            FROM fact_inventory_policy
            GROUP BY cat_id""",
        "Query 4 — Newsvendor vs EOQ impact (FOODS)": """
            SELECT ROUND(AVG(Q_star), 2) as avg_newsvendor_qty,
                   ROUND(AVG(EOQ), 2) as avg_eoq_qty,
                   ROUND(AVG(profit_improvement), 4) as avg_profit_improvement,
                   ROUND(AVG(100.0 * profit_improvement / ABS(expected_profit_eoq)), 1) as avg_profit_improvement_pct
            FROM fact_newsvendor""",
        "Query 5 — LP fill rate by scenario": """
            SELECT scenario,
                   ROUND(SUM(fulfilled) / SUM(demand_mean) * 100, 1) as fill_rate_pct,
                   ROUND(SUM(budget_spent), 2) as total_spent
            FROM fact_pulp_optimization
            GROUP BY scenario""",
    }
    print("\n" + "-" * 72)
    print("VERIFICATION QUERIES")
    print("-" * 72)
    for title, q in queries.items():
        print(f"\n{title}")
        print(pd.read_sql_query(q, con).to_string(index=False))

    con.commit()
    con.close()
    print("\n" + "=" * 72)
    print("All inventory tables saved + verified.")
    print("=" * 72)


if __name__ == "__main__":
    main()
