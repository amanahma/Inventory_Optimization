"""
generate_summary_stats.py  --  M5 Inventory Optimizer (Week 4, Task 2)

Generates every number that goes into the README and CV bullet points, all
computed from the actual data files (nothing hardcoded). Saves the results to
outputs/project_summary_stats.json so the README script can reference them.

Data scope: Walmart M5 dataset, CA_1 store only (3,049 item-store combinations).
"""

import os
import json
import numpy as np
import pandas as pd

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def p(label, value):
    print(f"  {label}: {value}")


def main():
    # ----------------------------------------------------------------- load
    main_df = pd.read_csv(os.path.join(OUT, "powerbi_main_dashboard.csv"))
    nv_df = pd.read_csv(os.path.join(OUT, "powerbi_newsvendor.csv"))
    pulp_df = pd.read_csv(os.path.join(OUT, "powerbi_pulp_scenarios.csv"))
    model_df = pd.read_csv(os.path.join(OUT, "model_comparison_final.csv"))

    # supporting files
    safety_df = pd.read_csv(os.path.join(OUT, "safety_stock.csv"))
    croston_df = pd.read_csv(os.path.join(OUT, "croston_metrics.csv"))
    policy_df = pd.read_csv(os.path.join(OUT, "policy_comparison.csv"))
    dim_date = pd.read_csv(os.path.join(OUT, "powerbi_dim_date.csv"))

    stats = {}

    cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]

    # ============================================================ DATASET
    print("=" * 70)
    print("DATASET STATS  --  Data scope: Walmart M5, CA_1 store only")
    print("=" * 70)

    n_combos = int(len(main_df))
    store = "CA_1"
    n_items = int(main_df["item_id"].nunique())
    cat_names = sorted(main_df["cat_id"].unique().tolist())

    # training date range: d_1 .. d_1885 (TRAIN_END_DAY in config)
    dd = dim_date.copy()
    train_dates = dd[dd["d"].isin([f"d_{i}" for i in range(1, 1886)])]
    train_start = str(pd.to_datetime(train_dates["date"]).min().date())
    train_end = str(pd.to_datetime(train_dates["date"]).max().date())
    date_range = f"{train_start} to {train_end}"

    stats["dataset"] = {
        "total_item_store_combinations": n_combos,
        "store_analyzed": store,
        "training_date_range": date_range,
        "training_start": train_start,
        "training_end": train_end,
        "total_unique_items": n_items,
        "n_categories": len(cat_names),
        "category_names": cat_names,
    }
    p("Total item-store combinations analyzed", n_combos)
    p("Store analyzed", store)
    p("Training date range (d_1..d_1885)", date_range)
    p("Total unique items", n_items)
    p("Total categories", f"{len(cat_names)}  {cat_names}")

    # ============================================================ FORECASTING
    print("\n" + "=" * 70)
    print("FORECASTING STATS")
    print("=" * 70)

    def model_rmse(model, cat):
        row = model_df[(model_df["Model"] == model) & (model_df["Category"] == cat)]
        return float(row["RMSE"].iloc[0]) if len(row) else None

    snaive_rmse = {c: model_rmse("Seasonal Naive", c) for c in cats}
    lgbm_rmse = {c: model_rmse("LightGBM", c) for c in cats}
    lgbm_improvement = {
        c: round((snaive_rmse[c] - lgbm_rmse[c]) / snaive_rmse[c] * 100.0, 1)
        for c in cats
    }

    # Z-class pairs covered by Croston
    n_zclass = int((main_df["xyz_class"] == "Z").sum())
    n_croston = int((main_df["best_model_used"].str.contains("Croston", case=False)).sum())

    # Croston vs Seasonal Naive on Z-class (croston_metrics.csv) — overall RMSE
    cr = croston_df.pivot_table(index="Category", columns="Metric",
                                values="Croston SBA", aggfunc="first")
    sn = croston_df.pivot_table(index="Category", columns="Metric",
                                values="Seasonal Naive", aggfunc="first")
    croston_imp_by_cat = {}
    for c in cats:
        cr_rmse = float(cr.loc[c, "RMSE"])
        sn_rmse = float(sn.loc[c, "RMSE"])
        croston_imp_by_cat[c] = round((sn_rmse - cr_rmse) / sn_rmse * 100.0, 1)
    # overall (weighted simple mean across categories of pooled Z-class)
    cr_all = float(np.mean([float(cr.loc[c, "RMSE"]) for c in cats]))
    sn_all = float(np.mean([float(sn.loc[c, "RMSE"]) for c in cats]))
    croston_imp_overall = round((sn_all - cr_all) / sn_all * 100.0, 1)

    stats["forecasting"] = {
        "seasonal_naive_rmse_by_category": {c: round(snaive_rmse[c], 4) for c in cats},
        "lightgbm_rmse_by_category": {c: round(lgbm_rmse[c], 4) for c in cats},
        "lightgbm_pct_improvement_over_snaive": lgbm_improvement,
        "n_zclass_pairs_covered_by_croston": n_zclass,
        "n_pairs_using_croston_model": n_croston,
        "croston_pct_improvement_over_snaive_by_category": croston_imp_by_cat,
        "croston_pct_improvement_over_snaive_overall": croston_imp_overall,
    }
    print("  Seasonal Naive RMSE by category:")
    for c in cats:
        p(f"    {c}", round(snaive_rmse[c], 4))
    print("  LightGBM RMSE by category:")
    for c in cats:
        p(f"    {c}", round(lgbm_rmse[c], 4))
    print("  LightGBM % improvement over Seasonal Naive:")
    for c in cats:
        p(f"    {c}", f"{lgbm_improvement[c]}%")
    p("Z-class (intermittent) item-store pairs", n_zclass)
    p("Pairs forecast with Croston SBA model", n_croston)
    print("  Croston % improvement over Seasonal Naive (Z-class):")
    for c in cats:
        p(f"    {c}", f"{croston_imp_by_cat[c]}%")
    p("Croston % improvement (overall, Z-class)", f"{croston_imp_overall}%")

    # ============================================================ INVENTORY
    print("\n" + "=" * 70)
    print("INVENTORY STATS")
    print("=" * 70)

    n_ss = int(main_df["safety_stock"].notna().sum())
    ss_by_abc = {k: round(float(v), 2)
                 for k, v in main_df.groupby("abc_class")["safety_stock"].mean().items()}

    # total annual safety stock holding cost
    if "ss_holding_cost" in policy_df.columns:
        total_ss_cost = round(float(policy_df["ss_holding_cost"].sum()), 2)
    else:
        total_ss_cost = round(float(safety_df["ss_holding_cost"].sum()), 2)

    eoq_by_cat = {c: round(float(v), 2)
                  for c, v in main_df.groupby("cat_id")["EOQ"].mean().items()}

    pct_stockout = round(float(main_df["stockout_risk"].mean()) * 100.0, 1)
    n_a_at_risk = int(((main_df["abc_class"] == "A") & (main_df["stockout_risk"] == 1)).sum())

    doh_by_cat = {c: round(float(v), 2)
                  for c, v in main_df.groupby("cat_id")["DOH"].mean().items()}
    turnover_by_cat = {c: round(float(v), 2)
                       for c, v in main_df.groupby("cat_id")["turnover"].mean().items()}

    stats["inventory"] = {
        "n_with_safety_stock": n_ss,
        "avg_safety_stock_by_abc": ss_by_abc,
        "total_annual_ss_holding_cost": total_ss_cost,
        "avg_eoq_by_category": eoq_by_cat,
        "pct_items_stockout_risk": pct_stockout,
        "n_a_class_at_stockout_risk": n_a_at_risk,
        "avg_doh_by_category": doh_by_cat,
        "avg_turnover_by_category": turnover_by_cat,
    }
    p("Item-store combinations with safety stock", n_ss)
    print("  Average safety stock by ABC class:")
    for k in ["A", "B", "C"]:
        p(f"    {k}", f"{ss_by_abc.get(k)} units")
    p("Total annual safety-stock holding cost", f"${total_ss_cost:,.2f}")
    print("  Average EOQ by category:")
    for c in cats:
        p(f"    {c}", f"{eoq_by_cat.get(c)} units")
    p("% items flagged stockout risk", f"{pct_stockout}%")
    p("A-class items at stockout risk", n_a_at_risk)
    print("  Average DOH by category:")
    for c in cats:
        p(f"    {c}", doh_by_cat.get(c))
    print("  Average inventory turnover by category:")
    for c in cats:
        p(f"    {c}", turnover_by_cat.get(c))

    # ============================================================ COST COMPARISON
    print("\n" + "=" * 70)
    print("COST COMPARISON STATS")
    print("=" * 70)

    # Headline policy saving uses the FULL policy cost (ordering + holding +
    # stockout) from policy_comparison.csv -> $195,100 / 54.4% reduction.
    total_naive = round(float(policy_df["naive_total_cost"].sum()), 2)
    total_opt = round(float(policy_df["optimized_total_cost"].sum()), 2)
    total_saving = round(total_naive - total_opt, 2)
    pct_reduction = round(total_saving / total_naive * 100.0, 1)

    saving_by_cat = {}
    for c in cats:
        sub = policy_df[policy_df["cat_id"] == c]
        saving_by_cat[c] = round(float(sub["annual_saving"].sum()), 2)

    # EOQ-only cost reduction (ordering + holding, excludes stockout) -> 49.1%
    eoq_naive = float(main_df["total_annual_cost_naive"].sum())
    eoq_opt = float(main_df["total_annual_cost_EOQ"].sum())
    eoq_reduction_pct = round((eoq_naive - eoq_opt) / eoq_naive * 100.0, 1)

    stats["cost_comparison"] = {
        "total_naive_annual_cost": total_naive,
        "total_optimized_annual_cost": total_opt,
        "total_annual_saving": total_saving,
        "pct_cost_reduction": pct_reduction,
        "saving_by_category": saving_by_cat,
        "eoq_only_cost_reduction_pct": eoq_reduction_pct,
        "eoq_only_naive_cost": round(eoq_naive, 2),
        "eoq_only_optimized_cost": round(eoq_opt, 2),
    }
    p("Total naive policy annual cost (full policy)", f"${total_naive:,.2f}")
    p("Total optimized policy annual cost (full policy)", f"${total_opt:,.2f}")
    p("Total annual saving", f"${total_saving:,.2f}")
    p("% cost reduction (full policy)", f"{pct_reduction}%")
    p("EOQ-only cost reduction (ordering+holding)", f"{eoq_reduction_pct}%")
    print("  Saving by category:")
    for c in cats:
        p(f"    {c}", f"${saving_by_cat[c]:,.2f}")

    # ============================================================ NEWSVENDOR
    print("\n" + "=" * 70)
    print("NEWSVENDOR STATS (FOODS only)")
    print("=" * 70)

    n_nv = int(len(nv_df))
    avg_cr = round(float(nv_df["CR"].mean()), 4)
    avg_qstar = round(float(nv_df["Q_star"].mean()), 2)
    total_profit_imp = round(float(nv_df["profit_improvement"].sum()), 2)

    stats["newsvendor"] = {
        "n_foods_item_store_combinations": n_nv,
        "avg_critical_ratio": avg_cr,
        "avg_q_star": avg_qstar,
        "total_expected_profit_improvement": total_profit_imp,
    }
    p("FOODS item-store combinations", n_nv)
    p("Average critical ratio (CR)", avg_cr)
    p("Average Q_star", avg_qstar)
    p("Total expected profit improvement", f"${total_profit_imp:,.2f}")

    # ====================================================== KEY INSIGHT (raw vs fc)
    # Forecast-error safety stock for A-class vs raw-demand safety stock at SL=98%.
    print("\n" + "=" * 70)
    print("KEY INSIGHT — forecast-error vs raw-demand safety stock (A-class)")
    print("=" * 70)
    from scipy.stats import norm
    z98 = float(norm.ppf(0.98))
    lead = 7

    a_safety = safety_df[safety_df["abc_class"] == "A"].copy()
    # forecast-error sizing (already in data): mean A-class safety stock
    fc_ss_a = round(float(main_df[main_df["abc_class"] == "A"]["safety_stock"].mean()), 2)
    # raw-demand sizing: z98 * raw_demand_std * sqrt(lead); raw demand std ~ actual std.
    # cv_actual * actual_mean approximates raw daily demand std.
    a_raw_std = (a_safety["cv_actual"] * a_safety["actual_mean"]).replace([np.inf, -np.inf], np.nan).dropna()
    raw_ss_a = round(float((z98 * a_raw_std * np.sqrt(lead)).mean()), 2)
    holding_rate_proxy = float((a_safety["ss_holding_cost"] / a_safety["safety_stock"].replace(0, np.nan)).mean())
    fc_cost = fc_ss_a * holding_rate_proxy
    raw_cost = raw_ss_a * holding_rate_proxy
    ss_reduction_pct = round((raw_ss_a - fc_ss_a) / raw_ss_a * 100.0, 1) if raw_ss_a else 0.0

    stats["key_insight"] = {
        "a_class_forecast_error_safety_stock": fc_ss_a,
        "a_class_raw_demand_safety_stock": raw_ss_a,
        "safety_stock_reduction_pct": ss_reduction_pct,
        "service_level_used": 0.98,
    }
    p("A-class safety stock (forecast-error sizing)", f"{fc_ss_a} units")
    p("A-class safety stock (raw-demand sizing, SL=98%)", f"{raw_ss_a} units")
    p("Safety-stock reduction from forecast-error sizing", f"{ss_reduction_pct}%")

    # cost assumptions (from config) for README reference
    stats["cost_assumptions"] = {
        "holding_cost_rate": 0.20,
        "ordering_cost": 5.00,
        "lead_time_days": 7,
        "margin_rate": 0.30,
        "disposal_cost_rate": 0.10,
    }

    # ----------------------------------------------------------------- save
    out_path = os.path.join(OUT, "project_summary_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("\n" + "=" * 70)
    print(f"Saved all summary stats -> {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
