"""
pulp_optimization.py  --  M5 Inventory Optimizer (Week 3, Task 7)

PROBLEM
-------
Given a fixed procurement budget, decide order quantity Q[i] for each item to
MAXIMISE total expected demand fulfilled.

Because fulfilled = min(Q, demand) is non-linear, we linearise with an aux
variable F[i] bounded by both Q[i] and demand_mean[i]; maximising sum(F)
forces F[i] = min(Q[i], demand_mean[i]).

DECISION VARS : Q[i] >= 0 (order qty),  F[i] >= 0 (fulfilled)
OBJECTIVE     : max sum(F[i])
CONSTRAINTS   : sum(Q[i]*price[i]) <= budget ; F[i] <= Q[i] ; F[i] <= demand[i]
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pulp

import config as cfg


def run_lp(items, demand_mean, prices, total_budget, scenario_name):
    prob = pulp.LpProblem(f"inventory_{scenario_name}", pulp.LpMaximize)

    Q = {i: pulp.LpVariable(f"Q_{i[0]}_{i[1]}", lowBound=0) for i in items}
    F = {i: pulp.LpVariable(f"F_{i[0]}_{i[1]}", lowBound=0) for i in items}

    # Objective: maximise total fulfilled demand.
    # When the budget is not binding the LP has many optima (any Q >= demand
    # fulfils the same units). We add a tiny lexicographic penalty on spend so
    # the solver reports the MINIMUM cost that still achieves the max fill —
    # otherwise CBC may "spend" the whole budget on useless excess Q. The
    # penalty (1e-6) is far below any fill contribution, so it never changes
    # which/how-much demand gets fulfilled.
    prob += (pulp.lpSum(F[i] for i in items)
             - 1e-6 * pulp.lpSum(Q[i] * prices[i] for i in items))

    # Budget constraint
    prob += pulp.lpSum(Q[i] * prices[i] for i in items) <= total_budget

    # Fulfilment constraints
    for i in items:
        prob += F[i] <= Q[i]
        prob += F[i] <= demand_mean[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    # compute fulfilled directly from F (objective now carries a tiny penalty)
    total_fulfilled = sum(pulp.value(F[i]) for i in items)
    total_spent = sum(pulp.value(Q[i]) * prices[i] for i in items)
    total_demand = sum(demand_mean[i] for i in items)
    fill_rate = total_fulfilled / total_demand if total_demand > 0 else 0

    results = [{
        'item_id': i[0],
        'store_id': i[1],
        'order_qty': max(0, pulp.value(Q[i])),
        'fulfilled': max(0, pulp.value(F[i])),
        'demand_mean': demand_mean[i],
        'sell_price': prices[i],
        'budget_spent': max(0, pulp.value(Q[i])) * prices[i],
        'scenario': scenario_name,
    } for i in items]

    print(f"\nScenario: {scenario_name}")
    print(f"  Status: {status}")
    if status != "Optimal":
        print(f"  *** WARNING: solver status is {status}, not Optimal — investigate! ***")
    print(f"  Budget: ${total_budget:,.0f}")
    print(f"  Total spent: ${total_spent:,.2f}")
    print(f"  Total demand: {total_demand:.1f} units")
    print(f"  Total fulfilled: {total_fulfilled:.1f} units")
    print(f"  Fill rate: {fill_rate*100:.1f}%")

    return pd.DataFrame(results), fill_rate


def main():
    print("=" * 72)
    print("TASK 7 — PuLP budget-constrained multi-item optimization")
    print("=" * 72)

    # ---------------------------------------------------------------- Step 1
    df = pd.read_csv(os.path.join(cfg.OUT, "reorder_point_results.csv"))
    ab = df[df["abc_class"].isin(["A", "B"])].copy()
    print(f"[Step 1] A+B class items: {len(ab):,}")

    if len(ab) > 500:
        before = len(ab)
        ab = ab[ab["store_id"] == "CA_1"].copy()
        print(f"         >500 items -> filtered to single store CA_1: "
              f"{before:,} -> {len(ab):,} items (runtime guard). "
              f"NOTE: data is already CA_1-only in this memory-fallback build.")

    # ---------------------------------------------------------------- Step 2
    ab["sell_price"] = ab["sell_price"].clip(lower=0.01)
    items = list(zip(ab["item_id"], ab["store_id"]))
    demand_mean = {k: float(v) for k, v in zip(items, ab["forecast_mean"])}
    prices = {k: float(v) for k, v in zip(items, ab["sell_price"])}

    # ---------------------------------------------------------------- Step 3
    scenarios = [
        (cfg.BUDGET_TIGHT, "Tight_50K"),
        (cfg.BUDGET_NORMAL, "Normal_100K"),
        (cfg.BUDGET_RELAXED, "Relaxed_200K"),
    ]
    all_res, fill_rates, spent_d, fulfilled_d, demand_d = [], {}, {}, {}, {}
    for budget, name in scenarios:
        res, fr = run_lp(items, demand_mean, prices, budget, name)
        all_res.append(res)
        fill_rates[name] = fr
        spent_d[name] = res["budget_spent"].sum()
        fulfilled_d[name] = res["fulfilled"].sum()
        demand_d[name] = res["demand_mean"].sum()

    combined = pd.concat(all_res, ignore_index=True)

    # ---------------------------------------------------------------- Step 4
    print("\n" + "-" * 72)
    print("SCENARIO COMPARISON TABLE")
    print("-" * 72)
    header = f"{'Scenario':<16}{'Budget':>12}{'Fill Rate':>12}{'Items Ordered':>16}{'Avg Order Qty':>16}"
    print(header)
    budget_map = dict((n, b) for b, n in scenarios)
    for _, name in scenarios:
        s = combined[combined["scenario"] == name]
        ordered = s[s["order_qty"] > 1e-6]
        n_ord = len(ordered)
        avg_q = ordered["order_qty"].mean() if n_ord else 0.0
        print(f"{name:<16}${budget_map[name]:>10,.0f}{fill_rates[name]*100:>11.1f}%"
              f"{n_ord:>16,}{avg_q:>16.1f}")

    # ---------------------------------------------------------------- Step 5
    out_path = os.path.join(cfg.OUT, "pulp_optimization_results.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n[Step 5] saved -> {out_path}")

    # ---------------------------------------------------------------- Step 6
    names = [n for _, n in scenarios]
    spent = [spent_d[n] for n in names]
    demand = [demand_d[n] for n in names]
    fulfilled = [fulfilled_d[n] for n in names]
    frate = [fill_rates[n] * 100 for n in names]

    x = np.arange(len(names))
    w = 0.25
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.bar(x - w, spent, w, label="Budget spent ($)", color="#4C72B0")
    ax1.bar(x, demand, w, label="Total demand (units)", color="#DD8452")
    ax1.bar(x + w, fulfilled, w, label="Total fulfilled (units)", color="#55A868")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Dollars / Units")
    ax1.set_title("PuLP budget scenarios — spend vs demand vs fulfilled")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, frate, "o-", color="crimson", linewidth=2, label="Fill rate (%)")
    ax2.set_ylabel("Fill rate (%)", color="crimson")
    ax2.set_ylim(0, 105)
    for xi, fv in zip(x, frate):
        ax2.annotate(f"{fv:.1f}%", (xi, fv), textcoords="offset points",
                     xytext=(0, 8), ha="center", color="crimson", fontweight="bold")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    chart_path = os.path.join(cfg.OUT, "pulp_scenario_comparison.png")
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"[Step 6] chart saved -> {chart_path}")

    # ---------------------------------------------------------------- Step 7
    ft = fill_rates["Tight_50K"] * 100
    fn = fill_rates["Normal_100K"] * 100
    fr_ = fill_rates["Relaxed_200K"] * 100
    # marginal return per $10k from Normal -> Relaxed
    marginal = (fr_ - fn) / ((cfg.BUDGET_RELAXED - cfg.BUDGET_NORMAL) / 10_000)
    min_spend_to_fulfill = spent_d["Relaxed_200K"]  # min cost achieving max fill
    print("\n" + "-" * 72)
    print("INSIGHT")
    print("-" * 72)
    if ft >= 99.5 and fn >= 99.5 and fr_ >= 99.5:
        # Budget is NOT the binding constraint at this scope — report honestly.
        print(f"At single-store CA_1 scope, total A+B daily demand costs only "
              f"~${min_spend_to_fulfill:,.0f} to fully procure.")
        print(f"That is far below even the tight ${cfg.BUDGET_TIGHT:,} budget, so all "
              f"three scenarios reach {ft:.1f}% fill rate.")
        print("The budget constraint is NON-BINDING here: the LP confirms demand "
              "coverage is limited by demand, not capital. The marginal value of")
        print(f"extra budget above ${cfg.BUDGET_TIGHT:,} is $0 — every additional "
              f"dollar is slack. (Constraint would bite at full 10-store scope or "
              f"with multi-day procurement horizons.)")
    else:
        print(f"At ${cfg.BUDGET_TIGHT:,} budget, we can only fulfill {ft:.1f}% of demand "
              f"for A+B class items.")
        print(f"Doubling the budget to ${cfg.BUDGET_NORMAL:,} increases fill rate to {fn:.1f}%.")
        print(f"The marginal return of additional budget above ${cfg.BUDGET_NORMAL:,} is "
              f"{marginal:.2f}% more fill rate per additional $10,000 — diminishing returns "
              f"set in at the normal budget level.")
    print("=" * 72)


if __name__ == "__main__":
    main()
