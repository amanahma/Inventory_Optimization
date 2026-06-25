"""
generate_readme_charts.py  --  M5 Inventory Optimizer (Week 4, Task 3)

Generates the 6 professional charts embedded in the GitHub README.
All charts: dpi=150, clear titles/axis labels, and a footer note
"Data scope: Walmart M5 dataset, CA_1 store" on every figure.

Data scope: Walmart M5 dataset, CA_1 store only (3,049 item-store combinations).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
SCOPE_NOTE = "Data scope: Walmart M5 dataset, CA_1 store"
CATS = ["FOODS", "HOBBIES", "HOUSEHOLD"]

# inventory cost assumptions (from config.py)
ORDERING_COST = 5.00
NAIVE_ORDER_QTY = 30  # naive fixed order quantity


def add_scope_note(fig):
    fig.text(0.99, 0.01, SCOPE_NOTE, ha="right", va="bottom",
             fontsize=8, style="italic", color="gray")


# ---------------------------------------------------------------- Chart 1
def chart_1_model_comparison():
    model_df = pd.read_csv(os.path.join(OUT, "model_comparison_final.csv"))
    croston_df = pd.read_csv(os.path.join(OUT, "croston_metrics.csv"))

    def rmse(model, cat):
        r = model_df[(model_df["Model"] == model) & (model_df["Category"] == cat)]
        return float(r["RMSE"].iloc[0]) if len(r) else np.nan

    snaive = [rmse("Seasonal Naive", c) for c in CATS]
    ets = [rmse("ETS", c) for c in CATS]
    lgbm = [rmse("LightGBM", c) for c in CATS]
    lgbm_imp = [(s - l) / s * 100 for s, l in zip(snaive, lgbm)]

    # Z-class SN vs Croston from croston_metrics
    cr = croston_df.pivot_table(index="Category", columns="Metric",
                                values="Croston SBA", aggfunc="first")
    sn = croston_df.pivot_table(index="Category", columns="Metric",
                                values="Seasonal Naive", aggfunc="first")
    z_sn = [float(sn.loc[c, "RMSE"]) for c in CATS]
    z_cr = [float(cr.loc[c, "RMSE"]) for c in CATS]
    z_imp = [(s - c) / s * 100 for s, c in zip(z_sn, z_cr)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(CATS))
    w = 0.25

    ax1.bar(x - w, snaive, w, label="Seasonal Naive", color="#9e9e9e")
    ax1.bar(x, ets, w, label="ETS", color="#5b9bd5")
    b3 = ax1.bar(x + w, lgbm, w, label="LightGBM", color="#1f3864")
    for rect, imp in zip(b3, lgbm_imp):
        ax1.annotate(f"-{imp:.0f}%", (rect.get_x() + rect.get_width() / 2,
                     rect.get_height()), ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color="#1f3864")
    ax1.set_xticks(x)
    ax1.set_xticklabels(CATS)
    ax1.set_ylabel("RMSE (units/day)")
    ax1.set_xlabel("Category")
    ax1.set_title("All items: LightGBM vs baselines\n(annotation = LightGBM RMSE reduction vs Seasonal Naive)",
                  fontsize=11)
    ax1.legend()

    # right panel: Z-class only
    xz = np.arange(len(CATS))
    wz = 0.35
    ax2.bar(xz - wz / 2, z_sn, wz, label="Seasonal Naive", color="#9e9e9e")
    bz = ax2.bar(xz + wz / 2, z_cr, wz, label="Croston SBA", color="#c55a11")
    for rect, imp in zip(bz, z_imp):
        ax2.annotate(f"-{imp:.0f}%", (rect.get_x() + rect.get_width() / 2,
                     rect.get_height()), ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color="#c55a11")
    ax2.set_xticks(xz)
    ax2.set_xticklabels(CATS)
    ax2.set_ylabel("RMSE (units/day)")
    ax2.set_xlabel("Category")
    ax2.set_title("Z-class SKUs only:\nCroston SBA vs Seasonal Naive", fontsize=11)
    ax2.legend()

    fig.suptitle("Forecast Accuracy: LightGBM vs Baselines",
                 fontsize=15, fontweight="bold")
    add_scope_note(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(OUT, "readme_chart_1_model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- Chart 2
def chart_2_abc_xyz_heatmap():
    df = pd.read_csv(os.path.join(OUT, "abc_xyz_classification.csv"))
    df = df[df["store_id"] == "CA_1"]

    abc_order = ["A", "B", "C"]
    xyz_order = ["X", "Y", "Z"]
    counts = (df.groupby(["abc_class", "xyz_class"]).size()
              .unstack(fill_value=0).reindex(index=abc_order, columns=xyz_order, fill_value=0))

    rec_model = {
        ("A", "X"): "LightGBM", ("A", "Y"): "LightGBM", ("A", "Z"): "Croston",
        ("B", "X"): "LightGBM", ("B", "Y"): "LightGBM", ("B", "Z"): "Croston",
        ("C", "X"): "ETS", ("C", "Y"): "ETS", ("C", "Z"): "Croston",
    }
    annot = np.empty_like(counts.values, dtype=object)
    for i, a in enumerate(abc_order):
        for j, xz in enumerate(xyz_order):
            annot[i, j] = f"{counts.loc[a, xz]}\n{rec_model[(a, xz)]}"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(counts, annot=annot, fmt="", cmap="YlGnBu", cbar_kws={"label": "Item count"},
                linewidths=1, linecolor="white", annot_kws={"fontsize": 11}, ax=ax)
    ax.set_xlabel("XYZ class  (X=steady, Y=variable, Z=intermittent demand)")
    ax.set_ylabel("ABC class  (A=high, B=mid, C=low revenue)")
    ax.set_title("ABC-XYZ Segmentation Matrix — CA_1 Store\n(cell = item count + recommended forecast model)",
                 fontsize=13, fontweight="bold")
    add_scope_note(fig)
    fig.tight_layout()
    path = os.path.join(OUT, "readme_chart_2_abc_xyz_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- Chart 3
def chart_3_policy_comparison():
    df = pd.read_csv(os.path.join(OUT, "policy_comparison.csv"))

    naive_ord, naive_hold, naive_stk = [], [], []
    opt_ord, opt_hold, opt_stk = [], [], []
    for c in CATS:
        sub = df[df["cat_id"] == c]
        naive_ord.append(sub["naive_annual_ordering_cost"].sum())
        naive_hold.append(sub["naive_annual_holding_cost"].sum())
        naive_stk.append(sub["naive_stockout_cost"].sum())
        opt_ord.append(sub["annual_ordering_cost"].sum())
        opt_hold.append(sub["annual_holding_cost"].sum())
        opt_stk.append(sub["optimized_stockout_cost"].sum())

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(CATS))
    w = 0.38

    # naive stacked (left bars)
    ax.bar(x - w / 2, naive_ord, w, label="Ordering cost", color="#5b9bd5")
    ax.bar(x - w / 2, naive_hold, w, bottom=naive_ord, label="Holding cost", color="#ed7d31")
    ax.bar(x - w / 2, naive_stk, w,
           bottom=np.array(naive_ord) + np.array(naive_hold),
           label="Stockout cost", color="#c00000")
    # optimized stacked (right bars) — same colors, no extra legend entries
    ax.bar(x + w / 2, opt_ord, w, color="#5b9bd5")
    ax.bar(x + w / 2, opt_hold, w, bottom=opt_ord, color="#ed7d31")
    ax.bar(x + w / 2, opt_stk, w,
           bottom=np.array(opt_ord) + np.array(opt_hold), color="#c00000")

    naive_tot = np.array(naive_ord) + np.array(naive_hold) + np.array(naive_stk)
    opt_tot = np.array(opt_ord) + np.array(opt_hold) + np.array(opt_stk)
    for i in range(len(CATS)):
        ax.annotate("Naive", (x[i] - w / 2, naive_tot[i]), ha="center", va="bottom", fontsize=8)
        ax.annotate("Optimized", (x[i] + w / 2, opt_tot[i]), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(CATS)
    ax.set_ylabel("Annual cost ($)")
    ax.set_xlabel("Category")
    ax.set_title("Annual Inventory Cost: Naive vs Optimized Policy",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Cost component (left=Naive, right=Optimized)")
    ax.annotate("49.1% cost reduction from EOQ optimization",
                xy=(0.5, 0.92), xycoords="axes fraction", ha="center",
                fontsize=11, fontweight="bold", color="#1f3864",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff2cc", ec="#bf9000"))
    add_scope_note(fig)
    fig.tight_layout()
    path = os.path.join(OUT, "readme_chart_3_policy_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- Chart 4
def chart_4_stockout_risk():
    df = pd.read_csv(os.path.join(OUT, "powerbi_main_dashboard.csv"))
    abc_order = ["A", "B", "C"]
    pct = [df[df["abc_class"] == c]["stockout_risk"].mean() * 100 for c in abc_order]
    n_a_risk = int(((df["abc_class"] == "A") & (df["stockout_risk"] == 1)).sum())

    colors = {"A": "#c00000", "B": "#ffc000", "C": "#70ad47"}
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(abc_order, pct, color=[colors[c] for c in abc_order])
    ax.invert_yaxis()  # A on top
    for rect, v in zip(bars, pct):
        ax.annotate(f"{v:.1f}%", (rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    va="center", ha="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("% of items at stockout risk")
    ax.set_ylabel("ABC value class")
    ax.set_xlim(0, max(pct) * 1.2)
    ax.set_title("Stockout Risk by Item Value Class — CA_1 Store",
                 fontsize=14, fontweight="bold")
    ax.annotate(f"{n_a_risk} A-class items at stockout risk",
                xy=(0.97, 0.12), xycoords="axes fraction", ha="right",
                fontsize=11, fontweight="bold", color="#c00000",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fde9e9", ec="#c00000"))
    add_scope_note(fig)
    fig.tight_layout()
    path = os.path.join(OUT, "readme_chart_4_stockout_risk.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- Chart 5
def chart_5_service_level_tradeoff():
    df = pd.read_csv(os.path.join(OUT, "safety_stock.csv"))
    levels = [90, 95, 98]
    cost_cols = {90: "ss_cost_90", 95: "ss_cost_95", 98: "ss_cost_98"}

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = {"FOODS": "#c55a11", "HOBBIES": "#5b9bd5", "HOUSEHOLD": "#70ad47"}
    for c in CATS:
        sub = df[df["cat_id"] == c]
        ys = [sub[cost_cols[l]].sum() for l in levels]
        ax.plot(levels, ys, marker="o", linewidth=2, label=c, color=colors[c])
        diff = ys[-1] - ys[0]
        ax.annotate(f"+${diff:,.0f}\n(90%->98%)",
                    xy=(levels[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=colors[c])

    ax.set_xticks(levels)
    ax.set_xticklabels([f"{l}%" for l in levels])
    ax.set_xlabel("Service level")
    ax.set_ylabel("Total safety-stock holding cost ($)")
    ax.set_xlim(88, 102)
    fig.suptitle("Service Level vs Safety Stock Cost Trade-off",
                 fontsize=14, fontweight="bold")
    ax.set_title("Higher service level = more buffer stock = higher holding cost",
                 fontsize=10, style="italic")
    ax.legend(title="Category")
    add_scope_note(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = os.path.join(OUT, "readme_chart_5_service_level_tradeoff.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- Chart 6
def chart_6_eoq_sensitivity():
    df = pd.read_csv(os.path.join(OUT, "eoq_results.csv"))
    top3 = df.sort_values("cost_saving_vs_naive", ascending=False).head(3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (_, row) in zip(axes, top3.iterrows()):
        D = float(row["D_annual"])
        h = float(row["holding_cost_per_unit"])
        eoq = float(row["EOQ"])
        item = row["item_id"]

        qs = np.linspace(1, 3 * eoq, 400)
        ordering = (D / qs) * ORDERING_COST
        holding = (qs / 2.0) * h
        total = ordering + holding

        ax.plot(qs, total, color="#1f3864", linewidth=2.2, label="Total cost")
        ax.plot(qs, ordering, color="#5b9bd5", linewidth=1.2, linestyle=":", label="Ordering cost")
        ax.plot(qs, holding, color="#ed7d31", linewidth=1.2, linestyle=":", label="Holding cost")

        eoq_cost = (D / eoq) * ORDERING_COST + (eoq / 2.0) * h
        ax.axvline(eoq, color="#70ad47", linestyle="--", linewidth=2)
        ax.annotate(f"EOQ={eoq:.0f}\n${eoq_cost:,.0f}/yr", xy=(eoq, eoq_cost),
                    xytext=(10, 30), textcoords="offset points",
                    fontsize=9, color="#548235", fontweight="bold")

        naive_cost = (D / NAIVE_ORDER_QTY) * ORDERING_COST + (NAIVE_ORDER_QTY / 2.0) * h
        ax.axvline(NAIVE_ORDER_QTY, color="#c00000", linestyle="-.", linewidth=2)
        ax.annotate(f"Naive Q={NAIVE_ORDER_QTY}\n${naive_cost:,.0f}/yr",
                    xy=(NAIVE_ORDER_QTY, naive_cost), xytext=(12, 0),
                    textcoords="offset points", fontsize=9, color="#c00000",
                    fontweight="bold", va="center")

        # Cap y-axis so the U-shape is visible (the Q->1 spike is clipped).
        ax.set_ylim(0, naive_cost * 1.3)
        ax.set_xlim(0, 3 * eoq)
        ax.set_title(f"{item}  (D={D:,.0f}/yr)", fontsize=10)
        ax.set_xlabel("Order quantity (units)")
        ax.set_ylabel("Annual cost ($)")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("EOQ Cost Optimization — Top 3 Items by Saving\n"
                 "EOQ minimizes total cost; fixed orders are suboptimal",
                 fontsize=14, fontweight="bold")
    add_scope_note(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    path = os.path.join(OUT, "readme_chart_6_eoq_sensitivity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    charts = [
        chart_1_model_comparison,
        chart_2_abc_xyz_heatmap,
        chart_3_policy_comparison,
        chart_4_stockout_risk,
        chart_5_service_level_tradeoff,
        chart_6_eoq_sensitivity,
    ]
    for fn in charts:
        path = fn()
        print(f"  saved {os.path.basename(path)}")
    print("All 6 README charts saved to outputs/")


if __name__ == "__main__":
    main()
