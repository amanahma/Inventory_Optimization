"""
app.py  --  M5 Dark-Store Inventory Optimizer  (Streamlit dashboard)

Interactive dashboard over the OR inventory-optimization outputs.

Data scope: Walmart M5 dataset, CA_1 store only (3,049 item-store combinations).
All cost parameters are standard industry assumptions — M5 has no real cost data.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
import os

st.set_page_config(
    page_title="M5 Inventory Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Resolve paths relative to this file so the app runs from any working dir.
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")


def opath(name):
    return os.path.join(OUT, name)


# ---------------------------------------------------------------- data loading
@st.cache_data
def load_data():
    main_df = pd.read_csv(opath("powerbi_main_dashboard.csv"))
    inv_df = pd.read_csv(opath("powerbi_inventory_policy.csv"))
    nv_df = pd.read_csv(opath("powerbi_newsvendor.csv"))
    pulp_df = pd.read_csv(opath("powerbi_pulp_scenarios.csv"))
    date_df = pd.read_csv(opath("powerbi_dim_date.csv"))
    model_df = pd.read_csv(opath("model_comparison_final.csv"))
    with open(opath("project_summary_stats.json"), encoding="utf-8") as f:
        stats_json = json.load(f)
    return main_df, inv_df, nv_df, pulp_df, date_df, model_df, stats_json


main_df, inv_df, nv_df, pulp_df, date_df, model_df, stats = load_data()


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.image(opath("readme_chart_2_abc_xyz_heatmap.png"),
             caption="ABC-XYZ Segmentation", use_container_width=True)

    st.markdown("## Filters")

    # Filter 1: ABC Class
    abc_filter = st.multiselect(
        "Item Class (ABC)",
        options=["A", "B", "C"],
        default=["A", "B", "C"]
    )

    # Filter 2: Category
    cat_filter = st.multiselect(
        "Product Category",
        options=["FOODS", "HOBBIES", "HOUSEHOLD"],
        default=["FOODS", "HOBBIES", "HOUSEHOLD"]
    )

    # Filter 3: Stockout Risk
    risk_filter = st.radio(
        "Stockout Risk",
        options=["All Items", "At Risk Only", "Safe Only"],
        index=0
    )

    st.markdown("---")
    st.markdown("### Project Info")
    st.markdown("**Dataset:** Walmart M5")
    st.markdown("**Scope:** CA_1 Store")
    st.markdown("**Items:** 3,049 item-store pairs")
    st.markdown("**Period:** 2011–2016")
    st.markdown("---")
    st.markdown("**⚠️ Cost Disclaimer**")
    st.markdown(
        "M5 has no real cost data. "
        "All cost parameters are standard industry assumptions: "
        "holding cost 20%, ordering cost $5, lead time 7 days."
    )

# Apply filters to main_df
filtered_df = main_df[main_df["abc_class"].isin(abc_filter)]
filtered_df = filtered_df[filtered_df["cat_id"].isin(cat_filter)]

if risk_filter == "At Risk Only":
    filtered_df = filtered_df[filtered_df["stockout_risk"] == 1]
elif risk_filter == "Safe Only":
    filtered_df = filtered_df[filtered_df["stockout_risk"] == 0]


# --------------------------------------------------------------- main header
st.title("📦 M5 Dark-Store Inventory Optimizer")
st.markdown(
    "**Demand forecasting + OR-based inventory optimization** "
    "on Walmart M5 dataset | CA_1 Store | 3,049 item-store combinations"
)
st.markdown("---")

# Guard: empty selection
if filtered_df.empty:
    st.warning(
        "No items match the current sidebar filters. "
        "Widen the ABC, Category, or Stockout Risk selection to see results."
    )
    st.stop()


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Executive Summary",
    "🎯 Forecast Accuracy",
    "🗂️ ABC-XYZ Segmentation",
    "📋 Inventory Policy",
    "⚠️ Stockout Risk",
    "💰 Cost Analysis",
    "🔧 LP Budget Scenarios"
])


# =====================================================================  TAB 1
with tab1:
    st.header("Executive Summary")
    st.caption("Key performance indicators across all filtered items")

    # Row 1: 4 KPI metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_items = len(filtered_df)
        st.metric(
            label="Total Items Analyzed",
            value=f"{total_items:,}"
        )

    with col2:
        total_saving = filtered_df["annual_saving"].sum()
        st.metric(
            label="Total Annual Saving",
            value=f"${total_saving:,.0f}"
        )

    with col3:
        naive_cost = filtered_df["total_annual_cost_naive"].sum()
        opt_cost = filtered_df["total_annual_cost_EOQ"].sum()
        reduction = ((naive_cost - opt_cost) / naive_cost * 100) if naive_cost > 0 else 0
        st.metric(
            label="Cost Reduction vs Naive",
            value=f"{reduction:.1f}%"
        )

    with col4:
        at_risk = (filtered_df["stockout_risk"] == 1).sum()
        risk_pct = at_risk / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.metric(
            label="Items at Stockout Risk",
            value=f"{at_risk:,}",
            delta=f"{risk_pct:.1f}% of items",
            delta_color="inverse"
        )

    st.markdown("---")

    # Row 2: Cost comparison bar chart + Safety stock chart side by side
    col_left, col_right = st.columns(2)

    with col_left:
        cost_by_cat = filtered_df.groupby("cat_id").agg(
            Naive_Cost=("total_annual_cost_naive", "sum"),
            Optimized_Cost=("total_annual_cost_EOQ", "sum")
        ).reset_index()

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            name="Naive Policy",
            x=cost_by_cat["cat_id"],
            y=cost_by_cat["Naive_Cost"],
            marker_color="#EF4444"
        ))
        fig1.add_trace(go.Bar(
            name="Optimized Policy",
            x=cost_by_cat["cat_id"],
            y=cost_by_cat["Optimized_Cost"],
            marker_color="#22C55E"
        ))
        fig1.update_layout(
            title="Annual Cost: Naive vs Optimized by Category",
            xaxis_title="Category",
            yaxis_title="Annual Cost ($)",
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=350
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        ss_by_abc = filtered_df.groupby("abc_class").agg(
            Avg_Safety_Stock=("safety_stock", "mean"),
            Avg_EOQ=("EOQ", "mean")
        ).reset_index()

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Avg Safety Stock",
            x=ss_by_abc["abc_class"],
            y=ss_by_abc["Avg_Safety_Stock"],
            marker_color="#3B82F6"
        ))
        fig2.update_layout(
            title="Average Safety Stock by ABC Class",
            xaxis_title="ABC Class",
            yaxis_title="Units",
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Row 3: Summary insight box
    st.info(
        f"**Key Insight:** EOQ-based optimization reduces annual inventory cost by "
        f"**{reduction:.1f}%** compared to a naive fixed-order policy. "
        f"This saving is driven by matching order quantities to actual demand "
        f"and sizing safety stock using forecast error (not raw demand variability). "
        f"**{at_risk}** items are currently below their reorder point and face stockout risk."
    )


# =====================================================================  TAB 2
with tab2:
    st.header("Forecast Accuracy")
    st.caption(
        "Model comparison: Seasonal Naive baseline vs LightGBM global model "
        "vs Croston SBA for intermittent (Z-class) SKUs. "
        "Time-based train/validation split — no random shuffling."
    )

    # Top: show the pre-generated chart image
    st.image(
        opath("readme_chart_1_model_comparison.png"),
        caption="RMSE comparison across models and categories",
        use_container_width=True
    )

    st.markdown("---")

    # Interactive version using model_comparison_final.csv
    if not model_df.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            # RMSE grouped bar by category and model
            if "Metric" in model_df.columns:
                rmse_df = model_df[model_df["Metric"] == "RMSE"]
            else:
                rmse_df = model_df

            if not rmse_df.empty and "RMSE" in rmse_df.columns:
                fig3 = px.bar(
                    rmse_df,
                    x="Category",
                    y="RMSE",
                    color="Model",
                    barmode="group",
                    title="RMSE by Model and Category",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)

        with col_right:
            st.subheader("Detailed Metrics Table")
            st.dataframe(
                model_df,
                use_container_width=True,
                height=400
            )

    # Key callout boxes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**LightGBM Global Model**\nOne model trained on all SKUs simultaneously using lag features, rolling stats, price, and calendar signals.")
    with col2:
        st.warning("**Croston / SBA**\nApplied to Z-class items (CV > 1.0). ~60% of M5 items have intermittent demand — standard ARIMA fails here.")
    with col3:
        st.info("**Validation Rule**\nAlways time-based split. Train: d_1 to d_1885. Validate: d_1886 to d_1913. No random shuffling.")


# =====================================================================  TAB 3
with tab3:
    st.header("ABC-XYZ Segmentation")
    st.caption(
        "ABC = revenue contribution (A=top 70%, B=next 20%, C=bottom 10%). "
        "XYZ = demand variability (X=CV<0.5, Y=CV 0.5-1.0, Z=CV>1.0). "
        "Segmentation drives which forecasting method is applied."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Pivot table for heatmap
        pivot = filtered_df.groupby(
            ["abc_class", "xyz_class"]
        ).size().reset_index(name="count")
        pivot_wide = pivot.pivot(
            index="abc_class", columns="xyz_class", values="count"
        ).fillna(0)

        # Ensure correct order
        for col in ["X", "Y", "Z"]:
            if col not in pivot_wide.columns:
                pivot_wide[col] = 0
        for idx in ["A", "B", "C"]:
            if idx not in pivot_wide.index:
                pivot_wide.loc[idx] = 0

        pivot_wide = pivot_wide[["X", "Y", "Z"]].loc[["A", "B", "C"]]

        fig4 = go.Figure(data=go.Heatmap(
            z=pivot_wide.values,
            x=["X (Stable)", "Y (Variable)", "Z (Intermittent)"],
            y=["A (High Value)", "B (Medium Value)", "C (Low Value)"],
            colorscale="Blues",
            text=pivot_wide.values.astype(int),
            texttemplate="%{text}",
            textfont={"size": 16, "color": "black"},
            showscale=True
        ))
        fig4.update_layout(
            title="Item Count by ABC-XYZ Segment",
            xaxis_title="Demand Variability (XYZ)",
            yaxis_title="Revenue Class (ABC)",
            height=400
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_right:
        # Bar chart of items per segment
        seg_count = filtered_df.groupby("abc_xyz").size().reset_index(name="count")
        seg_count = seg_count.sort_values("count", ascending=True)

        fig5 = px.bar(
            seg_count,
            x="count",
            y="abc_xyz",
            orientation="h",
            title="Items per ABC-XYZ Segment",
            color="count",
            color_continuous_scale="Blues"
        )
        fig5.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # Model assignment table
    st.subheader("Segment → Forecasting Method Mapping")
    mapping_data = {
        "Segment": ["AX", "AY", "BX", "BY", "CX", "CY", "AZ", "BZ", "CZ"],
        "Description": [
            "High value, stable", "High value, variable",
            "Med value, stable", "Med value, variable",
            "Low value, stable", "Low value, variable",
            "High value, intermittent", "Med value, intermittent",
            "Low value, intermittent"
        ],
        "Forecasting Model": [
            "LightGBM", "LightGBM",
            "LightGBM", "LightGBM",
            "ETS Baseline", "ETS Baseline",
            "Croston SBA", "Croston SBA", "Croston SBA"
        ],
        "Inventory Policy": [
            "EOQ + SS (98%)", "EOQ + SS (98%)",
            "EOQ + SS (95%)", "EOQ + SS (95%)",
            "EOQ + SS (90%)", "EOQ + SS (90%)",
            "Newsvendor (if FOODS)", "Newsvendor (if FOODS)",
            "Simple ROP"
        ]
    }
    st.dataframe(pd.DataFrame(mapping_data), use_container_width=True)


# =====================================================================  TAB 4
with tab4:
    st.header("Inventory Policy")
    st.caption(
        "Per-item inventory decisions: safety stock, EOQ, reorder point, "
        "days of inventory on hand, and turnover ratio."
    )

    # Top KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_ss = filtered_df["safety_stock"].mean()
        st.metric("Avg Safety Stock", f"{avg_ss:.1f} units")
    with col2:
        avg_eoq = filtered_df["EOQ"].mean()
        st.metric("Avg EOQ", f"{avg_eoq:.1f} units")
    with col3:
        avg_rop = filtered_df["ROP"].mean()
        st.metric("Avg Reorder Point", f"{avg_rop:.1f} units")
    with col4:
        avg_turn = filtered_df["turnover"].mean()
        st.metric("Avg Inventory Turnover", f"{avg_turn:.1f}x")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        # Safety stock by ABC and category
        ss_cat = filtered_df.groupby(["cat_id", "abc_class"]).agg(
            Avg_SS=("safety_stock", "mean")
        ).reset_index()

        fig6 = px.bar(
            ss_cat,
            x="cat_id",
            y="Avg_SS",
            color="abc_class",
            barmode="group",
            title="Average Safety Stock by Category and ABC Class",
            color_discrete_map={"A": "#EF4444", "B": "#F59E0B", "C": "#22C55E"},
            labels={"Avg_SS": "Avg Safety Stock (units)", "cat_id": "Category"}
        )
        fig6.update_layout(height=350)
        st.plotly_chart(fig6, use_container_width=True)

    with col_right:
        # DOH by category
        doh_cat = filtered_df.groupby("cat_id").agg(
            Avg_DOH=("DOH", "mean"),
            Avg_Turnover=("turnover", "mean")
        ).reset_index()

        fig7 = go.Figure()
        fig7.add_trace(go.Bar(
            name="Days of Inventory (DOH)",
            x=doh_cat["cat_id"],
            y=doh_cat["Avg_DOH"],
            marker_color="#3B82F6"
        ))
        fig7.update_layout(
            title="Average Days of Inventory on Hand by Category",
            xaxis_title="Category",
            yaxis_title="Days",
            height=350
        )
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown("---")

    # Detailed item-level table
    st.subheader("Item-Level Inventory Policy Table")
    st.caption("Use sidebar filters to narrow down. Click column headers to sort.")

    display_cols = [
        "item_id", "cat_id", "dept_id", "abc_class", "xyz_class",
        "safety_stock", "EOQ", "ROP", "DOH", "turnover",
        "stockout_risk", "sell_price", "forecast_mean"
    ]

    # Only show columns that exist
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[display_cols].round(2),
        use_container_width=True,
        height=400
    )


# =====================================================================  TAB 5
with tab5:
    st.header("Stockout Risk Analysis")
    st.caption(
        "Items flagged at stockout risk: current estimated inventory "
        "is below the calculated reorder point. A-class items at risk "
        "are the highest business priority."
    )

    at_risk_df = filtered_df[filtered_df["stockout_risk"] == 1]
    safe_df = filtered_df[filtered_df["stockout_risk"] == 0]

    # Top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Items at Risk",
            f"{len(at_risk_df):,}",
            delta=f"{len(at_risk_df)/len(filtered_df)*100:.1f}% of items",
            delta_color="inverse"
        )
    with col2:
        a_risk = len(at_risk_df[at_risk_df["abc_class"] == "A"])
        st.metric(
            "A-Class Items at Risk",
            f"{a_risk:,}",
            delta="Highest Priority",
            delta_color="inverse"
        )
    with col3:
        avg_doh_risk = at_risk_df["DOH"].mean() if len(at_risk_df) > 0 else 0
        st.metric(
            "Avg DOH (At-Risk Items)",
            f"{avg_doh_risk:.1f} days"
        )

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        # Risk by ABC class
        risk_abc = filtered_df.groupby("abc_class").agg(
            At_Risk=("stockout_risk", "sum"),
            Total=("stockout_risk", "count")
        ).reset_index()
        risk_abc["Risk_Pct"] = risk_abc["At_Risk"] / risk_abc["Total"] * 100

        color_map = {"A": "#EF4444", "B": "#F59E0B", "C": "#22C55E"}
        fig8 = go.Figure(go.Bar(
            x=risk_abc["Risk_Pct"],
            y=risk_abc["abc_class"],
            orientation="h",
            marker_color=[color_map.get(c, "gray") for c in risk_abc["abc_class"]],
            text=[f"{v:.1f}%" for v in risk_abc["Risk_Pct"]],
            textposition="outside"
        ))
        fig8.update_layout(
            title="Stockout Risk % by ABC Class",
            xaxis_title="% Items at Risk",
            yaxis_title="ABC Class",
            height=300
        )
        st.plotly_chart(fig8, use_container_width=True)

    with col_right:
        # Risk by category
        risk_cat = filtered_df.groupby("cat_id").agg(
            At_Risk=("stockout_risk", "sum"),
            Total=("stockout_risk", "count")
        ).reset_index()
        risk_cat["Risk_Pct"] = risk_cat["At_Risk"] / risk_cat["Total"] * 100

        fig9 = px.pie(
            risk_cat,
            values="At_Risk",
            names="cat_id",
            title="Items at Stockout Risk by Category",
            color_discrete_sequence=["#EF4444", "#F59E0B", "#3B82F6"]
        )
        fig9.update_layout(height=300)
        st.plotly_chart(fig9, use_container_width=True)

    st.markdown("---")

    # High priority table: A-class at risk
    st.subheader("⚠️ A-Class Items at Stockout Risk — Priority Order")
    a_risk_df = filtered_df[
        (filtered_df["abc_class"] == "A") &
        (filtered_df["stockout_risk"] == 1)
    ].copy()

    if len(a_risk_df) > 0:
        priority_cols = [
            c for c in [
                "item_id", "cat_id", "safety_stock", "ROP",
                "DOH", "EOQ", "sell_price", "forecast_mean"
            ] if c in a_risk_df.columns
        ]
        st.dataframe(
            a_risk_df[priority_cols].sort_values(
                "DOH" if "DOH" in a_risk_df.columns else priority_cols[0]
            ).round(2),
            use_container_width=True,
            height=350
        )
    else:
        st.success("No A-class items at stockout risk in current filter selection.")


# =====================================================================  TAB 6
with tab6:
    st.header("Cost Analysis")
    st.caption(
        "Comparing naive fixed-order policy vs EOQ-optimized policy. "
        "Cost components: ordering cost, holding cost, stockout cost."
    )

    # Show pre-generated chart
    if os.path.exists(opath("readme_chart_3_policy_comparison.png")):
        st.image(
            opath("readme_chart_3_policy_comparison.png"),
            caption="Policy cost comparison by category",
            use_container_width=True
        )

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        # Saving by category
        saving_cat = filtered_df.groupby("cat_id").agg(
            Total_Saving=("annual_saving", "sum")
        ).reset_index()

        fig10 = px.bar(
            saving_cat,
            x="cat_id",
            y="Total_Saving",
            title="Total Annual Saving by Category",
            color="cat_id",
            color_discrete_sequence=["#22C55E", "#3B82F6", "#F59E0B"],
            labels={"Total_Saving": "Annual Saving ($)", "cat_id": "Category"}
        )
        fig10.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig10, use_container_width=True)

    with col_right:
        # Scatter: forecast error vs safety stock (joined from inv_df)
        scatter_src = filtered_df
        if "forecast_error_std" not in scatter_src.columns and "forecast_error_std" in inv_df.columns:
            scatter_src = filtered_df.merge(
                inv_df[["item_id", "store_id", "forecast_error_std"]],
                on=["item_id", "store_id"], how="left"
            )

        if "forecast_error_std" in scatter_src.columns:
            sample = scatter_src.sample(min(500, len(scatter_src)))
            fig11 = px.scatter(
                sample,
                x="forecast_error_std",
                y="safety_stock",
                color="abc_class",
                size="sell_price" if "sell_price" in sample.columns else None,
                title="Forecast Error vs Safety Stock Requirement",
                color_discrete_map={
                    "A": "#EF4444", "B": "#F59E0B", "C": "#22C55E"
                },
                labels={
                    "forecast_error_std": "Forecast Error Std",
                    "safety_stock": "Safety Stock (units)"
                },
                opacity=0.6
            )
            fig11.update_layout(height=350)
            st.plotly_chart(fig11, use_container_width=True)
        else:
            st.image(
                opath("readme_chart_5_service_level_tradeoff.png"),
                caption="Service Level vs Safety Stock Cost",
                use_container_width=True
            )

    st.markdown("---")

    # Summary table
    st.subheader("Cost Summary by Category")
    cost_summary = filtered_df.groupby("cat_id").agg(
        Naive_Cost=("total_annual_cost_naive", "sum"),
        Optimized_Cost=("total_annual_cost_EOQ", "sum"),
        Total_Saving=("annual_saving", "sum"),
        Avg_Saving_Pct=("annual_saving_pct", "mean")
    ).round(2).reset_index()
    cost_summary["Saving_%"] = (
        cost_summary["Total_Saving"] / cost_summary["Naive_Cost"] * 100
    ).round(1)
    st.dataframe(cost_summary, use_container_width=True)

    st.info(
        "**Why does forecasting accuracy affect cost?** "
        "Safety stock is sized using forecast *error* standard deviation — "
        "not raw demand variability. A more accurate model produces smaller "
        "residuals, which means smaller safety stock, which means lower "
        "holding cost — at the same service level guarantee."
    )


# =====================================================================  TAB 7
with tab7:
    st.header("LP Budget Scenarios")
    st.caption(
        "PuLP linear programming: given a fixed procurement budget, "
        "allocate optimally across A+B class items to maximize demand fulfilled. "
        "CBC solver used. All 3 scenarios solve to Optimal status."
    )

    st.warning(
        "**Scope note:** At CA_1 single-store scale, A+B class items cost "
        "only ~$12,636 to fully stock — below even the tight $50K budget. "
        "All 3 scenarios reach 100% fill rate (budget constraint non-binding). "
        "At full 10-store scale with real procurement budgets, "
        "the trade-off becomes material."
    )

    if not pulp_df.empty:
        # Scenario summary
        scenario_summary = pulp_df.groupby("scenario").agg(
            Total_Demand=("demand_mean", "sum"),
            Total_Fulfilled=("fulfilled", "sum"),
            Total_Spent=("budget_spent", "sum"),
            Items=("item_id", "count")
        ).reset_index()
        scenario_summary["Fill_Rate_%"] = (
            scenario_summary["Total_Fulfilled"] /
            scenario_summary["Total_Demand"] * 100
        ).round(1)

        # Display summary table
        st.subheader("Scenario Comparison")
        st.dataframe(scenario_summary.round(2), use_container_width=True)

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            # Grouped bar: demand vs fulfilled per scenario
            fig12 = go.Figure()
            fig12.add_trace(go.Bar(
                name="Total Demand",
                x=scenario_summary["scenario"],
                y=scenario_summary["Total_Demand"],
                marker_color="#EF4444"
            ))
            fig12.add_trace(go.Bar(
                name="Total Fulfilled",
                x=scenario_summary["scenario"],
                y=scenario_summary["Total_Fulfilled"],
                marker_color="#22C55E"
            ))
            fig12.update_layout(
                title="Demand vs Fulfilled by Budget Scenario",
                xaxis_title="Budget Scenario",
                yaxis_title="Units",
                barmode="group",
                height=350
            )
            st.plotly_chart(fig12, use_container_width=True)

        with col_right:
            # Fill rate line chart
            fig13 = go.Figure()
            fig13.add_trace(go.Scatter(
                x=scenario_summary["scenario"],
                y=scenario_summary["Fill_Rate_%"],
                mode="lines+markers+text",
                text=[f"{v:.1f}%" for v in scenario_summary["Fill_Rate_%"]],
                textposition="top center",
                line=dict(color="#3B82F6", width=3),
                marker=dict(size=10)
            ))
            fig13.update_layout(
                title="Fill Rate by Budget Scenario",
                xaxis_title="Budget Scenario",
                yaxis_title="Fill Rate (%)",
                yaxis=dict(range=[0, 110]),
                height=350
            )
            st.plotly_chart(fig13, use_container_width=True)

    st.markdown("---")
    st.subheader("LP Formulation")
    st.code("""
Maximize:  Σ fulfilled[i]       for all items i

Subject to:
  Σ Q[i] × price[i] ≤ budget   (budget constraint)
  fulfilled[i] ≤ Q[i]           (cannot fulfill more than ordered)
  fulfilled[i] ≤ demand[i]      (cannot fulfill more than demanded)
  Q[i] ≥ 0, fulfilled[i] ≥ 0   (non-negativity)

Solver: PuLP CBC (open source)
Variables: Q[i] = order quantity, fulfilled[i] = demand met
    """, language="text")


# ------------------------------------------------------------------- footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Data: Walmart M5 Dataset (Kaggle) | Scope: CA_1 Store Only | "
    "Cost parameters are industry assumptions — M5 contains no real cost data | "
    "Built by Aman — IIT Kharagpur, ISE 2024 batch"
    "</div>",
    unsafe_allow_html=True
)
