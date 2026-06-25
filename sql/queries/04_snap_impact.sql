-- 04_snap_impact.sql
-- FOODS items in California stores: average daily units sold on SNAP days
-- (snap_CA = 1) vs non-SNAP days (snap_CA = 0), broken out by department.
-- A conditional AVG pivots the two regimes onto one row per dept so the
-- percentage uplift is directly comparable.
SELECT
    di.dept_id                                                          AS dept_id,
    ROUND(AVG(CASE WHEN dd.snap_CA = 1 THEN fs.units_sold END), 4)      AS avg_sales_snap,
    ROUND(AVG(CASE WHEN dd.snap_CA = 0 THEN fs.units_sold END), 4)      AS avg_sales_nonsnap,
    ROUND(
        100.0 * (AVG(CASE WHEN dd.snap_CA = 1 THEN fs.units_sold END)
               - AVG(CASE WHEN dd.snap_CA = 0 THEN fs.units_sold END))
              / AVG(CASE WHEN dd.snap_CA = 0 THEN fs.units_sold END), 2) AS pct_uplift
FROM fact_sales fs
JOIN dim_date   dd ON dd.date = fs.date
JOIN dim_item   di ON di.item_id = fs.item_id
JOIN dim_store  ds ON ds.store_id = fs.store_id
WHERE di.cat_id = 'FOODS'
  AND ds.state_id = 'CA'
GROUP BY di.dept_id
ORDER BY di.dept_id;
