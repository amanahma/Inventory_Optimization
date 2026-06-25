-- 03_top_20_items_revenue.sql
-- Top 20 items by total revenue across all stores.
-- Revenue uses the weekly price mapped through dim_date (date -> wm_yr_wk).
SELECT
    di.item_id                                   AS item_id,
    di.dept_id                                   AS dept_id,
    di.cat_id                                    AS cat_id,
    SUM(fs.units_sold)                           AS total_units,
    ROUND(AVG(fp.sell_price), 2)                 AS avg_price,
    ROUND(SUM(fs.units_sold * fp.sell_price), 2) AS total_revenue
FROM fact_sales  fs
JOIN dim_date    dd ON dd.date = fs.date
JOIN dim_item    di ON di.item_id = fs.item_id
JOIN fact_prices fp ON fp.item_id = fs.item_id
                   AND fp.store_id = fs.store_id
                   AND fp.wm_yr_wk = dd.wm_yr_wk
GROUP BY di.item_id, di.dept_id, di.cat_id
ORDER BY total_revenue DESC
LIMIT 20;
