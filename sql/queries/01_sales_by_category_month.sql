-- 01_sales_by_category_month.sql
-- Total units sold and total revenue (units x sell_price) by category and month+year.
-- Revenue requires the weekly price, so fact_sales is mapped date -> wm_yr_wk via
-- dim_date, then joined to fact_prices on (item_id, store_id, wm_yr_wk).
SELECT
    di.cat_id                              AS cat_id,
    dd.year                                AS year,
    dd.month                               AS month,
    SUM(fs.units_sold)                     AS total_units,
    ROUND(SUM(fs.units_sold * fp.sell_price), 2) AS total_revenue
FROM fact_sales  fs
JOIN dim_date    dd ON dd.date = fs.date
JOIN dim_item    di ON di.item_id = fs.item_id
JOIN fact_prices fp ON fp.item_id = fs.item_id
                   AND fp.store_id = fs.store_id
                   AND fp.wm_yr_wk = dd.wm_yr_wk
GROUP BY di.cat_id, dd.year, dd.month
ORDER BY di.cat_id, dd.year, dd.month;
