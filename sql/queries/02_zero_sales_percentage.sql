-- 02_zero_sales_percentage.sql
-- Per (item_id, store_id): total days, days with zero sales, and the zero-sale
-- percentage. Ordered by the highest share of zero-demand days first.
SELECT
    item_id,
    store_id,
    COUNT(*)                                              AS total_days,
    SUM(CASE WHEN units_sold = 0 THEN 1 ELSE 0 END)       AS zero_days,
    ROUND(100.0 * SUM(CASE WHEN units_sold = 0 THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                          AS zero_percentage
FROM fact_sales
GROUP BY item_id, store_id
ORDER BY zero_percentage DESC
LIMIT 1000;
