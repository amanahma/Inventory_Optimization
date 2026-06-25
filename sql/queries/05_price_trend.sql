-- 05_price_trend.sql
-- Average sell_price by category by year, with the year-over-year change so the
-- direction of the price trend across the 5 years is visible.
-- wm_yr_wk is mapped to a calendar year via a distinct week->year lookup to avoid
-- fanning each weekly price row out across the 7 days of that week.
WITH wk AS (
    SELECT DISTINCT wm_yr_wk, year FROM dim_date
),
cat_year AS (
    SELECT
        di.cat_id                  AS cat_id,
        wk.year                    AS year,
        ROUND(AVG(fp.sell_price), 4) AS avg_price,
        COUNT(*)                   AS num_price_points
    FROM fact_prices fp
    JOIN dim_item di ON di.item_id = fp.item_id
    JOIN wk        ON wk.wm_yr_wk = fp.wm_yr_wk
    GROUP BY di.cat_id, wk.year
)
SELECT
    cat_id,
    year,
    avg_price,
    num_price_points,
    ROUND(avg_price - LAG(avg_price) OVER (PARTITION BY cat_id ORDER BY year), 4)
                                       AS yoy_change,
    ROUND(100.0 * (avg_price - LAG(avg_price) OVER (PARTITION BY cat_id ORDER BY year))
                 / LAG(avg_price) OVER (PARTITION BY cat_id ORDER BY year), 2)
                                       AS yoy_pct_change
FROM cat_year
ORDER BY cat_id, year;
