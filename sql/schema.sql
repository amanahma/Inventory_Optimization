-- M5 Inventory Optimizer -- SQLite schema (Week 1, Task 4)
-- Star-schema: 3 dimension tables + 2 fact tables.

-- ============================ DIMENSIONS ============================
CREATE TABLE IF NOT EXISTS dim_date (
    date          TEXT PRIMARY KEY,
    d             TEXT,
    wm_yr_wk      INTEGER,
    day_of_week   INTEGER,
    week_of_year  INTEGER,
    month         INTEGER,
    year          INTEGER,
    event_name_1  TEXT,
    event_type_1  TEXT,
    event_name_2  TEXT,
    event_type_2  TEXT,
    snap_CA       INTEGER,
    snap_TX       INTEGER,
    snap_WI       INTEGER,
    is_weekend    INTEGER
);

CREATE TABLE IF NOT EXISTS dim_item (
    item_id  TEXT PRIMARY KEY,
    dept_id  TEXT,
    cat_id   TEXT
);

CREATE TABLE IF NOT EXISTS dim_store (
    store_id  TEXT PRIMARY KEY,
    state_id  TEXT
);

-- ============================== FACTS ===============================
CREATE TABLE IF NOT EXISTS fact_sales (
    item_id     TEXT,
    store_id    TEXT,
    date        TEXT,
    units_sold  INTEGER
);

CREATE TABLE IF NOT EXISTS fact_prices (
    item_id    TEXT,
    store_id   TEXT,
    wm_yr_wk   INTEGER,
    sell_price REAL
);

-- ============================= INDEXES ==============================
-- (Created AFTER bulk loading the fact tables -- see load_to_sql.py.
--  Building them on empty tables is instant; building them after the
--  load is far faster than maintaining them during 58M inserts.)
CREATE INDEX IF NOT EXISTS idx_fact_sales_item_store_date
    ON fact_sales (item_id, store_id, date);
-- wm_yr_wk appended so price lookups (date -> week -> price) are covered.
CREATE INDEX IF NOT EXISTS idx_fact_prices_item_store
    ON fact_prices (item_id, store_id, wm_yr_wk);
-- speeds the date -> wm_yr_wk join used by the revenue queries.
CREATE INDEX IF NOT EXISTS idx_dim_date_date
    ON dim_date (date);
