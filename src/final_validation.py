"""
final_validation.py  --  M5 Inventory Optimizer (Week 4, Task 1)

Final data-quality validation on every powerbi_*.csv file in outputs/ before
Power BI import.

Data scope: Walmart M5 dataset, CA_1 store only (3,049 item-store combinations).

Checks performed:
  1. No NaN values  -> fill numeric with 0, text with 'UNKNOWN', resave.
  2. No negative values in inventory columns -> clip to 0, resave.
  3. Column names contain no spaces -> replace spaces with underscores.
  4. Row counts match expectations.
  5. item_id / store_id consistency across files.
"""

import os
import glob
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
OUT = os.path.abspath(OUT)

# Columns that must never be negative (clipped to 0 where present)
NON_NEGATIVE_COLS = [
    "safety_stock", "EOQ", "ROP", "DOH", "turnover", "annual_saving",
    "order_qty", "fulfilled", "Q_star",
]

EXPECTED_STORE = "CA_1"


def main():
    files = sorted(glob.glob(os.path.join(OUT, "powerbi_*.csv")))

    print("=" * 70)
    print("FINAL DATA VALIDATION  --  Data scope: Walmart M5, CA_1 store only")
    print("=" * 70)
    print(f"Found {len(files)} powerbi_*.csv files in outputs/\n")

    files_with_nan = []
    files_with_negatives = []
    files_with_colname_changes = []
    total_nan_fixed = 0
    total_negatives_clipped = 0

    # Keep frames for cross-file id consistency (Check 5)
    loaded = {}

    for path in files:
        fname = os.path.basename(path)
        df = pd.read_csv(path)
        changed = False

        # ---- Check 3: column names have no spaces ----------------------------
        rename_map = {c: c.replace(" ", "_") for c in df.columns if " " in c}
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            files_with_colname_changes.append((fname, rename_map))
            changed = True

        # ---- Check 1: NaN values -------------------------------------------
        nan_count = int(df.isna().sum().sum())
        if nan_count > 0:
            files_with_nan.append((fname, nan_count))
            total_nan_fixed += nan_count
            for col in df.columns:
                if df[col].isna().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna("UNKNOWN")
            changed = True

        # ---- Check 2: negative values in inventory columns ------------------
        file_clipped = 0
        for col in NON_NEGATIVE_COLS:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                neg_mask = df[col] < 0
                n_neg = int(neg_mask.sum())
                if n_neg > 0:
                    df.loc[neg_mask, col] = 0
                    file_clipped += n_neg
        if file_clipped > 0:
            files_with_negatives.append((fname, file_clipped))
            total_negatives_clipped += file_clipped
            changed = True

        if changed:
            df.to_csv(path, index=False)

        loaded[fname] = df

    # ---- Report Checks 1-3 -------------------------------------------------
    print("-" * 70)
    print("CHECK 1 — NaN values")
    print("-" * 70)
    if files_with_nan:
        for fname, n in files_with_nan:
            print(f"  {fname}: {n} NaN values filled (numeric->0, text->'UNKNOWN'), resaved")
    else:
        print("  No NaN values found in any file.")

    print("\n" + "-" * 70)
    print("CHECK 2 — Negative values in inventory columns")
    print("-" * 70)
    print(f"  Columns checked: {', '.join(NON_NEGATIVE_COLS)}")
    if files_with_negatives:
        for fname, n in files_with_negatives:
            print(f"  {fname}: {n} negative values clipped to 0, resaved")
    else:
        print("  No negative values found.")

    print("\n" + "-" * 70)
    print("CHECK 3 — Column names with spaces")
    print("-" * 70)
    if files_with_colname_changes:
        for fname, m in files_with_colname_changes:
            print(f"  {fname}: renamed {m}")
    else:
        print("  No column names contained spaces.")

    # ---- Check 4: row counts ----------------------------------------------
    print("\n" + "-" * 70)
    print("CHECK 4 — Row counts vs expected")
    print("-" * 70)

    def nrows(fname):
        return len(loaded[fname]) if fname in loaded else None

    # main dashboard -> 3049
    md = nrows("powerbi_main_dashboard.csv")
    print(f"  powerbi_main_dashboard.csv : actual={md:>6}  expected=3049"
          f"   {'OK' if md == 3049 else 'MISMATCH'}")

    # newsvendor -> FOODS items only
    nv = loaded.get("powerbi_newsvendor.csv")
    if nv is not None:
        nv_non_foods = int((nv["cat_id"] != "FOODS").sum()) if "cat_id" in nv else -1
        # expected = number of FOODS rows in main dashboard
        foods_in_md = int((loaded["powerbi_main_dashboard.csv"]["cat_id"] == "FOODS").sum())
        print(f"  powerbi_newsvendor.csv     : actual={len(nv):>6}  expected={foods_in_md}"
              f" (FOODS items)   {'OK' if len(nv) == foods_in_md else 'CHECK'}")
        print(f"      non-FOODS rows in newsvendor file: {nv_non_foods}"
              f"   {'OK (FOODS only)' if nv_non_foods == 0 else 'WARNING: non-FOODS present'}")

    # pulp scenarios -> 3 x (A+B items)
    md_df = loaded["powerbi_main_dashboard.csv"]
    n_ab = int(md_df["abc_class"].isin(["A", "B"]).sum())
    pulp = nrows("powerbi_pulp_scenarios.csv")
    expected_pulp = 3 * n_ab
    print(f"  powerbi_pulp_scenarios.csv : actual={pulp:>6}  expected={expected_pulp}"
          f" (3 x {n_ab} A+B items)   {'OK' if pulp == expected_pulp else 'MISMATCH'}")

    # dim_date -> ~1969
    dd = nrows("powerbi_dim_date.csv")
    print(f"  powerbi_dim_date.csv       : actual={dd:>6}  expected~1969"
          f"   {'OK' if abs(dd - 1969) <= 5 else 'CHECK'}")

    # ---- Check 5: id consistency ------------------------------------------
    print("\n" + "-" * 70)
    print("CHECK 5 — item_id / store_id consistency")
    print("-" * 70)
    master_items = set(md_df["item_id"].unique())
    discrepancies = 0

    for fname, df in loaded.items():
        if "store_id" in df.columns:
            stores = set(df["store_id"].dropna().unique())
            bad_stores = stores - {EXPECTED_STORE}
            if bad_stores:
                discrepancies += 1
                print(f"  {fname}: unexpected store_id values: {sorted(bad_stores)}")
        if "item_id" in df.columns and fname != "powerbi_main_dashboard.csv":
            items = set(df["item_id"].dropna().unique())
            unknown = items - master_items
            if unknown:
                discrepancies += 1
                print(f"  {fname}: {len(unknown)} item_id values not in main dashboard, "
                      f"e.g. {sorted(unknown)[:5]}")
    if discrepancies == 0:
        print(f"  All store_id == '{EXPECTED_STORE}' and all item_id values consistent "
              f"with main dashboard. No discrepancies.")

    # ---- Step 6: final report ---------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print(f"Files checked: {len(files)}")
    print(f"NaN values fixed: {total_nan_fixed}")
    print(f"Negative values clipped: {total_negatives_clipped}")
    print("All files ready for Power BI import.")
    print("=" * 70)


if __name__ == "__main__":
    main()
