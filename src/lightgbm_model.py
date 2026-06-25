"""
lightgbm_model.py  --  M5 Inventory Optimizer (Week 2, Task 5)

ONE global LightGBM regressor trained on ALL item-stores stacked together
(NOT one model per item). Engineered lag / rolling / price / calendar features,
label-encoded categoricals, MAE objective, early stopping on the time-based
validation horizon.

Leakage control: every lag is >= 28 days (= the forecast horizon), and rolling
features are computed on the already-shifted lag_28 column, so no validation row
ever sees information from inside its own 28-day horizon. Features are built on the
combined train+val timeline (so val rows can read their late-train history) and
then split strictly by the cutoff date.

Outputs:
  outputs/label_encoders.pkl
  outputs/lgbm_model.pkl
  outputs/lgbm_forecasts.csv
  outputs/lgbm_metrics.csv
  outputs/lgbm_feature_importance.png

NOTE: runs on the CA_1 sample (config.BASE_PARQUET); see config.DATA_SCOPE.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

import config as C

LAGS = [28, 35, 42, 56, 364]
FEATURES = [
    "lag_28", "lag_35", "lag_42", "lag_56", "lag_364",
    "roll_mean_28", "roll_mean_56", "roll_std_28", "roll_std_56",
    "sell_price", "price_change", "price_vs_mean",
    "day_of_week", "week_of_year", "month", "year",
    "is_weekend", "is_event", "is_snap",
    "item_id_enc", "store_id_enc", "dept_id_enc", "cat_id_enc", "state_id_enc",
]
TARGET = "units_sold"


def main():
    print("=" * 78)
    print(f"DATA SCOPE: {C.DATA_SCOPE}")
    print("TASK 5 - LightGBM global model (single model, all item-stores)")
    print("=" * 78)

    # ---- Step 1: load train + val prepared --------------------------------
    cols = ["item_id", "store_id", "dept_id", "cat_id", "state_id", "date",
            "units_sold", "sell_price", "price_change",
            "day_of_week", "week_of_year", "month", "year",
            "is_weekend", "is_event", "is_snap", "abc_xyz"]
    train = pd.read_parquet(C.TRAIN_PARQUET, columns=cols)
    val = pd.read_parquet(C.VAL_PARQUET, columns=cols)
    print(f"STEP 1: loaded train {train.shape}, val {val.shape}")

    # combine on one timeline so val rows can read their late-train history
    df = pd.concat([train, val], ignore_index=True)
    del train, val
    for c in ["item_id", "store_id", "dept_id", "cat_id", "state_id"]:
        df[c] = df[c].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df["units_sold"] = df["units_sold"].astype("float32")

    # ---- Step 2: feature engineering --------------------------------------
    df = df.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
    g = df.groupby(["item_id", "store_id"], observed=True)["units_sold"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag).astype("float32")
    print("STEP 2a: built lag features", LAGS)

    gl = df.groupby(["item_id", "store_id"], observed=True)["lag_28"]
    df["roll_mean_28"] = gl.transform(lambda x: x.rolling(28, min_periods=1).mean()).astype("float32")
    df["roll_mean_56"] = gl.transform(lambda x: x.rolling(56, min_periods=1).mean()).astype("float32")
    df["roll_std_28"] = gl.transform(lambda x: x.rolling(28, min_periods=1).std()).astype("float32")
    df["roll_std_56"] = gl.transform(lambda x: x.rolling(56, min_periods=1).std()).astype("float32")
    print("STEP 2b: built rolling features on lag_28")

    # price feature: current price vs the item-store mean price
    mean_price = df.groupby(["item_id", "store_id"], observed=True)["sell_price"].transform("mean")
    df["price_vs_mean"] = (df["sell_price"] / mean_price).astype("float32")
    df["sell_price"] = df["sell_price"].astype("float32")
    df["price_change"] = df["price_change"].astype("float32")
    print("STEP 2c: built price_vs_mean")

    # label-encode categoricals (fit on full combined data so all levels are seen)
    encoders = {}
    for c in ["item_id", "store_id", "dept_id", "cat_id", "state_id"]:
        le = LabelEncoder()
        df[f"{c}_enc"] = le.fit_transform(df[c]).astype("int32")
        encoders[c] = le
    with open(os.path.join(C.OUT, "label_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print("STEP 2d: label-encoded item_id, store_id, dept_id, cat_id, state_id "
          "-> saved label_encoders.pkl")

    # ---- Step 3: drop rows with any NaN lag feature -----------------------
    lag_cols = [f"lag_{l}" for l in LAGS]
    before = len(df)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    print(f"STEP 3: dropped {before - len(df):,} early rows lacking full lag history; "
          f"{len(df):,} rows remain")

    # ---- Step 4/5: features, target, time-based split ---------------------
    cutoff, _ = C.get_cutoff_date()
    train_df = df[df["date"] <= cutoff]
    val_df = df[df["date"] > cutoff].copy()
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]
    print(f"STEP 5: time-based split -> X_train {X_train.shape}, X_val {X_val.shape}")

    # ---- Step 6: train LightGBM -------------------------------------------
    params = {
        "objective": "regression_l1",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": C.RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = lgb.LGBMRegressor(**params)
    print("STEP 6: training LightGBM ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    print(f"  best iteration : {model.best_iteration_}")
    best_score = model.best_score_["valid_0"]["l1"]
    print(f"  best valid L1 (MAE): {best_score:.4f}")

    # ---- Step 7: predict + clip -------------------------------------------
    val_df["lgbm_forecast"] = np.clip(model.predict(X_val), 0, None)

    # ---- Step 8: metrics by category --------------------------------------
    rows = []
    for cat in C.CATEGORIES:
        sub = val_df[val_df["cat_id"] == cat]
        rows.append({
            "Category": cat,
            "RMSE": C.rmse(sub[TARGET], sub["lgbm_forecast"]),
            "MAE": C.mae(sub[TARGET], sub["lgbm_forecast"]),
            "MAPE": C.mape(sub[TARGET], sub["lgbm_forecast"]),
            "Bias": C.bias(sub[TARGET], sub["lgbm_forecast"]),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(C.OUT, "lgbm_metrics.csv"), index=False)
    print("\n" + "=" * 64)
    print("LIGHTGBM METRICS BY CATEGORY")
    print("=" * 64)
    print(f"{'Category':<10} | {'RMSE':>8} | {'MAE':>8} | {'MAPE':>9} | {'Bias':>8}")
    print("-" * 64)
    for _, r in metrics.iterrows():
        print(f"{r['Category']:<10} | {r['RMSE']:>8.4f} | {r['MAE']:>8.4f} | "
              f"{r['MAPE']:>9.4f} | {r['Bias']:>8.4f}")

    # ---- Step 9: feature importance ---------------------------------------
    imp = (pd.DataFrame({"feature": FEATURES,
                         "gain": model.booster_.feature_importance(importance_type="gain")})
           .sort_values("gain", ascending=False).reset_index(drop=True))
    print("\nTOP 15 FEATURES BY GAIN")
    print("-" * 40)
    for _, r in imp.head(15).iterrows():
        print(f"  {r['feature']:<16} {r['gain']:>14,.0f}")

    top = imp.head(15).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["gain"], color="steelblue")
    plt.xlabel("Importance (gain)")
    plt.title("LightGBM Feature Importance (top 15)")
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT, "lgbm_feature_importance.png"), dpi=120)
    plt.close()
    print("saved outputs/lgbm_feature_importance.png")

    # ---- Step 10: save model + forecasts ----------------------------------
    with open(os.path.join(C.OUT, "lgbm_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    out_cols = ["item_id", "store_id", "date", "cat_id", "abc_xyz",
                "units_sold", "lgbm_forecast"]
    fc = val_df[out_cols].rename(columns={"units_sold": "actual"})
    fc.to_csv(os.path.join(C.OUT, "lgbm_forecasts.csv"), index=False)
    print(f"saved outputs/lgbm_model.pkl, outputs/lgbm_forecasts.csv ({len(fc):,} rows), "
          f"outputs/lgbm_metrics.csv")
    print("DONE.")


if __name__ == "__main__":
    main()
