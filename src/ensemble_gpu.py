import pandas as pd
import numpy as np
import os

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# =========================
# LOAD DATA
# =========================

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# =========================
# FEATURE ENGINEERING
# =========================

def process_time(df):

    df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)

    df["minute"] = df["timestamp"].str.split(":").str[1].astype(int)

    df["is_peak_hour"] = df["hour"].isin(
        [7,8,9,17,18,19]
    ).astype(int)

    df["is_weekend"] = (
        df["day"] % 7 >= 5
    ).astype(int)

    return df

train = process_time(train)
test = process_time(test)

# =========================
# FEATURES
# =========================

target = "demand"

drop_cols = ["Index", "timestamp", "demand"]

features = [c for c in train.columns if c not in drop_cols]

X = train[features].copy()
y = train[target]

X_test = test[features].copy()

# =========================
# CATEGORICAL COLUMNS
# =========================

cat_cols = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather"
]

# =========================
# HANDLE MISSING VALUES
# =========================

for col in cat_cols:
    X[col] = X[col].fillna("Unknown")
    X_test[col] = X_test[col].fillna("Unknown")

# convert categorical to string
for col in cat_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# label encode for LightGBM
for col in cat_cols:

    all_vals = pd.concat([
        X[col],
        X_test[col]
    ]).unique()

    mapping = {
        k:v for v,k in enumerate(all_vals)
    }

    X[col] = X[col].map(mapping)
    X_test[col] = X_test[col].map(mapping)

num_cols = X.select_dtypes(include=np.number).columns

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
    X_test[col] = X_test[col].fillna(X[col].median())

# =========================
# K-FOLD
# =========================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cat_oof = np.zeros(len(X))
lgb_oof = np.zeros(len(X))

cat_test = np.zeros(len(X_test))
lgb_test = np.zeros(len(X_test))

scores = []

# =========================
# TRAINING
# =========================

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):

    print(f"\n========== FOLD {fold+1} ==========")

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # =====================
    # CATBOOST GPU
    # =====================

    cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=200,
        task_type="GPU",
        devices='0'
    )

    cat_model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        verbose=200
    )

    cat_valid = cat_model.predict(X_valid)
    cat_test_preds = cat_model.predict(X_test)

    # =====================
    # LIGHTGBM
    # =====================

    lgb_model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        objective="regression",
        device="gpu"
    )

    lgb_model.fit(
        X_train,
        y_train
    )

    lgb_valid = lgb_model.predict(X_valid)
    lgb_test_preds = lgb_model.predict(X_test)

    # =====================
    # ENSEMBLE
    # =====================

    final_valid = (
        0.5 * cat_valid +
        0.5 * lgb_valid
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            final_valid
        )
    )

    print(f"Fold RMSE: {rmse}")

    scores.append(rmse)

    cat_oof[valid_idx] = cat_valid
    lgb_oof[valid_idx] = lgb_valid

    cat_test += cat_test_preds / 5
    lgb_test += lgb_test_preds / 5

# =========================
# FINAL ENSEMBLE
# =========================

final_oof = (
    0.5 * cat_oof +
    0.5 * lgb_oof
)

final_test = (
    0.5 * cat_test +
    0.5 * lgb_test
)

final_rmse = np.sqrt(
    mean_squared_error(
        y,
        final_oof
    )
)

print("\n==============================")
print("FINAL CV RMSE:", final_rmse)
print("FOLD SCORES:", scores)
print("==============================")

# =========================
# SUBMISSION
# =========================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": final_test
})

os.makedirs(
    "outputs/submissions",
    exist_ok=True
)

submission_path = (
    "outputs/submissions/submission_ensemble_gpu.csv"
)

submission.to_csv(
    submission_path,
    index=False
)

print("\nSubmission Created!")
print(submission_path)