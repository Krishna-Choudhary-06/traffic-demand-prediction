import pandas as pd
import numpy as np
import os

from catboost import CatBoostRegressor
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

num_cols = X.select_dtypes(include=np.number).columns

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
    X_test[col] = X_test[col].fillna(X[col].median())

# =========================
# K-FOLD CV
# =========================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

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

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=200,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100
    )

    valid_preds = model.predict(X_valid)

    rmse = np.sqrt(
        mean_squared_error(y_valid, valid_preds)
    )

    print(f"Fold RMSE: {rmse}")

    scores.append(rmse)

    oof_preds[valid_idx] = valid_preds

    test_preds += model.predict(X_test) / 5

# =========================
# FINAL SCORE
# =========================

final_rmse = np.sqrt(
    mean_squared_error(y, oof_preds)
)

print("\n==============================")
print("FINAL CV RMSE:", final_rmse)
print("FOLD SCORES:", scores)
print("==============================")

# =========================
# FEATURE IMPORTANCE
# =========================

importance = model.get_feature_importance()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:\n")
print(feature_importance)

# =========================
# SUBMISSION
# =========================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": test_preds
})

os.makedirs(
    "outputs/submissions",
    exist_ok=True
)

submission_path = (
    "outputs/submissions/submission_cv.csv"
)

submission.to_csv(
    submission_path,
    index=False
)

print("\nSubmission Created!")
print(submission_path)
