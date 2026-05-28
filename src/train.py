import pandas as pd
import numpy as np
import os

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =========================
# LOAD DATA
# =========================

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# =========================
# FEATURE ENGINEERING
# =========================
weather_map = {
    "Sunny": 0,
    "Foggy": 1,
    "Rainy": 2,
    "Snowy": 3
}


def process_time(df):
    df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)
    df["minute"] = df["timestamp"].str.split(":").str[1].astype(int)
    df["is_peak_hour"] = df["hour"].isin([7,8,9,17,18,19]).astype(int)
    df["is_weekend"] = (df["day"] % 7 >= 5).astype(int)
    df["weather_severity"] = df["Weather"].map(weather_map)
    
    return df

train = process_time(train)
test = process_time(test)

# =========================
# FEATURES & TARGET
# =========================

target = "demand"

drop_cols = ["Index", "timestamp", "demand"]

features = [col for col in train.columns if col not in drop_cols]

X = train[features].copy()
y = train[target].copy()

X_test = test[features]

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
# TRAIN VALIDATION SPLIT
# =========================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL
# =========================

model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    verbose=100
)

# =========================
# TRAIN MODEL
# =========================

model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100
)

# =========================
# VALIDATION
# =========================

preds = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, preds))

print("\nValidation RMSE:", rmse)

# =========================
# TRAIN FULL MODEL
# =========================

model.fit(
    X,
    y,
    cat_features=cat_cols
)

# =========================
# TEST PREDICTIONS
# =========================

test_preds = model.predict(X_test)

# =========================
# CREATE SUBMISSION FILE
# =========================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": test_preds
})

# =========================
# SAVE SUBMISSION
# =========================

os.makedirs("outputs/submissions", exist_ok=True)

submission_path = "outputs/submissions/submission.csv"

submission.to_csv(submission_path, index=False)

print("\nSubmission file created!")
print(submission_path)