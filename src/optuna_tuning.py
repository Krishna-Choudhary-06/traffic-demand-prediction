import optuna
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# =========================
# LOAD DATA
# =========================

train = pd.read_csv("data/train.csv")

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

target = "demand"

drop_cols = ["Index", "timestamp", "demand"]

features = [c for c in train.columns if c not in drop_cols]

X = train[features].copy()
y = train[target]

cat_cols = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather"
]

for col in cat_cols:
    X[col] = X[col].fillna("Unknown")

num_cols = X.select_dtypes(include=np.number).columns

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

# =========================
# OPTUNA OBJECTIVE
# =========================

def objective(trial):

    params = {
        "iterations": trial.suggest_int(
            "iterations",
            500,
            2500
        ),

        "depth": trial.suggest_int(
            "depth",
            5,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.1,
            log=True
        ),

        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            1,
            10
        ),

        "loss_function": "RMSE",

        "task_type": "GPU",

        "verbose": False
    }

    kf = KFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    scores = []

    for train_idx, valid_idx in kf.split(X):

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = CatBoostRegressor(**params)

        model.fit(
            X_train,
            y_train,
            cat_features=cat_cols
        )

        preds = model.predict(X_valid)

        rmse = np.sqrt(
            mean_squared_error(
                y_valid,
                preds
            )
        )

        scores.append(rmse)

    return np.mean(scores)

# =========================
# STUDY
# =========================

study = optuna.create_study(
    direction="minimize"
)

study.optimize(
    objective,
    n_trials=25
)

print("\nBEST SCORE:")
print(study.best_value)

print("\nBEST PARAMS:")
print(study.best_params)