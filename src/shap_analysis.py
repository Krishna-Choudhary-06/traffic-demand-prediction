import pandas as pd
import shap

from catboost import CatBoostRegressor

# load data
train = pd.read_csv("data/train.csv")

# same preprocessing as best model

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

drop_cols = ["Index", "timestamp", "demand"]

features = [c for c in train.columns if c not in drop_cols]

X = train[features].copy()

cat_cols = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather"
]

for col in cat_cols:
    X[col] = X[col].fillna("Unknown")

# load model
model = CatBoostRegressor()
model.load_model("outputs/models/best_model.cbm")

# sample for speed
sample = X.sample(1000, random_state=42)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(sample)

shap.summary_plot(
    shap_values,
    sample,
    show=True
)   