import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

print("Loading dataset...")

train = pd.read_csv("data/train.csv")
train["date"] = pd.to_datetime(train["date"])

# ------------------------------------------
# Monthly aggregation
# ------------------------------------------

print("Preparing monthly data...")

train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month

monthly = (
    train.groupby(
        ["store_nbr", "family", "year", "month"],
        as_index=False
    )
    .agg(
        sales=("sales", "sum"),
        onpromotion=("onpromotion", "sum")
    )
)

monthly["date"] = pd.to_datetime(
    monthly["year"].astype(str)
    + "-"
    + monthly["month"].astype(str)
    + "-01"
)

monthly = monthly.sort_values(
    ["store_nbr", "family", "date"]
)

# ------------------------------------------
# Lag features
# ------------------------------------------

group = monthly.groupby(["store_nbr", "family"])["sales"]

monthly["lag_1"] = group.shift(1)
monthly["lag_2"] = group.shift(2)
monthly["lag_3"] = group.shift(3)

monthly["rolling_mean_3"] = (
    monthly.groupby(["store_nbr", "family"])["sales"]
    .transform(lambda x: x.shift(1).rolling(3).mean())
)

monthly = monthly.dropna()

# ------------------------------------------
# Encode product family
# ------------------------------------------

encoder = LabelEncoder()

monthly["family_encoded"] = encoder.fit_transform(
    monthly["family"]
)

# ------------------------------------------
# Features
# ------------------------------------------

features = [
    "store_nbr",
    "family_encoded",
    "year",
    "month",
    "onpromotion",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3"
]

X = monthly[features]
y = monthly["sales"]

print("Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# ------------------------------------------
# Save model
# ------------------------------------------

joblib.dump(model, "general_demand_model.pkl")
joblib.dump(encoder, "family_encoder.pkl")

print()
print("===================================")
print("GENERAL MODEL TRAINING COMPLETED")
print("===================================")
print("Training records:", len(monthly))
print("Stores:", monthly["store_nbr"].nunique())
print("Product families:", monthly["family"].nunique())
print("Model saved successfully!")