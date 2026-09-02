import pandas as pd

# Load dataset
train = pd.read_csv("data/train.csv")

# Convert date to datetime
train["date"] = pd.to_datetime(train["date"])

# Remove duplicate rows
train = train.drop_duplicates()

# Remove invalid negative sales
train = train[train["sales"] >= 0]

# Sort data chronologically
train = train.sort_values("date")

# Reset index
train = train.reset_index(drop=True)

# Check the cleaned dataset
print("Dataset shape:", train.shape)
print("\nMissing values:")
print(train.isnull().sum())

print("\nData types:")
print(train.dtypes)

print("\nFirst 5 rows:")
print(train.head())
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Daily Sales Trend
daily_sales = train.groupby("date")["sales"].sum()

plt.figure(figsize=(14, 5))
plt.plot(daily_sales)
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# 2. Monthly Sales
train["month"] = train["date"].dt.month

monthly_sales = train.groupby("month")["sales"].sum()

plt.figure(figsize=(10, 5))
monthly_sales.plot(kind="bar")
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# 3. Sales by Product Family
family_sales = (
    train.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 7))
family_sales.plot(kind="bar")
plt.title("Sales by Product Family")
plt.xlabel("Product Family")
plt.ylabel("Total Sales")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 4. Promotion vs Average Sales
promotion_sales = train.groupby("onpromotion")["sales"].mean()

plt.figure(figsize=(8, 5))
promotion_sales.plot(kind="bar")
plt.title("Promotion vs Average Sales")
plt.xlabel("Number of Products on Promotion")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.show()
# =========================
# STEP 7 - DATE FEATURES
# =========================

train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["day_of_week"] = train["date"].dt.dayofweek
train["quarter"] = train["date"].dt.quarter

# 1 = Weekend, 0 = Weekday
train["is_weekend"] = (
    train["day_of_week"] >= 5
).astype(int)

print("\nDate features created successfully!")
print(train[
    [
        "date",
        "year",
        "month",
        "day",
        "day_of_week",
        "quarter",
        "is_weekend"
    ]
].head())
# ==========================================
# STEP 7 - DATE FEATURES
# ==========================================

train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["day_of_week"] = train["date"].dt.dayofweek
train["quarter"] = train["date"].dt.quarter
train["is_weekend"] = (train["day_of_week"] >= 5).astype(int)


# ==========================================
# STEP 8 - LAG FEATURES
# ==========================================

group_cols = ["store_nbr", "family"]

train["lag_1"] = train.groupby(group_cols)["sales"].shift(1)
train["lag_7"] = train.groupby(group_cols)["sales"].shift(7)
train["lag_14"] = train.groupby(group_cols)["sales"].shift(14)
train["lag_30"] = train.groupby(group_cols)["sales"].shift(30)


# ==========================================
# ROLLING FEATURES
# ==========================================

train["rolling_mean_7"] = (
    train.groupby(group_cols)["sales"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

train["rolling_mean_14"] = (
    train.groupby(group_cols)["sales"]
    .transform(lambda x: x.shift(1).rolling(14).mean())
)

train["rolling_mean_30"] = (
    train.groupby(group_cols)["sales"]
    .transform(lambda x: x.shift(1).rolling(30).mean())
)


print("\nFeature engineering completed!")

print("\nNew columns:")
print(train.columns.tolist())
# ==========================================
# STEP 9 - PREPARE DATA FOR MACHINE LEARNING
# ==========================================

# Select one store and one product family
store = 1
product_family = "BEVERAGES"

model_data = train[
    (train["store_nbr"] == store) &
    (train["family"] == product_family)
].copy()

# Remove rows where lag features are missing
model_data = model_data.dropna()

# Select features
features = [
    "onpromotion",
    "year",
    "month",
    "day",
    "day_of_week",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30"
]

X = model_data[features]
y = model_data["sales"]

# Time-based 80/20 split
split = int(len(model_data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nModel data prepared!")
print("Total records:", len(model_data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
# ==========================================
# STEP 10-13 - MACHINE LEARNING MODELS
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np
import time

# -------------------------------
# Linear Regression
# -------------------------------

print("\nTraining Linear Regression...")

lr = LinearRegression()

start = time.time()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_time = time.time() - start

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("Linear Regression completed!")


# -------------------------------
# Random Forest
# -------------------------------

print("\nTraining Random Forest...")

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

start = time.time()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_time = time.time() - start

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest completed!")


# -------------------------------
# XGBoost
# -------------------------------

print("\nTraining XGBoost...")

xgb = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

start = time.time()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

xgb_time = time.time() - start

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("XGBoost completed!")


# ==========================================
# MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],
    "MAE": [
        lr_mae,
        rf_mae,
        xgb_mae
    ],
    "RMSE": [
        lr_rmse,
        rf_rmse,
        xgb_rmse
    ],
    "R2": [
        lr_r2,
        rf_r2,
        xgb_r2
    ],
    "Time_seconds": [
        lr_time,
        rf_time,
        xgb_time
    ]
})

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(results.round(2))


# Best model based on RMSE
best_model_name = results.loc[
    results["RMSE"].idxmin(),
    "Model"
]

print("\nBest Model:", best_model_name)
# ==========================================
# STEP 15 - FINAL FORECAST
# ==========================================

# Use Random Forest as the final model
final_model = rf

# Predictions on test data
final_predictions = final_model.predict(X_test)

# Create result dataframe
forecast_results = pd.DataFrame({
    "Actual_Sales": y_test.values,
    "Predicted_Sales": final_predictions
})

print("\nForecast Results:")
print(forecast_results.head(10))

# ==========================================
# ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(14, 5))

plt.plot(
    y_test.values[:100],
    label="Actual Sales"
)

plt.plot(
    final_predictions[:100],
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Data Points")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
# ==========================================
# STEP 16 - SAVE FORECAST RESULTS
# ==========================================

forecast_results.to_csv(
    "forecast_results.csv",
    index=False
)

print("\nForecast results saved successfully!")
import joblib

# Save the trained Random Forest model
joblib.dump(rf, "random_forest_model.pkl")

print("Random Forest model saved!")