# DemandSense – Intelligent Retail Demand Forecasting

## Project Overview

DemandSense is a machine learning-based retail demand forecasting system that predicts future product demand using historical sales data, store information, product families, promotions, and date-based features.

## Problem Statement

Retail businesses need accurate demand forecasts to manage inventory, reduce stockouts, avoid overstocking, and improve business planning. Traditional methods may not effectively capture patterns in large retail datasets.

## Solution

DemandSense uses Machine Learning models to analyze historical retail sales and predict future demand. The system provides an interactive Streamlit dashboard where users can select a store and product family, enter previous sales and promotion information, select a prediction date, and receive an estimated future demand.

## Models Used

- Linear Regression
- Random Forest
- XGBoost

Random Forest was selected for the dashboard based on its lower RMSE and higher R² compared with the other models.

## Key Features

- Retail sales data analysis
- Data preprocessing
- Date-based feature engineering
- Lag and rolling features
- Machine learning model training
- Model comparison
- Interactive demand prediction
- Store and product family selection
- Promotion-based forecasting
- Streamlit dashboard

## Dataset

The project uses the Store Sales - Time Series Forecasting dataset from Corporación Favorita.

Dataset features include:

- Store number
- Product family
- Sales
- Products on promotion
- Date

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Joblib
- Streamlit

## Project Structure

```text
DemandSense-Retail-Demand-Forecasting/
│
├── app.py
├── demand_forecasting.py
├── train_general_model.py
├── general_demand_model.pkl
├── family_encoder.pkl
├── forecast_results.csv
├── requirements.txt
└── README.md