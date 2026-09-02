import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="DemandSense",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📈 DemandSense")

st.subheader("Intelligent Retail Demand Forecasting")

st.write(
    "Predict future retail demand using historical sales, "
    "store information, product family and promotion data."
)

# --------------------------------------------------
# LOAD GENERAL MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("general_demand_model.pkl")


model = load_model()

# --------------------------------------------------
# AVAILABLE STORES AND PRODUCT FAMILIES
# --------------------------------------------------

stores = list(range(1, 55))

families = [
    "AUTOMOTIVE",
    "BABY CARE",
    "BEAUTY",
    "BEVERAGES",
    "BOOKS",
    "BREAD/BAKERY",
    "CELEBRATION",
    "CLEANING",
    "DAIRY",
    "DELI",
    "EGGS",
    "FROZEN FOODS",
    "GROCERY I",
    "GROCERY II",
    "HARDWARE",
    "HOME AND KITCHEN I",
    "HOME AND KITCHEN II",
    "HOME APPLIANCES",
    "HOME CARE",
    "LADIESWEAR",
    "LAWN AND GARDEN",
    "LINGERIE",
    "LIQUOR,WINE,BEER",
    "MAGAZINES",
    "MEATS",
    "PERSONAL CARE",
    "PET SUPPLIES",
    "PLAYERS AND ELECTRONICS",
    "POULTRY",
    "PREPARED FOODS",
    "PRODUCE",
    "SCHOOL AND OFFICE SUPPLIES",
    "SEAFOOD"
]

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.markdown("---")

st.header("📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", "3,000,888")
col2.metric("Number of Stores", "54")
col3.metric("Product Families", "33")
col4.metric("Date Range", "2013 - 2017")

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.markdown("---")

st.header("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Selected Model", "Random Forest")
col2.metric("RMSE", "446.39")
col3.metric("R² Score", "0.46")

# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------

st.markdown("---")

st.header("🔮 Predict Future Demand")

st.write(
    "Enter the latest sales information below and "
    "predict the expected future demand."
)

# --------------------------------------------------
# STORE AND PRODUCT FAMILY
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    store = st.selectbox(
        "🏪 Select Store",
        stores
    )

with col2:
    family = st.selectbox(
        "🛒 Select Product Family",
        families
    )

# --------------------------------------------------
# SALES AND PROMOTION
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    previous_sales = st.number_input(
        "💰 Previous Sales",
        min_value=0.0,
        value=1800.0,
        step=100.0
    )

with col2:
    promotion = st.number_input(
        "📢 Products on Promotion",
        min_value=0,
        value=10,
        step=1
    )

# --------------------------------------------------
# PREDICTION DATE
# --------------------------------------------------

prediction_date = st.date_input(
    "📅 Prediction Date"
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 Predict Future Demand", type="primary"):

    date = pd.to_datetime(prediction_date)

    # Create input row
    input_data = pd.DataFrame({
        "store_nbr": [store],
        "family": [family],
        "year": [date.year],
        "month": [date.month],
        "day": [date.day],
        "day_of_week": [date.dayofweek],
        "quarter": [date.quarter],
        "is_weekend": [1 if date.dayofweek >= 5 else 0],
        "onpromotion": [promotion],
        "sales": [previous_sales],
        "lag_1": [previous_sales],
        "lag_7": [previous_sales],
        "lag_14": [previous_sales],
        "lag_30": [previous_sales],
        "rolling_mean_7": [previous_sales],
        "rolling_mean_14": [previous_sales],
        "rolling_mean_30": [previous_sales]
    })

    # --------------------------------------------------
    # MAKE MODEL INPUT COMPATIBLE
    # --------------------------------------------------

    try:

        if hasattr(model, "feature_names_in_"):

            required_columns = list(model.feature_names_in_)

            final_input = pd.DataFrame(index=[0])

            for col in required_columns:

                if col in input_data.columns:

                    final_input[col] = input_data[col].values

                elif col.startswith("family_"):

                    family_name = col.replace("family_", "")

                    final_input[col] = [
                        1 if family == family_name else 0
                    ]

                else:

                    final_input[col] = 0

        else:

            final_input = input_data

        # --------------------------------------------------
        # PREDICT
        # --------------------------------------------------

        prediction = model.predict(final_input)[0]

        prediction = max(0, prediction)

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.success("✅ Demand prediction completed!")

        st.markdown("---")

        st.header("🎯 Predicted Future Demand")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Expected Sales",
            f"{prediction:,.0f} units"
        )

        result_col2.metric(
            "Selected Store",
            f"Store {store}"
        )

        result_col3.metric(
            "Product Family",
            family
        )

        st.info(
            f"📊 Based on the entered information, "
            f"the estimated future demand is "
            f"**{prediction:,.0f} units**."
        )

    except Exception as e:

        st.error("Prediction could not be generated.")

        st.write("Please check the model input format.")

        st.code(str(e))

# --------------------------------------------------
# SALES TREND
# --------------------------------------------------

st.markdown("---")

st.header("📊 Sales Trend")

st.write(
    "The original retail dataset contains daily sales records "
    "from 2013 to 2017 across 54 stores and 33 product families."
)

trend_data = pd.DataFrame({
    "Year": [2013, 2014, 2015, 2016, 2017],
    "Dataset Period": [
        "Jan - Dec",
        "Jan - Dec",
        "Jan - Dec",
        "Jan - Dec",
        "Jan - Aug"
    ]
})

st.dataframe(
    trend_data,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.markdown("---")

st.header("🤖 Machine Learning Model Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ],
    "MAE": [
        340.02,
        320.04,
        311.89
    ],
    "RMSE": [
        487.14,
        446.39,
        463.51
    ],
    "R²": [
        0.36,
        0.46,
        0.42
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# RMSE GRAPH
# --------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.bar(
    comparison["Model"],
    comparison["RMSE"]
)

ax2.set_xlabel("Model")
ax2.set_ylabel("RMSE")
ax2.set_title("RMSE Comparison of Machine Learning Models")

st.pyplot(fig2)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "DemandSense | Intelligent Retail Demand Forecasting "
    "Using Machine Learning"
)
