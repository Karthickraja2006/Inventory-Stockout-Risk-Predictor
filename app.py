import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Inventory Stockout Risk Predictor",
    page_icon="📦",
    layout="centered"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("logistic_regression_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# =========================================================
# PAGE TITLE
# =========================================================

st.title("📦 Inventory Stockout Risk Predictor")

st.write(
    "Predict whether an inventory item is at risk of stockout "
    "using machine learning."
)

st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("Enter Inventory Details")


# Category
category = st.selectbox(
    "Product Category",
    [
        "Clothing",
        "Electronics",
        "Grocery",
        "Home & Kitchen",
        "Personal Care",
        "Sports"
    ]
)


# Season
season = st.selectbox(
    "Season",
    [
        "Autumn",
        "Spring",
        "Summer",
        "Winter"
    ]
)


# Current Stock
current_stock = st.number_input(
    "Current Stock",
    min_value=0,
    max_value=1000,
    value=100
)


# Daily Sales Rate
sales_rate_daily = st.number_input(
    "Daily Sales Rate",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.1
)


# Historical Demand
historical_demand_30d = st.number_input(
    "Historical Demand (30 Days)",
    min_value=0,
    max_value=1000,
    value=100
)


# Forecast Demand
forecast_demand_30d = st.number_input(
    "Forecast Demand (30 Days)",
    min_value=0,
    max_value=1000,
    value=100
)


# Lead Time
lead_time_days = st.number_input(
    "Supplier Lead Time (Days)",
    min_value=1,
    max_value=60,
    value=10
)


# In Transit Quantity
in_transit_qty = st.number_input(
    "In-Transit Quantity",
    min_value=0,
    max_value=1000,
    value=20
)


# Supplier Reliability
supplier_reliability = st.slider(
    "Supplier Reliability",
    min_value=0.50,
    max_value=1.00,
    value=0.90,
    step=0.01
)


# Promotion
promotion_active = st.selectbox(
    "Promotion Active?",
    ["No", "Yes"]
)


# Seasonality Index
seasonality_index = st.number_input(
    "Seasonality Index",
    min_value=0.50,
    max_value=2.00,
    value=1.00,
    step=0.01
)


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔍 Predict Stockout Risk",
    use_container_width=True
):

    # Convert promotion to numeric
    promotion_value = 1 if promotion_active == "Yes" else 0


    # Create input dataframe
    input_data = pd.DataFrame({

        "current_stock": [current_stock],

        "sales_rate_daily": [sales_rate_daily],

        "historical_demand_30d": [
            historical_demand_30d
        ],

        "forecast_demand_30d": [
            forecast_demand_30d
        ],

        "lead_time_days": [
            lead_time_days
        ],

        "in_transit_qty": [
            in_transit_qty
        ],

        "supplier_reliability": [
            supplier_reliability
        ],

        "promotion_active": [
            promotion_value
        ],

        "seasonality_index": [
            seasonality_index
        ]
    })


    # Add category columns
    for cat in [
        "category_Electronics",
        "category_Grocery",
        "category_Home & Kitchen",
        "category_Personal Care",
        "category_Sports"
    ]:

        input_data[cat] = 0


    # Set selected category
    category_column = f"category_{category}"

    if category_column in input_data.columns:
        input_data[category_column] = 1


    # Add season columns
    for s in [
        "season_Spring",
        "season_Summer",
        "season_Winter"
    ]:

        input_data[s] = 0


    # Set selected season
    season_column = f"season_{season}"

    if season_column in input_data.columns:
        input_data[season_column] = 1


    # Make sure columns are in the same order
    # as the training dataset
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # =====================================================
    # MAKE PREDICTION
    # =====================================================

    prediction = model.predict(input_data)[0]


    # Prediction probability
    probability = model.predict_proba(
        input_data
    )[0][1]


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.divider()

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "🔴 HIGH STOCKOUT RISK"
        )

        st.write(
            f"The model estimates a "
            f"{probability * 100:.2f}% probability "
            f"of stockout risk."
        )

        st.warning(
            "Consider reviewing stock levels, "
            "sales demand and replenishment time."
        )

    else:

        st.success(
            "🟢 LOW STOCKOUT RISK"
        )

        st.write(
            f"The model estimates a "
            f"{probability * 100:.2f}% probability "
            f"of stockout risk."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Machine Learning Project — Inventory Stockout Risk Classification"
)