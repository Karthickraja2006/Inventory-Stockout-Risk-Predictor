# Inventory Stockout Risk Predictor

A Machine Learning project that predicts whether an inventory product is at risk of stockout based on inventory, demand, supplier, promotion, and seasonal factors.

## Project Overview

Inventory stockouts can affect product availability and business operations. This project uses machine learning classification models to identify whether a product is likely to experience a stockout.

The project includes data analysis, preprocessing, model training, model comparison, and a Streamlit-based prediction application.

## Objectives

- Analyze inventory and demand-related data
- Perform exploratory data analysis (EDA)
- Preprocess categorical and numerical features
- Train machine learning classification models
- Compare model performance
- Predict stockout risk for new inventory inputs
- Provide an interactive web interface using Streamlit

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Machine Learning Models

The project uses:

- Logistic Regression
- Decision Tree Classifier

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Input Features

The prediction application uses inventory-related features such as:

- Product Category
- Season
- Current Stock
- Daily Sales Rate
- Historical Demand (30 Days)
- Forecast Demand (30 Days)
- Supplier Lead Time
- In-Transit Quantity
- Supplier Reliability
- Promotion Status
- Seasonality Index

## Project Structure

```text
Inventory-Stockout-Risk-Predictor/
│
├── app.py
├── model.py
├── preprocess.py
├── eda.py
├── inventory_stockout_risk_dataset.csv
├── logistic_regression_model.pkl
├── decision_tree_model.pkl
└── feature_columns.pkl
