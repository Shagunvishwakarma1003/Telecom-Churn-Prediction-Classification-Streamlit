Md

# Telecom Customer Churn Prediciton (Streamlit App)

A machine learning app to predict customer churn probability using an XGBoost model.

## project Overview

- This project builds a machine learning model to predict telecom customer churn using Logistic Regression, Random Forest, and XGBoost.
- XGBoost was selected as the final model based on  ROC-AUC and PR-AUC performance.
- Threshold tuning (0.35) was applied to improve churn recall for business retention strategy.

## Model Performance

- ROC: ~0.85
- PR-AUC: ~0.73
- Recall (after threshold tuning): 78%

## Application Preview

![App Sceenshot](app_screenshot.jpeg)

## Features Used

- AccountWeeks
- ContractRenewal
- DataPlan
- DataUsage
- CustServCalls
- DayMins
- DayCalls
- MonthlyCharge
- OverageFee
- RoamMins

## How to Run Locally

1. Install requirements:
'''bash
pip install -r requirements
2. (Optional) Generate model file (churn_model.pkl) by running the notebook: AnalysisBook_Telecom_churn_Classification_Project.ipynb
3. Run the Streamlit app:

Bash
streamlit run app.py

Notes:
* Threshold used: 0.35 (for higher churn recall)