import streamlit as st
import joblib
import pandas as pd
model = joblib.load('churn_model.pkl')
st.title('Telecom Customer Churn Prediction')
st.write('Enter customer details:')
AccountWeeks = st.number_input('Account Weeks', min_value=0)
ContractRenewal = st.selectbox('Contract Renewal (0=No, 1=Yes)', [0, 1])
DataPlan = st.selectbox('Data Plan (0=No, 1=Yes)', [0, 1])
DataUsage = st.number_input('Data Usage (GB)', min_value=0.0)
CustServCalls = st.number_input('Customer Service Calls', min_value=0)
DayMins = st.number_input('Day Minutes', min_value=0.0)
DayCalls = st.number_input('Day Calls', min_value=0)
MonthlyCharge = st.number_input('Monthly Charge', min_value=0.0)
OverageFee = st.number_input('Overage Fee', min_value=0.0)
RoamMins = st.number_input('Roaming Minutes', min_value=0.0)

if st.button('Predict Churn'):
    input_df = pd.DataFrame([[
        AccountWeeks,
        ContractRenewal,
        DataPlan,
        DataUsage,
        CustServCalls,
        DayMins,
        DayCalls,
        MonthlyCharge,
        OverageFee,
        RoamMins
    ]],
    columns=[
        'AccountWeeks',
        'ContractRenewal',
        'DataPlan',
        'DataUsage',
        'CustServCalls',
        'DayMins',
        'DayCalls',
        'MonthlyCharge',
        'OverageFee',
        'RoamMins'
    ])

    prob = model.predict_proba(input_df)[0][1]
    st.progress(float(prob))
    st.subheader('Result:')
    st.metric('Churn probability:', f'{round(prob * 100, 2)}%')

    if prob > 0.35:
        st.error('High risk of Churn')
    else:
        st.success('Low risk of Churn')

st.markdown('---')
st.markdown('Model : XBGBoost | threshold : 0.35 | Developed by Shagun Vishwakarma')