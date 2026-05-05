import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.sidebar.title("Credit Risk Scoring Model Simulation")

st.sidebar.info("The __Credit Risk Scoring__ Model Used is __LightGBM__, optimized using Hyperparameter Tuning and Recursive Feature Extraction CV")

@st.cache_resource
def load_model():
    model = joblib.load('lgb_model.pkl')
    return model

model = load_model()

st.header("Please Insert The Following Value")
st.markdown("---")

emp_length_int = st.slider("Employment length in years", 0, 30)
mths_since_issue_d = st.slider("The number of months since the borrower's last loan issue date", 0, 250)
int_rate = st.slider("Interest rate of the loan application", 0, 40)
mths_since_earliest_cr_line = st.slider("The number of months since the borrower's first loan issue date", 0, 720)
inq_last_6mths = st.slider("Inq_Last_6mths The number of inquiries by borrowers during the past 6 months", 0, 30)
annual_inc = st.slider("The annual income provided by the borrower during registration", 0, 100000000)
dti = st.slider("A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested loan, divided by the borrower’s self-reported monthly income", 0, 100)
grade_G = st.select_slider("Acquired grade G of the loan", ['No', 'Yes'])
home_ownership_MORTGAGE = st.select_slider("The home ownership status provided by the borrower during registration is Mortgage", ['No', 'Yes'])
home_ownership_RENT = st.select_slider("The home ownership status provided by the borrower during registration is Rent", ['No', 'Yes'])
purpose_small_business = st.select_slider("A category provided by the borrower for the loan request is for small bussiness", ['No', 'Yes'])
term_36_months = st.select_slider("Credit term period (in months) is 36", ['No', 'Yes'])

inference_data = pd.DataFrame({
    'emp_length_int': [emp_length_int],
    'mths_since_issue_d': [mths_since_issue_d],
    'int_rate': [int_rate],
    'mths_since_earliest_cr_line': [mths_since_earliest_cr_line],
    'inq_last_6mths': [inq_last_6mths],
    'annual_inc': [annual_inc],
    'dti': [dti],
    'grade_G': [grade_G],
    'home_ownership_MORTGAGE': [home_ownership_MORTGAGE],
    'home_ownership_RENT': [home_ownership_RENT],
    'purpose_small_business': [purpose_small_business],
    'term_36_months': [term_36_months]
})

inference_data.replace('No', 0, inplace=True)
inference_data.replace('Yes', 1, inplace=True)

class_1_index = np.where(model.classes_ == 1)[0][0]
score = model.predict_proba(inference_data)[:, class_1_index][0]

st.markdown("---")
st.header("Your Default Probablity Score")
st.markdown(f'<p style="font-size:40px; font-weight:bold;">{score}</p>', unsafe_allow_html=True)