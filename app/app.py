# Loan Approval Prediction Streamlit App
import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load('loan_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

st.title("Loan Approval Prediction")

def encode_input(data):
    encoded = []
    encoded.append(label_encoders['Gender'].transform([data['Gender']])[0])
    encoded.append(label_encoders['Married'].transform([data['Married']])[0])
    encoded.append(data['Dependents'])
    encoded.append(label_encoders['Education'].transform([data['Education']])[0])
    encoded.append(label_encoders['Self_Employed'].transform([data['Self_Employed']])[0])
    encoded.append(data['ApplicantIncome'])
    encoded.append(data['CoapplicantIncome'])
    encoded.append(data['LoanAmount'])
    encoded.append(data['Loan_Amount_Term'])
    encoded.append(data['Credit_History'])
    encoded.append(label_encoders['Property_Area'].transform([data['Property_Area']])[0])
    return [encoded]

# User Input
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

if st.button("Predict Loan Approval"):
    input_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }
    encoded_input = encode_input(input_data)
    prediction = model.predict(encoded_input)
    result = label_encoders['Loan_Status'].inverse_transform(prediction)[0]
    st.success(f"Loan Status: {'Approved ✅' if result == 'Y' else 'Not Approved ❌'}")
