import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("svm_loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Approval Prediction System")

st.write("Enter applicant details to get instant loan decision")

# ---- User Inputs ----
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
app_income = st.number_input("Applicant Income", min_value=0)
coapp_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", value=360)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---- Encoding (same as training) ----
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Feature vector
input_data = np.array([[gender, married, dependents, education,
                        self_employed, app_income, coapp_income,
                        loan_amount, loan_term, credit_history,
                        property_area]])

# Scale
input_scaled = scaler.transform(input_data)

# ---- Prediction ----
if st.button("Check Loan Status"):
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled).max()

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Confidence:** {confidence*100:.2f}%")
