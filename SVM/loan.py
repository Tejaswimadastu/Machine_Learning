import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# -----------------------------
# App Title & Description
# -----------------------------
st.markdown(
    """
    <style>
    /* Main App Background */
.stApp {
    background: linear-gradient(135deg, #f8fafc, #e3f2fd);
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
h1 {
    color: #1f2937;
    text-align: center;
    font-weight: 700;
}

/* Subheaders */
h2, h3 {
    color: #374151;
    font-weight: 600;
}

/* Sidebar Background */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: #1f2937 !important;
}

/* Buttons */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 2.8em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
    border: none;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Input Boxes */
input, select {
    border-radius: 6px !important;
    padding: 6px;
    border: 1px solid #d1d5db;
}

/* Alert Boxes (Success / Error) */
div[data-testid="stAlert"] {
    border-radius: 10px;
    font-size: 16px;
}

/* Radio Buttons */
div[role="radiogroup"] > label {
    font-weight: 500;
    color: #111827;
}

    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Smart Loan Approval System", layout="centered")

st.title("💳 Smart Loan Approval System")
st.write(
    "This system uses **Support Vector Machines (SVM)** to predict whether a loan will be approved "
    "based on applicant financial details."
)

# -----------------------------
# Load & Prepare Data
# -----------------------------
@st.cache_data
def load_and_prepare_data():
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, "loan2.csv"))
  # make sure loan.csv is in same folder

    # Encode target
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # Select features
    df = df[['ApplicantIncome', 'LoanAmount', 'Credit_History',
             'Self_Employed', 'Property_Area', 'Loan_Status']]

    # Encode categorical
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Impute
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, imputer, scaler, X.columns


X_train, X_test, y_train, y_test, imputer, scaler, feature_names = load_and_prepare_data()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📋 Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=150)

credit = st.sidebar.radio("Credit History", ["Yes", "No"])
credit = 1 if credit == "Yes" else 0

employment = st.sidebar.selectbox("Employment Status", ["Yes", "No"])
employment = 1 if employment == "Yes" else 0

property_area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

# Property area encoding
prop_urban = 1 if property_area == "Urban" else 0
prop_semiurban = 1 if property_area == "Semiurban" else 0

# -----------------------------
# Model Selection
# -----------------------------
st.subheader("⚙️ Select SVM Kernel")

kernel = st.radio(
    "Choose SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

if kernel == "Linear SVM":
    model = SVC(kernel="linear", probability=True)
elif kernel == "Polynomial SVM":
    model = SVC(kernel="poly", degree=3, probability=True)
else:
    model = SVC(kernel="rbf", probability=True)

# Train selected model
model.fit(X_train, y_train)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Check Loan Eligibility"):
    user_data = np.array([[income, loan_amount, credit,
                            employment, prop_semiurban, prop_urban]])

    # Impute + Scale
    user_data = imputer.transform(user_data)
    user_data = scaler.transform(user_data)

    prediction = model.predict(user_data)[0]
    confidence = model.predict_proba(user_data)[0][prediction]

    # -----------------------------
    # Output Section
    # -----------------------------
    st.subheader("📊 Loan Decision")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Kernel Used:** {kernel}")
    st.write(f"**Model Confidence:** {confidence:.2%}")

    # -----------------------------
    # Business Explanation
    # -----------------------------
    st.subheader("📌 Decision Explanation")

    if prediction == 1:
        st.write(
            "Based on the applicant’s **credit history and income pattern**, "
            "the model predicts a **high likelihood of loan repayment**."
        )
    else:
        st.write(
            "Based on **credit risk and income pattern**, the model predicts "
            "a **lower likelihood of successful loan repayment**."
        )
