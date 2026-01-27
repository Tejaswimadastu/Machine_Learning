import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ================= LOAD DATA ==================
df = pd.read_csv("credit_risk_data.csv")

# Handle missing values
df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)

# Encode categorical
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= STREAMLIT UI ==================

st.set_page_config(page_title="Customer Risk Prediction System", layout="wide")

# ---------- HEADER ----------
st.title("💳 Customer Risk Prediction System (KNN)")
st.write("This system predicts customer risk by comparing them with similar customers.")

# ---------- SIDEBAR ----------
st.sidebar.header("🔹 Customer Input")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Annual Income", 1000, 200000, 50000)
loan_amount = st.sidebar.number_input("Loan Amount", 1000, 100000, 10000)
credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])

k_value = st.sidebar.slider("K Value (Number of Neighbors)", 1, 15, 5)

# Convert credit history
credit_history = 1 if credit_history == "Yes" else 0

# ---------- MODEL ----------
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_scaled, y)

# ---------- BUTTON ----------
if st.button("🚀 Predict Customer Risk"):

    # Prepare input (MATCH COLUMN ORDER)
    input_data = np.array([[age, income, loan_amount, credit_history]])

    # Add missing dummy columns if needed
    input_df = pd.DataFrame(input_data, columns=["person_age","person_income","loan_amnt","cb_person_default_on_file_Y"])
    
    # Fill missing columns with 0
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X.columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = knn.predict(input_scaled)[0]

    # ---------- OUTPUT ----------
    st.subheader("🔮 Prediction Result")

    if prediction == 1:
        st.error("🔴 High Risk Customer")
    else:
        st.success("🟢 Low Risk Customer")

    # ---------- NEAREST NEIGHBORS ----------
    st.subheader("👥 Nearest Neighbors Explanation")

    distances, indices = knn.kneighbors(input_scaled, n_neighbors=k_value)
    neighbors = df.iloc[indices[0]]

    st.write(f"Number of Neighbors Used: {k_value}")
    st.write("Majority Class Among Neighbors:", neighbors["loan_status"].mode()[0])

    st.dataframe(neighbors[["person_age","person_income","loan_amnt","loan_status"]])

    # ---------- BUSINESS INSIGHT ----------
    st.subheader("📊 Business Insight")
    st.write("This decision is based on similarity with nearby customers in feature space. "
             "KNN compares customer attributes and assigns risk based on majority behavior of similar past customers.")
