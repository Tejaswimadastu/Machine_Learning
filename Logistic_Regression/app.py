import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Load model
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "churn_model.pkl")

model = joblib.load(model_path)

model = joblib.load("churn_model.pkl")

# Load dataset (for dashboard visuals)
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

# ================= TITLE =================
st.title("📊 Telco Customer Churn Prediction Dashboard")

# ================= DATASET PREVIEW =================
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ================= CHURN DISTRIBUTION =================
st.subheader("Churn Distribution")

fig1, ax1 = plt.subplots()
df["Churn"].value_counts().plot(kind="bar", ax=ax1)
ax1.set_xlabel("Churn (0 = Stay, 1 = Leave)")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# ================= CONFUSION MATRIX =================
st.subheader("Confusion Matrix (Model Performance)")

# Dummy evaluation (for dashboard view)
X_demo = np.random.rand(300, 3)
y_demo = np.random.randint(0, 2, 300)
y_pred_demo = model.predict(X_demo)

cm = confusion_matrix(y_demo, y_pred_demo)

fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# ================= METRICS =================
st.subheader("Model Performance Metrics")

accuracy = accuracy_score(y_demo, y_pred_demo)
col1, col2 = st.columns(2)

col1.metric("Accuracy", f"{accuracy*100:.2f}%")
col2.metric("Total Predictions", len(y_pred_demo))

# ================= PREDICTION SECTION =================
st.subheader("🔮 Predict Customer Churn")

col3, col4, col5 = st.columns(3)

with col3:
    monthly = st.number_input("Monthly Charges", min_value=0.0)
with col4:
    total = st.number_input("Total Charges", min_value=0.0)
with col5:
    tenure = st.number_input("Tenure (months)", min_value=0)

if st.button("Predict Churn"):
    input_data = np.array([[monthly, total, tenure]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ================= PATH HANDLING =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
df = pd.read_csv(os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

# ================= TITLE =================
st.title("📊 Telco Customer Churn Prediction Dashboard")

# ================= DATASET PREVIEW =================
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ================= CHURN DISTRIBUTION =================
st.subheader("Churn Distribution")

fig1, ax1 = plt.subplots()
df["Churn"].value_counts().plot(kind="bar", ax=ax1)
ax1.set_xlabel("Churn (0 = Stay, 1 = Leave)")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# ================= MODEL EVALUATION =================
st.subheader("Model Performance")

X = df[["MonthlyCharges", "TotalCharges", "tenure"]]
y = df["Churn"]

y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)

fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy*100:.2f}%")
col2.metric("Total Customers", len(y))

# ================= PREDICTION SECTION =================
st.subheader("🔮 Predict Customer Churn")

col3, col4, col5 = st.columns(3)

with col3:
    monthly = st.number_input("Monthly Charges", min_value=0.0)
with col4:
    total = st.number_input("Total Charges", min_value=0.0)
with col5:
    tenure = st.number_input("Tenure (months)", min_value=0)

if st.button("Predict Churn"):
    input_data = np.array([[monthly, total, tenure]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")
