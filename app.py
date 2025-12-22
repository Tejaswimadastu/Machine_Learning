import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================= PAGE CONFIG =================
st.set_page_config("Multiple Linear Regression - Car Price", layout="centered")

# ================= BACKGROUND IMAGE =================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
            linear-gradient(rgba(255,255,255,0.65), rgba(255,255,255,0.65)),
            url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            min-height: 100vh;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("bg.jpg")   # make sure bg.jpg is in same folder

# ================= LOAD CSS =================
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ================= TITLE =================
st.markdown("""
<div class="card">
    <h1>🚗 Multiple Linear Regression</h1>
    <p>Predict <b>Car Selling Price</b> using <b>Multiple Features</b></p>
</div>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("car_details.csv")

df = load_data()

# Drop non-useful column
df.drop(columns=["name"], inplace=True)

# One-hot encode categorical columns
df = pd.get_dummies(
    df,
    columns=["fuel", "seller_type", "transmission", "owner"],
    drop_first=True
)

# ================= DATA PREVIEW =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview (Encoded)")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ================= PREPARE DATA =================
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= TRAIN MODEL =================
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ================= METRICS =================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)

# ================= VISUALIZATION (WITH LR LINE) =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Year vs Selling Price (with Regression Line)")

fig, ax = plt.subplots()

# Scatter plot
ax.scatter(df["year"], df["selling_price"], alpha=0.4, label="Actual Data")

# Prepare regression line (vary year, keep others mean)
X_line = X.copy()
for col in X_line.columns:
    X_line[col] = X[col].mean()

X_line["year"] = df["year"]
X_line_scaled = scaler.transform(X_line)
y_line = model.predict(X_line_scaled)

# Regression line
ax.plot(df["year"], y_line, color="red", linewidth=2, label="Regression Line")

ax.set_xlabel("Year")
ax.set_ylabel("Selling Price")
ax.legend()

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# ================= PERFORMANCE =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:,.0f}")
c2.metric("RMSE", f"{rmse:,.0f}")

c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.3f}")
c4.metric("Adjusted R²", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# ================= MODEL INTERPRETATION =================
st.markdown(
    f"""
    <div class="card">
        <h3>Model Interpretation</h3>
        <p>
            <b>Intercept:</b> {model.intercept_:,.2f}<br>
            <b>Total Features Used:</b> {X.shape[1]}
        </p>
        <p>
            Car selling price depends on <b>year</b>, <b>kilometers driven</b>,
            <b>fuel type</b>, <b>seller type</b>, <b>transmission</b>, and <b>ownership</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ================= PREDICTION =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Car Price (Sample Input)")

year = st.slider("Manufacturing Year", int(df["year"].min()), int(df["year"].max()), 2018)
km = st.slider("Kilometers Driven", int(df["km_driven"].min()), int(df["km_driven"].max()), 30000)

# Create input with all features = 0
input_data = pd.DataFrame(
    np.zeros((1, X.shape[1])),
    columns=X.columns
)

input_data["year"] = year
input_data["km_driven"] = km

input_scaled = scaler.transform(input_data)
price = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Car Price: ₹ {price:,.0f}</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
