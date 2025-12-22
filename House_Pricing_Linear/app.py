import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config("Simple Linear Regression", layout="centered")

# ---------- BACKGROUND IMAGE ----------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            font-family: 'Segoe UI';
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("bg.jpg")

# ---------- CSS (GLOBAL SHADOW FIX) ----------
st.markdown("""
<style>
/* ================= CARD ================= */
.card{
    background: rgba(255,255,255,0.96);
    padding:25px;
    border-radius:16px;
    box-shadow:0 12px 30px rgba(0,0,0,.3);
    margin-bottom:25px;
}

/* ================= METRIC BOX ================= */
.metric-box{
    background: rgba(255,255,255,0.97);
    padding:18px;
    border-radius:14px;
    text-align:center;
    box-shadow:0 10px 25px rgba(0,0,0,.25);
}

/* ================= GLOBAL TEXT (HEADERS + SUBHEADERS) ================= */
h1, h2, h3, h4, h5, h6,
p, span, label,
[data-testid="stMarkdownContainer"] *,
[data-testid="stHeader"],
[data-testid="stSubheader"] {
    color: #000000 !important;
    font-weight: 700;
    text-shadow:
        3px 3px 4px #ffffff,
       -3px -3px 4px #ffffff,
        3px -3px 4px #ffffff,
       -3px  3px 4px #ffffff,
        0px  0px 12px #ffffff;
}

/* ================= METRIC VALUES ================= */
.metric-value{
    font-size:30px;
    font-weight:900;
    color:black;
    margin-top:6px;
    text-shadow:
        3px 3px 4px white,
       -3px -3px 4px white,
        3px -3px 4px white,
       -3px  3px 4px white,
        0px  0px 14px white;
}

/* ================= METRIC LABEL ================= */
.metric-label{
    font-size:16px;
    font-weight:700;
    color:black;
    text-shadow:
        3px 3px 4px white,
       -3px -3px 4px white,
        3px -3px 4px white,
       -3px  3px 4px white,
        0px  0px 10px white;
}

/* ================= PREDICTION BOX ================= */
.prediction-box{
    background:linear-gradient(135deg,#2563eb,#1e40af);
    color:white;
    padding:18px;
    border-radius:14px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("""
<div class="card">
<h1>🏠 House Price Prediction</h1>
<p>Simple Linear Regression (Area → Price)</p>
</div>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
df = pd.read_csv("house_data.csv")

# ---------- DATASET PREVIEW ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview (Full Data)")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ---------- MODEL ----------
X = df[["area"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred) / 100000
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) / 100000
r2 = r2_score(y_test, y_pred)

# ---------- GRAPH ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Area vs Price")

fig, ax = plt.subplots()
ax.scatter(df["area"], df["price"]/100000, alpha=0.6)
ax.plot(df["area"],
        model.predict(scaler.transform(df[["area"]]))/100000,
        color="red")
ax.set_xlabel("Area (sqft)")
ax.set_ylabel("Price (Lakhs)")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- PERFORMANCE ----------
st.markdown('<div class="card"><h3>Model Performance</h3>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">MAE (Lakhs)</div>
        <div class="metric-value">{mae:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">RMSE (Lakhs)</div>
        <div class="metric-value">{rmse:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">R² Score</div>
        <div class="metric-value">{r2:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Feature Used</div>
        <div class="metric-value">Area</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREDICTION ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict House Price")

area = st.slider("Area (sqft)", 500, 12000, 3000)
price = model.predict(scaler.transform([[area]]))[0] / 100000

st.markdown(
    f'<div class="prediction-box">Predicted Price: ₹ {price:.2f} Lakhs</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
