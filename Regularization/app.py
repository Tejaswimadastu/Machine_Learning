# app.py
import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Page Config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# UI Design
st.markdown("""
<style>
body {
background-color: #0f172a;
color: white;
}
.main {
background-color: #111827;
padding: 20px;
border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detection System")
st.write("Enter news text and check whether it is REAL or FAKE")

news = st.text_area("✍️ Paste News Article Here")

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter news text")
    else:
        vector = tfidf.transform([news])
        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        st.write("### 🔍 Prediction Result")
        st.write(f"Real Probability: {prob[1]*100:.2f}%")
        st.write(f"Fake Probability: {prob[0]*100:.2f}%")

        if pred == "REAL":
            st.success("✅ This news is REAL")
        else:
            st.error("❌ This news is FAKE")
