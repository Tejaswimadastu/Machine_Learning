import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("📰 Fake News Detection System")

@st.cache_data
def load_model():
    data = pd.read_csv("fake_or_real_news.csv")
    data['label'] = data['label'].map({'REAL': 0, 'FAKE': 1})

    def clean(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    data['text'] = data['text'].apply(clean)

    X = data['text']
    y = data['label']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train = tfidf.fit_transform(X_train)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    return tfidf, model

tfidf, model = load_model()

user_input = st.text_area("Enter News Text", height=200)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        clean_input = re.sub(r'[^a-z\s]', '', user_input.lower())
        vector = tfidf.transform([clean_input])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("🚨 This news is FAKE")
        else:
            st.success("✅ This news is REAL")
