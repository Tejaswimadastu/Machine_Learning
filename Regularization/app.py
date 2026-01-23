import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("📰 Fake News Detection System")
st.write("Enter news text to check whether it is REAL or FAKE")

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Accuracy
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

    return tfidf, nb_model, lr_model, nb_acc, lr_acc

tfidf, nb_model, lr_model, nb_acc, lr_acc = load_model()

st.write("### Model Accuracy")
st.write("Naive Bayes Accuracy:", nb_acc)
st.write("Logistic Regression Accuracy:", lr_acc)

user_input = st.text_area("Enter News Text", height=200)

model_choice = st.selectbox("Choose Model", ["Naive Bayes", "Logistic Regression"])

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        clean_input = re.sub(r'[^a-z\s]', '', user_input.lower())
        vector = tfidf.transform([clean_input])

        if model_choice == "Naive Bayes":
            model = nb_model
        else:
            model = lr_model

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        st.write("Real Probability:", prob[0])
        st.write("Fake Probability:", prob[1])

        if prediction == 1:
            st.error("🚨 This news is FAKE")
        else:
            st.success("✅ This news is REAL")
