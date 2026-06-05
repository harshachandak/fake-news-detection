import streamlit as st
import joblib
import re

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

st.title("📰 Fake News Detection System")
st.write("Enter a news article to check if it's REAL or FAKE")

text = st.text_area("Enter News Text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        # ⚡ FIX: better interpretation logic
        if prob[1] > 0.55:
            st.success(f"✅ REAL NEWS ({prob[1]*100:.2f}% confidence)")
        else:
            st.error(f"❌ FAKE NEWS ({prob[0]*100:.2f}% confidence)")