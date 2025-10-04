import streamlit as st
import joblib
import re, string

# Load trained model & vectorizer
model = joblib.load("src/fake_news_model.pkl")
vectorizer = joblib.load("src/vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.write("Paste a news article below and check if it is **Fake** or **Real**.")

# Input box
news_text = st.text_area("Enter News Text Here:", height=200)

# Button
if st.button("Check News"):
    if news_text.strip() != "":
        clean_news = clean_text(news_text)
        vect_text = vectorizer.transform([clean_news])
        prediction = model.predict(vect_text)[0]
        confidence = model.predict_proba(vect_text).max() * 100

        if prediction.lower() == "fake":
            st.error(f"🚨 This news seems **FAKE** (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ This news seems **REAL** (Confidence: {confidence:.2f}%)")
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.caption("Made with ❤️ using Streamlit")