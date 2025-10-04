import streamlit as st
import joblib
import re, string
from streamlit_lottie import st_lottie
import requests

# ---------------- Load model & vectorizer ----------------
model = joblib.load("src/fake_news_model.pkl")
vectorizer = joblib.load("src/vectorizer.pkl")

# ---------------- Text cleaning function ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# ---------------- Lottie Animation Loader ----------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Confetti animation JSON
confetti_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jbrw3hcz.json")

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Dark mode + custom CSS
st.markdown("""
<style>
body {
    background-color: #121212;
    color: #ffffff;
}
h1 {
    color: #ffd700;
}
.stButton>button {
    background: linear-gradient(90deg,#ff4b4b,#ff6b81);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 24px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}
.stTextArea textarea {
    background-color: #1e1e1e;
    color: white;
    border-radius: 12px;
    border: 1px solid #444;
}
.card {
    background-color: #1e1e1e;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(255,215,0,0.3);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>📰 Tehelka Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Paste your news article below to check if it is <b>Fake</b> or <b>Real</b>.</p>", unsafe_allow_html=True)

# Input container
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    news_text = st.text_area("Enter News Text Here:", height=200)
    st.markdown("</div>", unsafe_allow_html=True)

# Button & result
if st.button("Check News"):
    if news_text.strip() != "":
        clean_news = clean_text(news_text)
        vect_text = vectorizer.transform([clean_news])
        prediction = model.predict(vect_text)[0]
        confidence = model.predict_proba(vect_text).max() * 100

        # Result card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if prediction.lower() == "fake":
            st.markdown(f"<h3 style='color:#ff4b4b;'>🚨 This news seems <b>FAKE</b></h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:#00ff99;'>✅ This news seems <b>REAL</b></h3>", unsafe_allow_html=True)
            # Show confetti animation for REAL news
            st_lottie(confetti_animation, height=250, key="confetti")
        st.markdown(f"<p>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("<p style='text-align:center; color:gray; font-size:14px;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
