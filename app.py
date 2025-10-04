import streamlit as st
import joblib
import re, string
import streamlit.components.v1 as components

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

# ---------------- Initialize history ----------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Animated gradient background
st.markdown("""
<style>
body {
    margin: 0;
    height: 100vh;
    background: linear-gradient(270deg, #1e3c72, #2a5298, #1e3c72);
    background-size: 600% 600%;
    animation: gradientBG 15s ease infinite;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
@keyframes gradientBG {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
h1 {
    color: #ffd700;
}
.stButton>button {
    background: linear-gradient(90deg,#4facfe,#00f2fe);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 24px;
    transition: transform 0.2s;
}
.stButton>button:hover {
    transform: scale(1.05);
}
.stTextArea textarea {
    background-color: rgba(0,0,0,0.5);
    color: white;
    border-radius: 12px;
    border: 1px solid #444;
}
.card {
    background-color: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}
.tooltip {
    border-bottom: 1px dotted white; 
    cursor: help;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Paste your news article below to check if it is <b>Fake</b> or <b>Real</b>. <span class='tooltip' title='Fake news is misleading or false content. Real news is verified and reliable.'>ℹ️</span></p>", unsafe_allow_html=True)

# Input container
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    news_text = st.text_area("Enter News Text Here:", height=200)
    st.markdown("</div>", unsafe_allow_html=True)

# Button & result
result_text = ""
if st.button("Check News"):
    if news_text.strip() != "":
        clean_news = clean_text(news_text)
        vect_text = vectorizer.transform([clean_news])
        prediction = model.predict(vect_text)[0]
        confidence = model.predict_proba(vect_text).max() * 100

        # Update history
        st.session_state.history.append((news_text, prediction, confidence))
        if len(st.session_state.history) > 5:
            st.session_state.history = st.session_state.history[-5:]

        # Result card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if prediction.lower() == "fake":
            st.markdown(f"<h3 style='color:#ff4b4b;'>🚨 This news seems <b>FAKE</b></h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:#00ff99;'>✅ This news seems <b>REAL</b></h3>", unsafe_allow_html=True)
            # Confetti HTML
            confetti_html = """
            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
            <lottie-player src="https://assets10.lottiefiles.com/packages/lf20_jbrw3hcz.json"  
            background="transparent"  
            speed="1"  
            style="width: 300px; height: 300px;"  
            loop  
            autoplay></lottie-player>
            """
            components.html(confetti_html, height=320)
        # Confidence meter
        st.progress(int(confidence))
        st.markdown(f"<p>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
        result_text = f"{prediction.upper()} ({confidence:.2f}%)"
        st.markdown("</div>", unsafe_allow_html=True)

# Copy result button
if result_text:
    st.write("Copy result:")
    st.text_area("Result", value=result_text, height=50)

# History panel
if st.session_state.history:
    st.markdown("<h3>Last 5 Predictions:</h3>", unsafe_allow_html=True)
    for i, (text, pred, conf) in enumerate(reversed(st.session_state.history)):
        st.markdown(f"<div class='card'><b>News {i+1}:</b> {text[:100]}... <br> Prediction: <b>{pred.upper()}</b> | Confidence: {conf:.2f}%</div>", unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align:center; color:gray; font-size:14px;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)    box-shadow: 0 0 20px rgba(255,215,0,0.3);
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
