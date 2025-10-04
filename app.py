import streamlit as st
import joblib
import re, string
import streamlit.components.v1 as components

# Load model
model = joblib.load("src/fake_news_model.pkl")
vectorizer = joblib.load("src/vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

if 'history' not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# ---------------- Dramatic CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0d0d0d;
    color: #00ff99;
    font-family: 'Courier New', monospace;
}
h1 {
    text-align: center;
    color: #ff0000;
    text-shadow: 0 0 20px red, 0 0 40px red;
}
.stTextArea textarea {
    background-color: #1a1a1a;
    color: #00ff99;
    border-radius: 12px;
    border: 2px solid #00ff99;
    font-weight: bold;
}
.stButton>button {
    background-color: #ff0000;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    padding: 12px 24px;
    box-shadow: 0 0 20px #ff0000;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 10px #ff0000; }
    50% { box-shadow: 0 0 30px #ff4b4b; }
    100% { box-shadow: 0 0 10px #ff0000; }
}
.card {
    background-color: #1a1a1a;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #00ff99;
    box-shadow: 0 0 20px #00ff99;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>📰 FAKE NEWS DETECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ff0000;'>Paste news below. Be ready for the truth...</p>", unsafe_allow_html=True)

# Input
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    news_text = st.text_area("Enter News Here:", height=200)
    st.markdown("</div>", unsafe_allow_html=True)

# Button & Result
result_text = ""
if st.button("CHECK"):
    if news_text.strip() != "":
        clean_news = clean_text(news_text)
        vect_text = vectorizer.transform([clean_news])
        prediction = model.predict(vect_text)[0]
        confidence = model.predict_proba(vect_text).max() * 100

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if prediction.lower() == "fake":
            # Fake news dramatic effect
            st.markdown(f"<h2 style='color:red; text-shadow:0 0 20px red;'>🚨 FAKE NEWS ALERT 🚨</h2>", unsafe_allow_html=True)
            # Screen shake effect (simple simulation)
            components.html("""
            <script>
            let body = document.body;
            let i=0;
            function shake(){
                let x = Math.random()*10-5;
                let y = Math.random()*10-5;
                body.style.transform='translate('+x+'px,'+y+'px)';
                if(i<20){i++; requestAnimationFrame(shake);} else {body.style.transform='translate(0,0)';}
            }
            shake();
            </script>
            """, height=0)
        else:
            st.markdown(f"<h2 style='color:#00ff00; text-shadow:0 0 20px #00ff00;'>✅ REAL NEWS</h2>", unsafe_allow_html=True)
            # Confetti
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
        # Confidence
        st.progress(int(confidence))
        st.markdown(f"<p>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
        result_text = f"{prediction.upper()} ({confidence:.2f}%)"
        st.markdown("</div>", unsafe_allow_html=True)

# Copy Result
if result_text:
    st.text_area("Copy Result", value=result_text, height=50)

# Footer
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ - Himanshu</p>", unsafe_allow_html=True)
