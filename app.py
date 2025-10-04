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

# ---------------- Page Config ----------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
body {margin:0; background-color:#0d0d0d; color:#00ff99; font-family:'Courier New', monospace; overflow-x:hidden;}
h1 {text-align:center; color:#ff0000; text-shadow:0 0 20px red, 0 0 40px red; margin-top:20px;}
.stTextArea textarea {background-color:#1a1a1a !important; color:#00ff99 !important; border-radius:12px !important; border:2px solid #00ff99 !important; font-weight:bold;}
.stButton>button {background-color:#ff0000; color:white; font-size:18px; font-weight:bold; border-radius:10px; padding:12px 24px; box-shadow:0 0 20px #ff0000; animation:pulse 2s infinite; transition:transform 0.2s;}
.stButton>button:hover {transform:scale(1.05);}
@keyframes pulse {0% {box-shadow:0 0 10px #ff0000;}50% {box-shadow:0 0 30px #ff4b4b;}100% {box-shadow:0 0 10px #ff0000;}}
.card {background-color:#1a1a1a; padding:20px; border-radius:15px; border:2px solid #00ff99; box-shadow:0 0 20px rgba(255,215,0,0.3); margin-bottom:20px; transition: transform 0.2s, box-shadow 0.3s;}
.card:hover {transform: scale(1.02); box-shadow:0 0 40px #00ff99;}
.footer {text-align:center; color:gray; font-size:14px; margin-top:20px;}
.main-container {display:flex; justify-content:center; align-items:center; flex-direction:column;}
.radio-right {display:flex; justify-content:flex-end; margin-right:20px;}
</style>

<!-- Particle background -->
<div id="particles-js" style="position:fixed; top:0; left:0; width:100%; height:100%; z-index:-1;"></div>
<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
<script>
particlesJS("particles-js", {
  "particles": {"number":{"value":80,"density":{"enable":true,"value_area":800}},
  "color":{"value":"#00ff99"},"shape":{"type":"circle"},"opacity":{"value":0.5,"random":true},
  "size":{"value":3,"random":true},"line_linked":{"enable":true,"distance":150,"color":"#00ff99","opacity":0.4,"width":1},
  "move":{"enable":true,"speed":2,"direction":"none","random":false,"straight":false,"out_mode":"out"}},
  "interactivity":{"detect_on":"canvas","events":{"onhover":{"enable":true,"mode":"repulse"},"onclick":{"enable":true,"mode":"push"}},
  "modes":{"repulse":{"distance":100},"push":{"particles_nb":4}}},"retina_detect":true
});
</script>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("<h1>📰 FAKE NEWS DETECTOR</h1>", unsafe_allow_html=True)

# ---------------- Navigation (fixed empty label issue & unique key) ----------------
st.markdown('<div class="radio-right">', unsafe_allow_html=True)
page = st.radio(
    label="Menu",  # non-empty label
    options=["Check News", "History", "About"],
    horizontal=True,
    key="unique_menu_radio",  # unique key to avoid duplicate error
    label_visibility="hidden"
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Page: Check News ----------------
if page == "Check News":
    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        news_text = st.text_area("Enter News Here:", key="news_input", height=200)
        st.markdown("</div>", unsafe_allow_html=True)

        result_text = ""
        if st.button("CHECK", key="check_btn"):
            if news_text.strip() != "":
                clean_news = clean_text(news_text)
                vect_text = vectorizer.transform([clean_news])
                prediction = model.predict(vect_text)[0]
                confidence = model.predict_proba(vect_text).max() * 100

                # Update history
                st.session_state.history.append((news_text, prediction, confidence))
                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[-10:]

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                if prediction.lower() == "fake":
                    st.markdown(f"<h2 style='color:red; text-shadow:0 0 20px red;'>🚨 FAKE NEWS ALERT 🚨</h2>", unsafe_allow_html=True)
                    # 🔊 Play sound
                    st.audio("https://www.soundjay.com/button/beep-07.mp3", format="audio/mp3", start_time=0)
                else:
                    st.markdown(f"<h2 style='color:#00ff00; text-shadow:0 0 20px #00ff00;'>✅ REAL NEWS</h2>", unsafe_allow_html=True)
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

                st.progress(int(confidence))
                st.markdown(f"<p>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
                result_text = f"{prediction.upper()} ({confidence:.2f}%)"
                st.markdown("</div>", unsafe_allow_html=True)

        if result_text:
            st.text_area("Copy Result", value=result_text, height=50, key="result_area")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Page: History ----------------
elif page == "History":
    st.markdown("<h2 style='text-align:center;'>Last Predictions</h2>", unsafe_allow_html=True)
    if st.session_state.history:
        for i, (text, pred, conf) in enumerate(reversed(st.session_state.history)):
            st.markdown(f"""
            <div class='card'>
                <b>News {i+1}:</b> {text[:150]}... <br>
                Prediction: <b>{pred.upper()}</b> | Confidence: {conf:.2f}%
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align:center;'>No history yet. Check some news!</p>", unsafe_allow_html=True)

# ---------------- Page: About ----------------
elif page == "About":
    st.markdown("""
    <div style='text-align:center;'>
        <h3>About This App</h3>
        <p>📰 Cinematic Fake News Detector built with Streamlit</p>
        <p>✅ Real news shows confetti</p>
        <p>🚨 Fake news triggers alert + sound</p>
        <p>💡 Neon theme, particle background, hover effects for professional look</p>
        <p>Made with ❤️ by Himanshu</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown('<p class="footer">Made with ❤️ - Himanshu</p>', unsafe_allow_html=True)    options=["Check News", "History", "About"],
    horizontal=True,
    key="menu",
    label_visibility="hidden"  # hide label
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Page: Check News ----------------
if page == "Check News":
    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        news_text = st.text_area("Enter News Here:", height=200)
        st.markdown("</div>", unsafe_allow_html=True)

        result_text = ""
        if st.button("CHECK"):
            if news_text.strip() != "":
                clean_news = clean_text(news_text)
                vect_text = vectorizer.transform([clean_news])
                prediction = model.predict(vect_text)[0]
                confidence = model.predict_proba(vect_text).max() * 100

                # Update history
                st.session_state.history.append((news_text, prediction, confidence))
                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[-10:]

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                if prediction.lower() == "fake":
                    st.markdown(f"<h2 style='color:red; text-shadow:0 0 20px red;'>🚨 FAKE NEWS ALERT 🚨</h2>", unsafe_allow_html=True)
                    # 🔊 Play sound
                    st.audio("https://www.soundjay.com/button/beep-07.mp3", format="audio/mp3")
                else:
                    st.markdown(f"<h2 style='color:#00ff00; text-shadow:0 0 20px #00ff00;'>✅ REAL NEWS</h2>", unsafe_allow_html=True)
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

                st.progress(int(confidence))
                st.markdown(f"<p>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
                result_text = f"{prediction.upper()} ({confidence:.2f}%)"
                st.markdown("</div>", unsafe_allow_html=True)

        if result_text:
            st.text_area("Copy Result", value=result_text, height=50)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Page: History ----------------
elif page == "History":
    st.markdown("<h2 style='text-align:center;'>Last Predictions</h2>", unsafe_allow_html=True)
    if st.session_state.history:
        for i, (text, pred, conf) in enumerate(reversed(st.session_state.history)):
            st.markdown(f"""
            <div class='card'>
                <b>News {i+1}:</b> {text[:150]}... <br>
                Prediction: <b>{pred.upper()}</b> | Confidence: {conf:.2f}%
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align:center;'>No history yet. Check some news!</p>", unsafe_allow_html=True)

# ---------------- Page: About ----------------
elif page == "About":
    st.markdown("""
    <div style='text-align:center;'>
        <h3>About This App</h3>
        <p>📰 Cinematic Fake News Detector built with Streamlit</p>
        <p>✅ Real news shows confetti</p>
        <p>🚨 Fake news triggers alert + sound</p>
        <p>💡 Neon theme, particle background, hover effects for professional look</p>
        <p>Made with ❤️ by Himanshu</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown('<p class="footer">Made with ❤️ - Himanshu</p>', unsafe_allow_html=True)# ---------------- Navigation ----------------
st.markdown('<div class="radio-right">', unsafe_allow_html=True)
page = st.radio("", ["Check News", "History", "About"], horizontal=True, key="menu")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Page: Check News ----------------
if page == "Check News":
    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        news_text = st.text_area("Enter News Here:", height=200)
        st.markdown("</div>", unsafe_allow_html=True)

        result_text = ""
        if st.button("CHECK"):
            if news_text.strip() != "":
                clean_news = clean_text(news_text)
                vect_text = vectorizer.transform([clean_news])
                prediction = model.predict(vect_text)[0]
                confidence = model.predict_proba(vect_text).max() * 100

                # Update history
                st.session_state.history.append((news_text, prediction, confidence))
                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[-10:]

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                if prediction.lower() == "fake":
                    st.markdown(f"<h2 style='color:red; text-shadow:0 0 20px red;'>🚨 FAKE NEWS ALERT 🚨</h2>", unsafe_allow_html=True)
                    # Play sound
                    st.audio("https://www.soundjay.com/button/beep-07.mp3", format="audio/mp3")
                else:
                    st.markdown(f"<h2 style='color:#00ff00; text-shadow:0 0 20px #00ff00;'>✅ REAL NEWS</h2>", unsafe_allow_html=True)
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

                st.progress(int(confidence))
                st.markdown(f"<p>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
                result_text = f"{prediction.upper()} ({confidence:.2f}%)"
                st.markdown("</div>", unsafe_allow_html=True)

        if result_text:
            st.text_area("Copy Result", value=result_text, height=50)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Page: History ----------------
elif page == "History":
    st.markdown("<h2 style='text-align:center;'>Last Predictions</h2>", unsafe_allow_html=True)
    if st.session_state.history:
        for i, (text, pred, conf) in enumerate(reversed(st.session_state.history)):
            st.markdown(f"""
            <div class='card'>
                <b>News {i+1}:</b> {text[:150]}... <br>
                Prediction: <b>{pred.upper()}</b> | Confidence: {conf:.2f}%
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align:center;'>No history yet. Check some news!</p>", unsafe_allow_html=True)

# ---------------- Page: About ----------------
elif page == "About":
    st.markdown("""
    <div style='text-align:center;'>
        <h3>About This App</h3>
        <p>📰 Cinematic Fake News Detector built with Streamlit</p>
        <p>✅ Real news shows confetti</p>
        <p>🚨 Fake news triggers alert + sound</p>
        <p>💡 Neon theme, particle background, hover effects for professional look</p>
        <p>Made with ❤️ by Himanshu</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown('<p class="footer">Made with ❤️ - Himanshu</p>', unsafe_allow_html=True)
