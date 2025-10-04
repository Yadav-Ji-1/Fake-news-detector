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

# ---------------- CSS + Particle background + Hover ----------------
st.markdown("""
<style>
body {
    margin:0;
    background-color: #0d0d0d;
    color: #00ff99;
    font-family: 'Courier New', monospace;
    overflow-x: hidden;
}
h1 {
    text-align: center;
    color: #ff0000;
    text-shadow: 0 0 20px red, 0 0 40px red;
}
.stTextArea textarea {
    background-color: #1a1a1a !important;
    color: #00ff99 !important;
    border-radius: 12px !important;
    border: 2px solid #00ff99 !important;
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
    transition: transform 0.2s;
}
.stButton>button:hover {
    transform: scale(1.05);
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
    box-shadow: 0 0 20px rgba(255,215,0,0.3);
    margin-bottom: 20px;
    transition: transform 0.2s, box-shadow 0.3s;
}
.card:hover {
    transform: scale(1.02);
    box-shadow: 0 0 40px #00ff99;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 20px;
}
.main-container {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}
</style>

<!-- Particle background -->
<div id="particles-js" style="position:fixed; top:0; left:0; width:100%; height:100%; z-index:-1;"></div>
<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
<script>
particlesJS("particles-js", {
  "particles": {
    "number": {"value":80,"density":{"enable":true,"value_area":800}},
    "color": {"value":"#00ff99"},
    "shape": {"type":"circle"},
    "opacity": {"value":0.5,"random":true},
    "size": {"value":3,"random":true},
    "line_linked": {"enable":true,"distance":150,"color":"#00ff99","opacity":0.4,"width":1},
    "move": {"enable":true,"speed":2,"direction":"none","random":false,"straight":false,"out_mode":"out"}
  },
  "interactivity": {
    "detect_on":"canvas",
    "events":{"onhover":{"enable":true,"mode":"repulse"},"onclick":{"enable":true,"mode":"push"}},
    "modes":{"repulse":{"distance":100},"push":{"particles_nb":4}}
  },
  "retina_detect": true
});
</script>
""", unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Check News", "History", "About"])

# ---------------- Tab 1: Check News ----------------
with tab1:
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

                # Result card
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                if prediction.lower() == "fake":
                    st.markdown(f"<h2 style='color:red; text-shadow:0 0 20px red;'>🚨 FAKE NEWS ALERT 🚨</h2>", unsafe_allow_html=True)
                    # Shake + sound
                    components.html("""
                    <script>
                    let audio = new Audio('https://freesound.org/data/previews/514/514975_10296327-lq.mp3');
                    audio.play();
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

                st.progress(int(confidence))
                st.markdown(f"<p>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
                result_text = f"{prediction.upper()} ({confidence:.2f}%)"
                st.markdown("</div>", unsafe_allow_html=True)

        if result_text:
            st.text_area("Copy Result", value=result_text, height=50)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Tab 2: History ----------------
with tab2:
    if st.session_state.history:
        st.markdown("<h3 style='text-align:center;'>Last Predictions:</h3>", unsafe_allow_html=True)
        for i, (text, pred, conf) in enumerate(reversed(st.session_state.history)):
            st.markdown(f"""
            <div class='card'>
                <b>News {i+1}:</b> {text[:150]}... <br>
                Prediction: <b>{pred.upper()}</b> | Confidence: {conf:.2f}%
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align:center;'>No history yet. Check some news!</p>", unsafe_allow_html=True)

# ---------------- Tab 3: About ----------------
with tab3:
    st.markdown("""
    <div style='text-align:center;'>
        <h3>About This App</h3>
        <p>📰 This is a cinematic Fake News Detector built with Streamlit.</p>
        <p>✅ Real news shows confetti, 🚨 Fake news triggers shake + sound.</p>
        <p>💡 Neon theme, particle background, hover effects for professional look.</p>
        <p>Made with ❤️ by Himanshu</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown('<p class="footer">Made with ❤️ - Himanshu</p>', unsafe_allow_html=True)
