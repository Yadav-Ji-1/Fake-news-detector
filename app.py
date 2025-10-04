import streamlit as st
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer, util

# ---------------- Settings ----------------
st.set_page_config(page_title="Smart Fake News Checker-Ekanthydv", page_icon="🧠", layout="centered")

# ---------------- API Key ----------------
# Streamlit secrets me dal diya hai
API_KEY = st.secrets["SERPAPI_KEY"]

# ---------------- Model ----------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- UI ----------------
st.title("🧠 Smart Fake News Detector-Ekantydv")
st.markdown("Check any news article or claim and get real-time evidence 🔍")

news = st.text_area("Paste any news or claim here:", height=150,
                    placeholder="e.g. India defeats Pakistan in Asian Cup 2025")

num_results = st.slider("Number of news results to check", 5, 20, 10)

if st.button("Check with Proof"):
    if not news.strip():
        st.warning("Please enter some news text!")
    else:
        with st.spinner("🔍 Searching for supporting sources..."):
            # ---------------- Step 1: Search Google News via SerpAPI ----------------
            search = GoogleSearch({
                "engine": "google",
                "q": news,
                "tbm": "nws",  # Google News
                "num": num_results,
                "api_key": API_KEY
            })
            results = search.get_dict().get("news_results", [])

            if not results:
                st.error("No matching sources found. Try clearer wording.")
            else:
                # ---------------- Step 2: Compute embeddings & similarity ----------------
                claim_emb = embedder.encode(news, convert_to_tensor=True)
                texts = [r.get("title", "") + ". " + r.get("snippet", "") for r in results]
                evidence_emb = embedder.encode(texts, convert_to_tensor=True)
                sims = util.cos_sim(claim_emb, evidence_emb)[0].cpu().numpy()

                for i, r in enumerate(results):
                    r["similarity"] = float(sims[i])
                results = sorted(results, key=lambda x: x["similarity"], reverse=True)

                # ---------------- Step 3: Verdict ----------------
                avg_sim = sum(sims) / len(sims)
                if avg_sim > 0.55:
                    verdict = "✅ Likely REAL (matches online sources)"
                    color = "green"
                elif avg_sim > 0.35:
                    verdict = "⚠️ Uncertain — mixed evidence"
                    color = "orange"
                else:
                    verdict = "🚨 Likely FAKE (no supporting info found)"
                    color = "red"

                st.markdown(f"<h2 style='color:{color};text-align:center'>{verdict}</h2>", unsafe_allow_html=True)

                # ---------------- Step 4: Top 5 Evidence ----------------
                st.subheader("Top Evidence from Web")
                for r in results[:5]:
                    st.markdown(f"**[{r['title']}]({r['link']})**")
                    st.write(r.get("snippet", ""))
                    st.progress(r['similarity'])

st.markdown("---")
st.caption("Made with ❤️ using SerpAPI & SentenceTransformers-Ekanth yadav")
