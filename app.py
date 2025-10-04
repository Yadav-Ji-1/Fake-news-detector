import streamlit as st
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer, util

# ---------------- Settings ----------------
st.set_page_config(page_title="Smart Fake News Checker", page_icon="🧠", layout="centered")

# 🔑 SerpAPI key from Streamlit Secrets
API_KEY = st.secrets["SERPAPI_KEY"]

# Load SentenceTransformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- UI ----------------
st.title("🧠 Smart Fake News Detector")
st.markdown("Check any news article and get real-time web evidence 🔍")

news = st.text_area("Paste any news or claim here:", height=150)

if st.button("Check with Proof"):
    if not news.strip():
        st.warning("Please enter some news text!")
    else:
        with st.spinner("🔍 Searching for supporting sources..."):
            # Step 1: Search Google using Cloud-compatible serpapi
            try:
                params = {
                    "engine": "google",
                    "q": news,
                    "api_key": API_KEY,
                    "num": "5"
                }
                search = GoogleSearch(params)
                results_dict = search.get_dict()
                results = results_dict.get("organic_results", [])
            except Exception as e:
                st.error(f"Error while fetching search results: {e}")
                results = []

            if not results:
                st.error("No matching sources found. Try clearer wording.")
            else:
                # Step 2: Compare embeddings
                claim_emb = embedder.encode(news, convert_to_tensor=True)
                texts = [
                    r.get("title", "") + ". " + r.get("snippet", "")
                    for r in results
                ]
                evidence_emb = embedder.encode(texts, convert_to_tensor=True)
                sims = util.cos_sim(claim_emb, evidence_emb)[0].cpu().numpy()

                for i, r in enumerate(results):
                    r["similarity"] = float(sims[i])
                results = sorted(results, key=lambda x: x["similarity"], reverse=True)

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

                # Step 3: Show result
                st.markdown(f"<h2 style='color:{color};text-align:center'>{verdict}</h2>", unsafe_allow_html=True)

                # Step 4: Show top 3 proofs
                st.subheader("Top Evidence from Web")
                for r in results[:3]:
                    st.markdown(f"**[{r.get('title','No Title')}]({r.get('link','#')})**")
                    st.write(r.get("snippet", ""))
                    st.progress(min(max(int(r['similarity'] * 100), 0), 100))

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Made with ❤️ using SerpAPI & SentenceTransformers")
