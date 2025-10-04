import streamlit as st
from sentence_transformers import SentenceTransformer, util
from googlesearch import search  # Free, no API key required

# ---------------- Settings ----------------
st.set_page_config(
    page_title="Smart Fake News Checker",
    page_icon="🧠",
    layout="wide"
)

# Load SentenceTransformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center'>🧠 Smart Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Check any news article and get real-time web evidence 🔍</p>", unsafe_allow_html=True)

# Input area
news = st.text_area("Paste any news or claim here:", height=120)

if st.button("Check with Proof"):
    if not news.strip():
        st.warning("Please enter some news text!")
    else:
        with st.spinner("🔍 Searching for supporting sources..."):
            # Step 1: Google search (free, no API)
            results = []
            try:
                for url in search(news, num_results=5):
                    results.append({"link": url, "title": url, "snippet": ""})
            except Exception as e:
                st.error(f"Error fetching search results: {e}")

            if not results:
                st.error("No matching sources found. Try clearer wording.")
            else:
                # Step 2: Compare embeddings
                claim_emb = embedder.encode(news, convert_to_tensor=True)
                texts = [r["title"] + ". " + r.get("snippet", "") for r in results]
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

                # Step 4: Top Evidence in 3 columns
                st.subheader("Top Evidence from Web")
                cols = st.columns(3)
                for i, r in enumerate(results[:3]):
                    with cols[i]:
                        st.markdown(f"**[{r['title']}]({r['link']})**")
                        st.write(r.get("snippet", ""))
                        st.progress(min(max(int(r['similarity']*100), 0), 100))

# Footer
st.markdown("---")
st.caption("Made with ❤️ using googlesearch & SentenceTransformers")
