import streamlit as st
from sentence_transformers import SentenceTransformer, util
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

# ---------------- Settings ----------------
st.set_page_config(page_title="Smart Fake News Checker", page_icon="🧠", layout="wide")

# Load model (cached by SentenceTransformers)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Trusted domains list (small heuristic list - expand as needed)
TRUSTED_KEYWORDS = ["reuters", "bbc", "apnews", "aljazeera", "theguardian", "timesofindia", "ndtv", "espn", "icc-cricket"]

# Helpers
def fetch_snippet_from_url(url, max_paragraphs=5, timeout=6):
    """Fetch first few <p> texts joined. Return '' if fails."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SimpleBot/1.0)"}
        r = requests.get(url, timeout=timeout, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        # collect first max_paragraphs non-empty p texts
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if text:
                paragraphs.append(text)
            if len(paragraphs) >= max_paragraphs:
                break
        return " ".join(paragraphs)
    except Exception:
        return ""

def domain_score(url):
    """Simple credibility heuristic based on domain keywords."""
    try:
        domain = urlparse(url).netloc.lower()
    except:
        domain = url.lower()
    score = 0.0
    for k in TRUSTED_KEYWORDS:
        if k in domain:
            score += 0.12  # boost for trusted domain
    # small penalty for obscure TLDs? (skip to keep simple)
    return score

def extract_years(text):
    # find 4-digit years between 1900 and 2099
    return re.findall(r"\b(19|20)\d{2}\b", text)

def pretty_confidence_label(sim):
    if sim >= 0.6:
        return "High"
    if sim >= 0.45:
        return "Medium"
    return "Low"

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center'>🧠 Smart Fake News Detector — Upgraded</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>If exact match not found, we'll return related/similar sources and explain why.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])
with col1:
    news = st.text_area("Paste any news claim here:", height=140, placeholder="e.g. India defeats Pakistan in Asian Cup 2025")
    num_results = st.selectbox("Number of search results to check", options=[5, 10, 15], index=2)
    max_paras = st.slider("Paragraphs per page to fetch", 1, 7, 5)
    run_btn = st.button("Check with Proof")
with col2:
    st.markdown("**Tips:**")
    st.write("- Add context: year, location, event (eg. 'Asian Cup 2025').")
    st.write("- Try different phrasing if results weak.")
    st.write("- Click a source to open it and verify date & author.")
    st.markdown("---")
    st.write("Confidence thresholds:")
    st.write("High ≥ 60%, Medium 45–60%, Low < 45%")

# Run check
if run_btn:
    if not news.strip():
        st.warning("Please enter a claim or news text.")
    else:
        with st.spinner("Searching and analysing... (may take a few seconds)"):
            # 1) collect URLs
            urls = []
            try:
                for u in search(news, num_results=num_results):
                    if u not in urls:
                        urls.append(u)
            except Exception as e:
                st.error("Search failed: " + str(e))
                urls = []

            # 2) fetch snippets (first max_paras <p> tags) and build docs
            results = []
            for url in urls:
                snippet = fetch_snippet_from_url(url, max_paragraphs=max_paras)
                title = url  # we don't have easy title from googlesearch; using url as fallback
                # try to get <title> tag quickly
                try:
                    r = requests.get(url, timeout=5, headers={"User-Agent":"Mozilla/5.0"})
                    soup = BeautifulSoup(r.text, "html.parser")
                    t = soup.find("title")
                    if t and t.get_text().strip():
                        title = t.get_text().strip()
                except:
                    pass
                results.append({
                    "link": url,
                    "title": title,
                    "snippet": snippet
                })

            if not results:
                st.error("No results found. Try clearer wording or increase result count.")
            else:
                # 3) compute embeddings and similarities
                claim_emb = embedder.encode(news, convert_to_tensor=True)
                texts = [r["title"] + ". " + r.get("snippet", "") for r in results]
                evidence_emb = embedder.encode(texts, convert_to_tensor=True)
                sims = util.cos_sim(claim_emb, evidence_emb)[0].cpu().numpy()

                # attach similarity + domain boost + final score
                for i, r in enumerate(results):
                    base_sim = float(sims[i])
                    boost = domain_score(r["link"])
                    final_sim = base_sim + boost
                    # clamp 0..1
                    final_sim = max(0.0, min(final_sim, 0.9999))
                    r["base_sim"] = base_sim
                    r["final_sim"] = final_sim
                    r["confidence_label"] = pretty_confidence_label(final_sim)

                # sort by final_sim
                results = sorted(results, key=lambda x: x["final_sim"], reverse=True)

                # compute average best similarity to decide verdict
                top_k = results[:4]
                avg_sim = sum(r["final_sim"] for r in top_k) / len(top_k)

                # verdict heuristics
                if avg_sim >= 0.6:
                    verdict = "✅ Likely REAL (strong supporting sources found)"
                    color = "green"
                elif avg_sim >= 0.45:
                    verdict = "⚠️ Possibly REAL / Mixed (related sources found)"
                    color = "orange"
                else:
                    verdict = "🚨 Likely FAKE / Unverified (no good supporting sources)"
                    color = "red"

                # Show verdict & explanation
                st.markdown(f"<h2 style='color:{color};text-align:center'>{verdict}</h2>", unsafe_allow_html=True)
                st.markdown(f"**Claim:** {news}")
                st.markdown(f"**Top {len(top_k)} match average similarity:** {avg_sim:.2f}")

                # Show suggestions for refinement
                st.markdown("**Suggestions to improve search:**")
                st.write("- Add a year or event name: e.g. 'in Asian Cup 2025' or 'on October 3, 2025'.")
                st.write("- Add context: 'cricket', 'football', 'war', etc.")
                st.write("- Try alternative phrasing or named entities (person/organisation).")

                st.markdown("---")
                st.subheader("Top 4 matching sources (similar / related)")
                cols = st.columns(2)
                for idx, r in enumerate(top_k):
                    with cols[idx % 2]:
                        st.markdown(f"**[{r['title']}]({r['link']})**")
                        sim_pct = int(r["final_sim"] * 100)
                        st.write(f"Similarity: **{sim_pct}%** — Confidence: **{r['confidence_label']}**")
                        snippet_preview = r.get("snippet") or "No snippet fetched."
                        # highlight short preview
                        st.write(snippet_preview[:600] + ("..." if len(snippet_preview) > 600 else ""))
                        # quick metadata hints
                        years_found = re.findall(r"\b(19|20)\d{2}\b", snippet_preview)
                        if years_found:
                            st.write("Years mentioned in snippet:", ", ".join(years_found))
                        # domain credit
                        st.write("Domain credibility boost:", f"{domain_score(r['link']):.2f}")
                        st.progress(min(max(sim_pct, 0), 100))
                        st.markdown("---")

                # extra: if avg low, show possible "related claims"
                if avg_sim < 0.45:
                    st.subheader("Related / partly-matching headlines (may be paraphrases)")
                    for r in results[:6]:
                        st.write(f"- [{r['title']}]({r['link']}) — {int(r['final_sim']*100)}%")

# Footer
st.markdown("---")
st.caption("Made with ❤️ — improves with context. For very recent or breaking events, use official news sites or paid news APIs for highest reliability.")
