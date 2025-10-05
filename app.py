# app.py
import streamlit as st
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urlparse
import numpy as np
import textwrap

# ---------- Settings ----------
st.set_page_config(page_title="Smart Fake News Checker — Clear Verdict", page_icon="🧠", layout="centered")

# ---------- Load model (cached) ----------
@st.cache_resource
def load_embedder():
    # force cpu to avoid device errors on Cloud
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

embedder = load_embedder()

# ---------- Helpers ----------
def domain_from_url(url):
    try:
        return urlparse(url).netloc.replace("www.", "")
    except:
        return url

def pretty_pct(x):
    return f"{int(x*100)}%"

def build_explanation_true(top_hits, avg_top_sim):
    # top_hits is list of dicts with title, snippet, link, similarity
    domains = []
    examples = []
    for h in top_hits[:3]:
        d = domain_from_url(h.get("link", h.get("source_link", "")))
        domains.append(d)
        snip = h.get("snippet") or h.get("snippet_plain", "")
        if snip:
            # take first 160 chars
            examples.append(textwrap.shorten(snip, width=160, placeholder="..."))
        else:
            examples.append(h.get("title",""))
    domains_str = ", ".join(dict.fromkeys(domains))  # unique preserve order
    explanation = (
        f"This claim is classified as TRUE. An analysis of top news results shows multiple independent "
        f"sources ({domains_str}) reporting similar information. The average similarity of top supporting "
        f"articles is {pretty_pct(avg_top_sim)}, which indicates strong semantic match with the claim. "
    )
    if examples:
        explanation += "Example excerpts from these reports: " + " | ".join(examples[:2])
    return explanation

def build_explanation_fake(best_hit):
    # best_hit is the single best match dict or None
    if not best_hit:
        return ("This claim is classified as FAKE. No credible news reports or reliable sources were found that "
                "support this statement. Try adding a date, event name, or more context (e.g. 'cricket', 'Asia Cup 2025').")
    # if there is a low-quality match, we explain
    domain = domain_from_url(best_hit.get("link",""))
    sim = best_hit.get("similarity", 0.0)
    snippet = best_hit.get("snippet") or best_hit.get("title","")
    explanation = (
        f"This claim is classified as FAKE. The highest-related result comes from {domain} with similarity "
        f"{pretty_pct(sim)}, but it does not strongly support the claim. Excerpt: {textwrap.shorten(snippet, width=200, placeholder='...')}. "
        "No multiple independent, high-confidence reports were found."
    )
    return explanation

# ---------- UI ----------
st.title("🧠 Smart Fake News Checker — Clear TRUE / FAKE Verdict")
st.write("Paste any news claim; the tool compares it against live news results and returns a simple TRUE or FAKE verdict with a short reason.")

claim = st.text_area("Enter claim or news sentence:", height=140, placeholder="e.g. India defeats Pakistan in Asia Cup 2025")
num_results = st.slider("How many news results to check (SerpAPI)", min_value=5, max_value=30, value=10, step=5)
top_k_for_verdict = st.slider("Consider top K matches for verdict", min_value=1, max_value=10, value=4)

if st.button("Verify Claim"):
    if not claim.strip():
        st.warning("Please enter a claim to verify.")
    else:
        with st.spinner("Fetching live news results and analysing..."):
            # 1) Call SerpAPI Google News
            try:
                params = {
                    "engine": "google",
                    "q": claim,
                    "tbm": "nws",
                    "num": num_results,
                    "api_key": st.secrets["SERPAPI_KEY"]
                }
                search = GoogleSearch(params)
                data = search.get_dict()
                results = data.get("news_results") or data.get("organic_results") or []
            except Exception as e:
                st.error(f"Search failed: {e}")
                results = []

            # normalize results into expected fields
            normalized = []
            for r in results:
                title = r.get("title") or r.get("title_raw") or r.get("title_original") or r.get("title", "")
                snippet = r.get("snippet") or r.get("snippet_highlighted") or r.get("snippet_raw") or r.get("excerpt") or ""
                link = r.get("link") or r.get("source", {}).get("url") or r.get("source_link") or r.get("source_url") or r.get("link", "")
                normalized.append({"title": title, "snippet": snippet, "link": link})

            if not normalized:
                # no news results found -> FAKE explanation
                st.markdown("<h2 style='color:red;text-align:center'>🚨 FAKE</h2>", unsafe_allow_html=True)
                st.write("Reason: No news results returned by the live search API for this claim. Try adding more context (date, event, sport, location).")
            else:
                # 2) Build texts and compute embeddings
                texts = [ (r["title"] + ". " + (r["snippet"] or "")) for r in normalized ]
                claim_emb = embedder.encode(claim, convert_to_tensor=True)
                evidence_emb = embedder.encode(texts, convert_to_tensor=True)
                sims = util.cos_sim(claim_emb, evidence_emb)[0].cpu().numpy()
                # attach similarity
                for i, r in enumerate(normalized):
                    r["similarity"] = float(sims[i])

                # sort by similarity descending
                normalized.sort(key=lambda x: x["similarity"], reverse=True)

                # pick top_k entries for decision
                top_k = normalized[: min(len(normalized), top_k_for_verdict)]
                # average similarity among top_k
                avg_top_sim = float(np.mean([r["similarity"] for r in top_k])) if top_k else 0.0

                # Decision boundary (binary)
                # These thresholds chosen to give clearer TRUE/FAKE decisions:
                # avg_top_sim >= 0.50 -> TRUE, otherwise FAKE
                threshold = 0.50
                if avg_top_sim >= threshold:
                    verdict = "TRUE"
                    color = "green"
                    explanation = build_explanation_true(top_k, avg_top_sim)
                else:
                    verdict = "FAKE"
                    color = "red"
                    best_hit = top_k[0] if top_k else None
                    explanation = build_explanation_fake(best_hit)

                # Display verdict
                header_html = f"<h2 style='color:{color};text-align:center'>{'✅ TRUE' if verdict=='TRUE' else '🚨 FAKE'}</h2>"
                st.markdown(header_html, unsafe_allow_html=True)

                # Provide short reason paragraph
                st.write("**Reason / Summary:**")
                st.info(explanation)

                # show confidence label (High/Medium/Low) but keep verdict binary
                if avg_top_sim >= 0.65:
                    conf = "High"
                elif avg_top_sim >= 0.5:
                    conf = "Medium"
                else:
                    conf = "Low"
                st.write(f"**Confidence:** {conf} (Top matches average similarity: {pretty_pct(avg_top_sim)})")

                # Optional: expandable details for transparency (click to view)
                with st.expander("Show top matching sources and scores (for transparency)"):
                    for idx, r in enumerate(normalized[:10]):
                        st.markdown(f"**{idx+1}. {r.get('title') or r.get('link','(no title)')}**")
                        st.write(f"- Domain: {domain_from_url(r.get('link',''))}  ")
                        snippet = r.get("snippet") or "(no snippet)"
                        st.write(f"- Excerpt: {snippet[:300]}{'...' if len(snippet)>300 else ''}")
                        st.write(f"- Similarity: {pretty_pct(r.get('similarity',0.0))}")
                        st.markdown("---")

        # done processing

# Footer
st.markdown("---")
st.caption("Made with ❤️ — uses SerpAPI + SentenceTransformers. Verdicts are automatic; verify important geopolitical or sensitive claims with official sources.")
