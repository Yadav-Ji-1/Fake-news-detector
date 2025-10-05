# app.py
import streamlit as st
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import textwrap

# ---------------- Settings ----------------
st.set_page_config(page_title="Super Smart Fake News Detector — Advanced", page_icon="🧠", layout="centered")
st.title("🧠 Super Smart Fake News Detector — Advanced")
st.write("Dynamic verdict (TRUE / FAKE / UNVERIFIABLE) using Semantic Similarity, NLI (Entailment/Contradiction), and Credibility weighting.")

# --- FULL POWER CONSTANTS ---
NUM_RESULTS = 30         # Maximum number of search results to fetch
TOP_K_FOR_VERDICT = 6    # Maximum number of top results to analyze
# ----------------------------

# ---------------- Caches / Model loaders ----------------
@st.cache_resource
def load_embedder():
    # Force CPU to avoid device issues on Cloud
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

@st.cache_resource
def load_nli_model():
    # NLI model (roberta-large-mnli) - CPU mode
    tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
    mdl = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    mdl.to("cpu")
    return tok, mdl

embedder = load_embedder()
nli_tok, nli_model = load_nli_model()

# ---------------- Utilities ----------------
def domain_from_url(url):
    try:
        return urlparse(url).netloc.replace("www.", "")
    except:
        return url

def pretty_pct(x):
    return f"{int(x*100)}%"

# ---------------- Rank-claim helpers (Wikipedia list check) ----------------
ORDINAL_WORDS = {
    "first":1, "second":2, "third":3, "fourth":4, "fifth":5, "sixth":6, "seventh":7, "eighth":8, "ninth":9, "tenth":10,
    "eleventh":11, "twelfth":12, "thirteenth":13, "fourteenth":14, "fifteenth":15, "sixteenth":16, "seventeenth":17,
    "eighteenth":18, "nineteenth":19, "twentieth":20
}
ROLE_KEYWORDS = ["prime minister", "prime-minister", "pm", "president", "chief minister", "cm", "governor", "chief justice"]

def find_ordinal_and_role(text):
    t = text.lower()
    num = None # Fixed Indentation
    m = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\b', t)
    if m:
        num = int(m.group(1))
    else:
        for w, n in ORDINAL_WORDS.items():
            if re.search(r'\b' + re.escape(w) + r'\b', t):
                num = n
                break
    role = None
    for rk in ROLE_KEYWORDS:
        if rk in t:
            role = rk.replace('-', ' ')
            break
    return num, role

def extract_person_candidate(text):
    patterns = [
        r"^([\w\s\.\-]{2,80}?)\s+is\s+the\b",
        r"^([\w\s\.\-]{2,80}?)\s+is\s+(\d{1,2})",
        r"is\s+([\w\s\.\-]{2,80}?)\s+the\s+\d{1,2}",
        r"^([\w\s\.\-]{2,80}?)\s+was\s+the\b",
    ]
    for p in patterns:
        mm = re.search(p, text, flags=re.IGNORECASE)
        if mm:
            name = mm.group(1).strip()
            if len(name) > 1 and not re.match(r'^(it|he|she|they|this|that)$', name.lower()):
                return name
    tokens = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
    if tokens:
        return tokens[0]
    return text.split()[0]

def normalize_name(s):
    return re.sub(r'[^a-z]', '', s.lower())

def find_wikipedia_list_page(role, country, serp_api_key):
    query = f'List of {role} of {country} site:en.wikipedia.org'
    try:
        params = {"engine":"google", "q": query, "api_key": serp_api_key, "num": 1}
        search = GoogleSearch(params)
        res = search.get_dict()
        organic = res.get("organic_results") or []
        if organic:
            return organic[0].get("link")
    except Exception:
        pass
    cand = f"https://en.wikipedia.org/wiki/List_of_{role.replace(' ','_')}_of_{country.replace(' ','_')}"
    return cand

def parse_wikipedia_list(url):
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, 'html.parser')
        names = []
        tables = soup.find_all("table", {"class": ["wikitable", "sortable"]})
        for table in tables:
            for tr in table.find_all("tr"):
                tds = tr.find_all(["td", "th"])
                if not tds:
                    continue
                textcells = [td.get_text(separator=" ").strip() for td in tds if td.get_text(strip=True)]
                for cell in textcells[:2]:
                    if re.search(r'\b(19|20)\d{2}\b', cell) and len(cell) < 30:
                        continue
                    if len(cell) > 1 and re.search(r'[A-Za-z]', cell):
                        cleaned = re.sub(r'\[.*?\]|\(.*?\)', '', cell).strip()
                        cand = re.split(r'\n|,|;|-', cleaned)[0].strip()
                        if len(cand) > 1 and not re.search(r'\b(year|term|born)\b', cand, re.I):
                            names.append(cand)
                            break
        if not names:
            for li in soup.find_all('li'):
                text = li.get_text().strip()
                if len(text) > 3 and re.search(r'\b[A-Z][a-z]+', text):
                    if re.search(r'\b(19|20)\d{2}\b', text) or re.search(r'\bPrime Minister\b', text, re.I):
                        cleaned = re.sub(r'\[.*?\]|\(.*?\)', '', text).strip()
                        names.append(cleaned.split('—')[0].split('-')[0].strip())
        normalized = []
        for n in names:
            nn = re.sub(r'\s+', ' ', n).strip()
            if nn and nn not in normalized:
                normalized.append(nn)
        return normalized
    except Exception:
        return []

def match_person_in_list(person_candidate, names_list):
    pc = normalize_name(person_candidate)
    for idx, full in enumerate(names_list):
        if not full:
            continue
        fn = normalize_name(full)
        if pc and (pc in fn or fn in pc):
            return idx+1, full
    tokens = person_candidate.lower().split()
    for idx, full in enumerate(names_list):
        fn = full.lower()
        if all(any(tok in part for part in fn.split()) for tok in tokens if len(tok)>2):
            return idx+1, full
    return None, None

def check_rank_claim_wikipedia(person, ordinal, role, country, serp_api_key):
    wiki_url = find_wikipedia_list_page(role, country, serp_api_key)
    names = parse_wikipedia_list(wiki_url)
    if not names:
        return {"decisive": False, "reason": "Could not retrieve list page or parse it.", "wiki_url": wiki_url}
    rank, matched_name = match_person_in_list(person, names)
    if rank is None:
        return {"decisive": False, "reason": "Person not found in list parsed from " + wiki_url, "wiki_url": wiki_url, "names_sample": names[:6]}
    else:
        if rank == ordinal:
            return {"decisive": True, "result": True, "rank": rank, "matched_name": matched_name, "wiki_url": wiki_url}
        else:
            return {"decisive": True, "result": False, "rank": rank, "matched_name": matched_name, "wiki_url": wiki_url}

# ---------------- NLI & sentence helpers ----------------
def nli_entailment_prob(premise, hypothesis):
    inputs = nli_tok.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
    # NLI labels for roberta-large-mnli are: 0=entailment, 1=neutral, 2=contradiction
    return probs[0].item(), probs[1].item(), probs[2].item() # Entailment, Neutral, Contradiction

def best_sentence_for_claim(snippet, claim):
    import re
    sents = re.split(r'(?<=[.!?])\s+', snippet) if snippet else []
    if not sents:
        return snippet or "", 0.0
    sent_embs = embedder.encode(sents, convert_to_tensor=True)
    claim_emb = embedder.encode(claim, convert_to_tensor=True)
    sims = util.cos_sim(claim_emb, sent_embs)[0].cpu().numpy()
    best_idx = int(sims.argmax())
    return sents[best_idx], float(sims[best_idx])

def domain_boost(domain):
    trusted = ["bbc", "reuters", "theguardian", "nytimes", "indiatimes", "ndtv", "timesofindia", "cnn", "espn", "espncricinfo", "aljazeera"]
    return 0.2 if any(t in domain for t in trusted) else 0.0

def analyze_top_articles(normalized, claim, top_k):
    sims, entails, neutral, contradicts, creds = [], [], [], [], []
    for r in normalized[:top_k]:
        text = (r.get("title","") + ". " + (r.get("snippet") or ""))
        best_sent, best_sim = best_sentence_for_claim(r.get("snippet",""), claim)
        # fallback semantic sim using whole text if best_sim==0
        sem_sim = best_sim if best_sim>0 else float(util.cos_sim(
            embedder.encode(claim, convert_to_tensor=True),
            embedder.encode(text, convert_to_tensor=True)
        )[0].item())
        
        try:
            entail_p, neutral_p, contra_p = nli_entailment_prob(best_sent or text, claim)
        except Exception:
            entail_p, neutral_p, contra_p = 0.0, 0.0, 0.0
        
        # --- NLI Smart Filter (Fixes high contradiction on high sim matches) ---
        if sem_sim > 0.80 and contra_p > 0.80 and entail_p < 0.10:
             # Assume this is the correct news headline reporting the claim, not contradicting it.
             entail_p = 0.80
             contra_p = 0.05
        # ----------------------------------------------------------------------
        
        domain = urlparse(r.get("link","")).netloc
        cred = domain_boost(domain)
        
        sims.append(sem_sim)
        entails.append(entail_p)
        neutral.append(neutral_p)
        contradicts.append(contra_p)
        creds.append(cred)
        
        r["entail_p"] = entail_p
        r["neutral_p"] = neutral_p
        r["contra_p"] = contra_p
        r["sem_sim"] = sem_sim
        r["cred"] = cred
        r["best_sent"] = best_sent 
        
    avg_sim = float(np.mean(sims)) if sims else 0.0
    avg_ent = float(np.mean(entails)) if entails else 0.0
    avg_neu = float(np.mean(neutral)) if neutral else 0.0
    avg_con = float(np.mean(contradicts)) if contradicts else 0.0
    avg_cred = float(np.mean(creds)) if creds else 0.0

    # Calculate net support as (Entailment - Contradiction)
    net_support = avg_ent - avg_con
    
    # DYNAMIC SCORING LOGIC
    # SCORE 1: Support Score (Prioritizes credible logical support)
    # This is the primary decision factor: Net Support * (1 + Credibility)
    support_score = net_support * (1 + avg_cred)
    
    # SCORE 2: Final Score (Used for general ranking/transparency)
    final_score = 0.50 * net_support + 0.30 * avg_sim + 0.20 * avg_cred
    
    metrics = {
        "avg_ent": avg_ent, 
        "avg_neu": avg_neu, 
        "avg_con": avg_con, 
        "avg_sim": avg_sim, 
        "avg_cred": avg_cred, 
        "net_support": net_support,
        "support_score": support_score
    }
    return final_score, metrics, normalized[:top_k]

# ---------------- Main UI inputs ----------------
claim = st.text_area("Enter claim or news sentence:", height=140, placeholder="e.g. India defeats Pakistan in Asia Cup 2025")

st.info(f"Using **{NUM_RESULTS}** recent news results (Last 24hrs) and analyzing top **{TOP_K_FOR_VERDICT}** matches (Full Power Mode).")

if st.button("Verify Claim"):
    if not claim.strip():
        st.warning("Please enter a claim.")
    else:
        with st.spinner("Analysing... (this may take a few seconds)"):
            
            # 1) Rank-claim check (Wikipedia) if applicable
            ordinal, role = find_ordinal_and_role(claim)
            person_candidate = None
            country = "India" if "india" in claim.lower() else "" 
            if ordinal and role:
                person_candidate = extract_person_candidate(claim)
                m_country = re.search(r'\bof\s+([A-Za-z\s]+)', claim, flags=re.IGNORECASE)
                if m_country:
                    country = m_country.group(1).strip()
                rank_check = check_rank_claim_wikipedia(person_candidate, ordinal, role, country or "India", st.secrets["SERPAPI_KEY"])
                if rank_check.get("decisive"):
                    if rank_check.get("result"):
                        st.markdown("<h2 style='color:green;text-align:center'>✅ TRUE</h2>", unsafe_allow_html=True)
                        st.write(f"Reason: Authoritative list ({rank_check.get('wiki_url')}) shows **{rank_check.get('matched_name')}** as the {ordinal}th {role} of {country or 'the country'}.")
                    else:
                        st.markdown("<h2 style='color:red;text-align:center'>🚨 FAKE</h2>", unsafe_allow_html=True)
                        st.write(f"Reason: Authoritative list ({rank_check.get('wiki_url')}) shows **{rank_check.get('matched_name')}** as the {rank_check.get('rank')}th {role}, not the {ordinal}th.")
                    st.write("Source (for verification):", rank_check.get("wiki_url"))
                    st.stop()  # done

            # 2) SerpAPI fetch (Filtering results to last 24hrs using tbs=qdr:d1)
            try:
                # Using tbs=qdr:d1 to filter results to the last 24 hours for better relevance
                params = {"engine":"google", "q": claim, "tbm":"nws", "tbs":"qdr:d1", "num": NUM_RESULTS, "api_key": st.secrets["SERPAPI_KEY"]}
                search = GoogleSearch(params)
                data = search.get_dict()
                results = data.get("news_results") or data.get("organic_results") or []
            except Exception as e:
                st.error("Search failed: " + str(e))
                results = []

            if not results:
                st.markdown("<h2 style='color:red;text-align:center'>🚨 FAKE</h2>", unsafe_allow_html=True)
                st.write("Reason: No relevant **recent** news results returned by the live search API. The claim is unconfirmed or outdated.")
            else:
                normalized = []
                for r in results:
                    title = r.get("title") or r.get("title_raw") or r.get("title_original") or ""
                    snippet = r.get("snippet") or r.get("snippet_highlighted") or r.get("excerpt") or ""
                    link = r.get("link") or r.get("source", {}).get("url") or r.get("source_link") or ""
                    normalized.append({"title": title, "snippet": snippet, "link": link})

                # compute decision via new intelligence module
                final_score, metrics, analyzed = analyze_top_articles(normalized, claim, top_k=TOP_K_FOR_VERDICT)

                # DYNAMIC VERDICT LOGIC: (TRUE / FAKE / UNVERIFIABLE)
                
                # Condition for TRUE: High credibility-weighted support AND good relevance.
                if metrics["support_score"] >= 0.15 and metrics["avg_sim"] >= 0.50:
                    st.markdown("<h2 style='color:green;text-align:center'>✅ TRUE</h2>", unsafe_allow_html=True)
                    st.write("Reason: **Strong logical support from credible sources** found, confirming the claim's relevance.")
                    verdict_msg = "TRUE"
                
                # Condition for UNVERIFIABLE: Not enough support (low support_score) and low relevance, but high neutrality (no strong contradiction found).
                elif metrics["avg_sim"] < 0.50 and metrics["avg_neu"] > 0.60:
                    st.markdown("<h2 style='color:orange;text-align:center'>⚠️ UNVERIFIABLE</h2>", unsafe_allow_html=True)
                    st.write("Reason: Low semantic relevance and high neutral logical probability across sources. The claim is either too vague, futuristic, or lacks sufficient recent confirmation.")
                    verdict_msg = "UNVERIFIABLE"

                # Default to FAKE: Insufficient support or strong contradiction present.
                else:
                    st.markdown("<h2 style='color:red;text-align:center'>🚨 FAKE</h2>", unsafe_allow_html=True)
                    st.write("Reason: Insufficient combined credibility and logical support, or strong refutation present. The claim is likely refuted, outdated, or lacks reliable confirmation.")
                    verdict_msg = "FAKE"


                st.write(f"Details — Support Score (Credibility Weighted): {metrics['support_score']:.2f}, avg semantic sim: {metrics['avg_sim']:.2f}, net support (E-C): {metrics['net_support']:.2f}")


                # show short synthesized reason
                if verdict_msg == "TRUE":
                    ex = []
                    for r in analyzed[:3]:
                        if r.get("sem_sim", 0.0) > 0.4 and r.get("entail_p", 0.0) > r.get("contra_p", 0.0):
                            ex.append(textwrap.shorten(r.get("best_sent") or r.get("snippet",""), width=160, placeholder="..."))
                    if ex:
                        st.info("Example supporting excerpts: " + " | ".join(ex))
                elif verdict_msg == "FAKE":
                    best = analyzed[0] if analyzed else None
                    if best and best.get("best_sent"):
                        st.info("Closest (but weak) excerpt: " + textwrap.shorten(best.get("best_sent") or best.get("snippet",""), width=220, placeholder="..."))

                # transparency
                with st.expander("Show analyzed top sources and scores"):
                    for idx, r in enumerate(analyzed):
                        st.markdown(f"**{idx+1}. {r.get('title') or r.get('link','(no title)')}**")
                        st.write(f"- Domain: {domain_from_url(r.get('link',''))}")
                        st.write(f"- Semantic similarity (sentence-level): {pretty_pct(r.get('sem_sim',0.0))}")
                        st.write(f"- **Net Support (Entail-Contra)**: {r.get('entail_p',0.0) - r.get('contra_p',0.0):.2f}")
                        st.write(f"  (E: {pretty_pct(r.get('entail_p',0.0))} | N: {pretty_pct(r.get('neutral_p',0.0))} | C: {pretty_pct(r.get('contra_p',0.0))})")
                        st.write(f"- Credibility boost: {r.get('cred',0.0):.2f}")
                        st.write(f"- Link: {r.get('link')}")
                        st.markdown("---")


# Footer
st.markdown("---")
st.caption("Project: NLP-driven Fact-Checking System. Use responsibly.")
tral_p"] = neutral_p
        r["contra_p"] = contra_p
        r["sem_sim"] = sem_sim
        r["cred"] = cred
        r["best_sent"] = best_sent 
        
    avg_sim = float(np.mean(sims)) if sims else 0.0
    avg_ent = float(np.mean(entails)) if entails else 0.0
    avg_neu = float(np.mean(neutral)) if neutral else 0.0
    avg_con = float(np.mean(contradicts)) if contradicts else 0.0
    avg_cred = float(np.mean(creds)) if creds else 0.0

    # Calculate net support as (Entailment - Contradiction)
    net_support = avg_ent - avg_con
    
    # DYNAMIC SCORING LOGIC
    # SCORE 1: Support Score (Prioritizes credible logical support)
    support_score = net_support * (1 + avg_cred)
    
    # SCORE 2: Final Score (Used for general ranking)
    final_score = 0.50 * net_support + 0.30 * avg_sim + 0.20 * avg_cred
    
    metrics = {
        "avg_ent": avg_ent, 
        "avg_neu": avg_neu, # Added for UNVERIFIABLE check
        "avg_con": avg_con, 
        "avg_sim": avg_sim, 
        "avg_cred": avg_cred, 
        "net_support": net_support,
        "support_score": support_score
    }
    return final_score, metrics, normalized[:top_k]

# ---------------- Main UI inputs ----------------
claim = st.text_area("Enter claim or news sentence:", height=140, placeholder="e.g. India defeats Pakistan in Asia Cup 2025")

st.info(f"Using **{NUM_RESULTS}** recent news results (Last 24hrs) and analyzing top **{TOP_K_FOR_VERDICT}** matches (Full Power Mode).")

if st.button("Verify Claim"):
    if not claim.strip():
        st.warning("Please enter a claim.")
    else:
        with st.spinner("Analysing... (this may take a few seconds)"):
            
            # 1) Rank-claim check (Wikipedia) if applicable
            ordinal, role = find_ordinal_and_role(claim)
            person_candidate = None
            country = "India" if "india" in claim.lower() else "" 
            if ordinal and role:
                person_candidate = extract_person_candidate(claim)
                m_country = re.search(r'\bof\s+([A-Za-z\s]+)', claim, flags=re.IGNORECASE)
                if m_country:
                    country = m_country.group(1).strip()
                rank_check = check_rank_claim_wikipedia(person_candidate, ordinal, role, country or "India", st.secrets["SERPAPI_KEY"])
                if rank_check.get("decisive"):
                    if rank_check.get("result"):
                        st.markdown("<h2 style='color:green;text-align:center'>✅ TRUE</h2>", unsafe_allow_html=True)
                        st.write(f"Reason: Authoritative list ({rank_check.get('wiki_url')}) shows **{rank_check.get('matched_name')}** as the {ordinal}th {role} of {country or 'the country'}.")
                    else:
                        st.markdown("<h2 style='color:red;text-align:center'>🚨 FAKE</h2>", unsafe_allow_html=True)
                        st.write(f"Reason: Authoritative list ({rank_check.get('wiki_url')}) shows **{rank_check.get('matched_name')}** as the {rank_check.get('rank')}th {role}, not the {ordinal}th.")
                    st.write("Source (for verification):", rank_check.get("wiki_url"))
                    st.stop()  # done

            # 2) SerpAPI fetch (Filtering results to last 24hrs using tbs=qdr:d1)
            try:
                # NEW: Added tbs=qdr:d1 to filter results to the last 24 hours
                params = {"engine":"google", "q": claim, "tbm":"nws", "tbs":"qdr:d1", "num": NUM_RESULTS, "api_key": st.secrets["SERPAPI_KEY"]}
                search = GoogleSearch(params)
                data = search.get_dict()
                results = data.get("news_results") or data.get("organic_results") or []
            except Exception as e:
                st.error("Search failed: " + str(e))
                results = []

            if not results:
                st.markdown("<h2 style='color:red;text-align:center'>🚨 FAKE</h2>", unsafe_allow_html=True)
                st.write("Reason: No relevant **recent** news results returned by the live search API. Try changing the date or adding context.")
            else:
                normalized = []
                for r in results:
                    title = r.get("title") or r.get("title_raw") or r.get("title_original") or ""
                    snippet = r.get("snippet") or r.get("snippet_highlighted") or r.get("excerpt") or ""
                    link = r.get("link") or r.get("source", {}).get("url") or r.get("source_link") or ""
                    normalized.append({"title": title, "snippet": snippet, "link": link})

                # compute decision via new intelligence module
                final_score, metrics, analyzed = analyze_top_articles(normalized, claim, top_k=TOP_K_FOR_VERDICT)

                # DYNAMIC VERDICT LOGIC: (TRUE / FAKE / UNVERIFIABLE)
                
                # Condition for TRUE
                if metrics["support_score"] >= 0.15 and metrics["avg_sim"] >= 0.50:
                    st.markdown("<h2 style='color:green;text-align:center'>✅ TRUE</h2>", unsafe_allow_html=True)
                    st.write("Reason: **Strong logical support from credible sources** found, confirming the claim's relevance.")
                    verdict_msg = "TRUE"
                
                # Condition for UNVERIFIABLE (if not TRUE, check NEUTRAL)
                elif metrics["avg_sim"] < 0.40 and metrics["avg_neu"] > 0.60:
                    st.markdown("<h2 style='color:orange;text-align:center'>⚠️ UNVERIFIABLE</h2>", unsafe_allow_html=True)
                    st.write("Reason: Low semantic relevance and high neutral logical probability across sources. The claim is either too vague, futuristic, or lacks confirmation.")
                    verdict_msg = "UNVERIFIABLE"

                # Default to FAKE
                else:
                    st.markdown("<h2 style='color:red;text-align:center'>🚨 FAKE</h2>", unsafe_allow_html=True)
                    st.write("Reason: Insufficient combined credibility and logical support. The claim is likely refuted, outdated, or lacks confirmation by relevant sources.")
                    verdict_msg = "FAKE"


                st.write(f"Details — Support Score (Credibility Weighted): {metrics['support_score']:.2f}, avg semantic sim: {metrics['avg_sim']:.2f}, net support: {metrics['net_support']:.2f}")


                # show short synthesized reason
                if verdict_msg == "TRUE":
                    # create short example excerpts from top analyzed
                    ex = []
                    for r in analyzed[:3]:
                        # Only show snippets that have high entailment and low contradiction
                        if r.get("sem_sim", 0.0) > 0.4 and r.get("entail_p", 0.0) > r.get("contra_p", 0.0):
                            ex.append(textwrap.shorten(r.get("best_sent") or r.get("snippet",""), width=160, placeholder="..."))
                    if ex:
                        st.info("Example supporting excerpts: " + " | ".join(ex))
                elif verdict_msg == "FAKE":
                    best = analyzed[0] if analyzed else None
                    if best and best.get("best_sent"):
                        st.info("Closest (but weak) excerpt: " + textwrap.shorten(best.get("best_sent") or best.get("snippet",""), width=220, placeholder="..."))

                # transparency
                with st.expander("Show analyzed top sources and scores"):
                    for idx, r in enumerate(analyzed):
                        st.markdown(f"**{idx+1}. {r.get('title') or r.get('link','(no title)')}**")
                        st.write(f"- Domain: {domain_from_url(r.get('link',''))}")
                        st.write(f"- Semantic similarity (sentence-level): {pretty_pct(r.get('sem_sim',0.0))}")
                        st.write(f"- **Net Support (Entail-Contra)**: {r.get('entail_p',0.0) - r.get('contra_p',0.0):.2f}")
                        st.write(f"  (E: {pretty_pct(r.get('entail_p',0.0))} | N: {pretty_pct(r.get('neutral_p',0.0))} | C: {pretty_pct(r.get('contra_p',0.0))})")
                        st.write(f"- Credibility boost: {r.get('cred',0.0):.2f}")
                        st.write(f"- Link: {r.get('link')}")
                        st.markdown("---")


# Footer
st.markdown("---")
st.caption("Project: NLP-driven Fact-Checking System. Use responsibly.")
    num = None
    m = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\b', t)
    if m:
        num = int(m.group(1))
    else:
        for w, n in ORDINAL_WORDS.items():
            if re.search(r'\b' + re.escape(w) + r'\b', t):
                num = n
                break
    role = None
    for rk in ROLE_KEYWORDS:
        if rk in t:
            role = rk.replace('-', ' ')
            break
    return num, role

def extract_person_candidate(text):
    # Best-effort extraction
    patterns = [
        r"^([\w\s\.\-]{2,80}?)\s+is\s+the\b",
        r"^([\w\s\.\-]{2,80}?)\s+is\s+(\d{1,2})",
        r"is\s+([\w\s\.\-]{2,80}?)\s+the\s+\d{1,2}",
        r"^([\w\s\.\-]{2,80}?)\s+was\s+the\b",
    ]
    for p in patterns:
        mm = re.search(p, text, flags=re.IGNORECASE)
        if mm:
            name = mm.group(1).strip()
            if len(name) > 1 and not re.match(r'^(it|he|she|they|this|that)$', name.lower()):
                return name
    tokens = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
    if tokens:
        return tokens[0]
    return text.split()[0]

def normalize_name(s):
    return re.sub(r'[^a-z]', '', s.lower())

def find_wikipedia_list_page(role, country, serp_api_key):
    # Try to find Wikipedia list using SerpAPI
    query = f'List of {role} of {country} site:en.wikipedia.org'
    try:
        params = {"engine":"google", "q": query, "api_key": serp_api_key, "num": 1}
        search = GoogleSearch(params)
        res = search.get_dict()
        organic = res.get("organic_results") or []
        if organic:
            return organic[0].get("link")
    except Exception:
        pass
    cand = f"https://en.wikipedia.org/wiki/List_of_{role.replace(' ','_')}_of_{country.replace(' ','_')}"
    return cand

def parse_wikipedia_list(url):
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, 'html.parser')
        names = []
        tables = soup.find_all("table", {"class": ["wikitable", "sortable"]})
        for table in tables:
            for tr in table.find_all("tr"):
                tds = tr.find_all(["td", "th"])
                if not tds:
                    continue
                textcells = [td.get_text(separator=" ").strip() for td in tds if td.get_text(strip=True)]
                for cell in textcells[:2]:
                    if re.search(r'\b(19|20)\d{2}\b', cell) and len(cell) < 30:
                        continue
                    if len(cell) > 1 and re.search(r'[A-Za-z]', cell):
                        cleaned = re.sub(r'\[.*?\]|\(.*?\)', '', cell).strip()
                        cand = re.split(r'\n|,|;|-', cleaned)[0].strip()
                        if len(cand) > 1 and not re.search(r'\b(year|term|born)\b', cand, re.I):
                            names.append(cand)
                            break
        if not names:
            for li in soup.find_all('li'):
                text = li.get_text().strip()
                if len(text) > 3 and re.search(r'\b[A-Z][a-z]+', text):
                    if re.search(r'\b(19|20)\d{2}\b', text) or re.search(r'\bPrime Minister\b', text, re.I):
                        cleaned = re.sub(r'\[.*?\]|\(.*?\)', '', text).strip()
                        names.append(cleaned.split('—')[0].split('-')[0].strip())
        normalized = []
        for n in names:
            nn = re.sub(r'\s+', ' ', n).strip()
            if nn and nn not in normalized:
                normalized.append(nn)
        return normalized
    except Exception:
        return []

def match_person_in_list(person_candidate, names_list):
    pc = normalize_name(person_candidate)
    for idx, full in enumerate(names_list):
        if not full:
            continue
        fn = normalize_name(full)
        if pc and (pc in fn or fn in pc):
            return idx+1, full
    tokens = person_candidate.lower().split()
    for idx, full in enumerate(names_list):
        fn = full.lower()
        if all(any(tok in part for part in fn.split()) for tok in tokens if len(tok)>2):
            return idx+1, full
    return None, None

def check_rank_claim_wikipedia(person, ordinal, role, country, serp_api_key):
    wiki_url = find_wikipedia_list_page(role, country, serp_api_key)
    names = parse_wikipedia_list(wiki_url)
    if not names:
        return {"decisive": False, "reason": "Could not retrieve list page or parse it.", "wiki_url": wiki_url}
    rank, matched_name = match_person_in_list(person, names)
    if rank is None:
        return {"decisive": False, "reason": "Person not found in list parsed from " + wiki_url, "wiki_url": wiki_url, "names_sample": names[:6]}
    else:
        if rank == ordinal:
            return {"decisive": True, "result": True, "rank": rank, "matched_name": matched_name, "wiki_url": wiki_url}
        else:
            return {"decisive": True, "result": False, "rank": rank, "matched_name": matched_name, "wiki_url": wiki_url}

# ---------------- NLI & sentence helpers ----------------
def nli_entailment_prob(premise, hypothesis):
    inputs = nli_tok.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    # Ensure inputs are on CPU device
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
    # NLI labels for roberta-large-mnli are: 0=entailment, 1=neutral, 2=contradiction
    # We return all three for flexibility in the analysis function
    return probs[0].item(), probs[1].item(), probs[2].item() # Entailment, Neutral, Contradiction

def best_sentence_for_claim(snippet, claim):
    import re
    sents = re.split(r'(?<=[.!?])\s+', snippet) if snippet else []
    if not sents:
        return snippet or "", 0.0
    sent_embs = embedder.encode(sents, convert_to_tensor=True)
    claim_emb = embedder.encode(claim, convert_to_tensor=True)
    sims = util.cos_sim(claim_emb, sent_embs)[0].cpu().numpy()
    best_idx = int(sims.argmax())
    return sents[best_idx], float(sims[best_idx])

def domain_boost(domain):
    trusted = ["bbc", "reuters", "theguardian", "nytimes", "indiatimes", "ndtv", "timesofindia", "cnn", "espn", "espncricinfo"]
    return 0.2 if any(t in domain for t in trusted) else 0.0

def analyze_top_articles(normalized, claim, top_k):
    sims, entails, neutral, contradicts, creds = [], [], [], [], []
    for r in normalized[:top_k]:
        text = (r.get("title","") + ". " + (r.get("snippet") or ""))
        best_sent, best_sim = best_sentence_for_claim(r.get("snippet",""), claim)
        # fallback semantic sim using whole text if best_sim==0
        sem_sim = best_sim if best_sim>0 else float(util.cos_sim(
            embedder.encode(claim, convert_to_tensor=True),
            embedder.encode(text, convert_to_tensor=True)
        )[0].item())
        # NLI on best sentence (or whole text)
        try:
            entail_p, neutral_p, contra_p = nli_entailment_prob(best_sent or text, claim)
        except Exception:
            entail_p, neutral_p, contra_p = 0.0, 0.0, 0.0
        
        # --- NEW LOGIC: Adjust NLI for high similarity matches (Fixes Trump/Nukes issue) ---
        # If the semantic similarity is very high (> 0.8), but NLI gives high contradiction, 
        # it's likely a news site reporting the claim/headline, not contradicting it.
        if sem_sim > 0.80 and contra_p > 0.80 and entail_p < 0.10:
             entail_p = 0.80
             contra_p = 0.05
        # ----------------------------------------------------------------------------------
        
        domain = urlparse(r.get("link","")).netloc
        cred = domain_boost(domain)
        
        sims.append(sem_sim)
        entails.append(entail_p)
        neutral.append(neutral_p)
        contradicts.append(contra_p)
        creds.append(cred)
        
        r["entail_p"] = entail_p
        r["neutral_p"] = neutral_p
        r["contra_p"] = contra_p
        r["sem_sim"] = sem_sim
        r["cred"] = cred
        r["best_sent"] = best_sent # Store best sentence for reporting
        
    avg_sim = float(np.mean(sims)) if sims else 0.0
    avg_ent = float(np.mean(entails)) if entails else 0.0
    avg_con = float(np.mean(contradicts)) if contradicts else 0.0
    avg_cred = float(np.mean(creds)) if creds else 0.0

    # Calculate net support as (Entailment - Contradiction)
    net_support = avg_ent - avg_con
    
    # Final Score based on Net Support, Semantic Similarity and Credibility
    final_score = 0.50 * net_support + 0.30 * avg_sim + 0.20 * avg_cred
    
    # Store all relevant average metrics
    metrics = {
        "avg_ent": avg_ent, 
        "avg_con": avg_con, 
        "avg_sim": avg_sim, 
        "avg_cred": avg_cred, 
        "net_support": net_support
    }
    return final_score, metrics, normalized[:top_k]

# ---------------- Main UI inputs ----------------
claim = st.text_area("Enter claim or news sentence:", height=140, placeholder="e.g. India defeats Pakistan in Asia Cup 2025")

# Sliders removed, using full power constants defined above.
st.info(f"Using **{NUM_RESULTS}** search results and analyzing top **{TOP_K_FOR_VERDICT}** matches (Full Power Mode).")

if st.button("Verify Claim"):
    if not claim.strip():
        st.warning("Please enter a claim.")
    else:
        with st.spinner("Analysing... (this may take a few seconds)"):
            # 1) Rank-claim check (Wikipedia) if applicable
            ordinal, role = find_ordinal_and_role(claim)
            person_candidate = None
            country = "India" if "india" in claim.lower() else ""  # basic default; can improve
            if ordinal and role:
                person_candidate = extract_person_candidate(claim)
                m_country = re.search(r'\bof\s+([A-Za-z\s]+)', claim, flags=re.IGNORECASE)
                if m_country:
                    country = m_country.group(1).strip()
                rank_check = check_rank_claim_wikipedia(person_candidate, ordinal, role, country or "India", st.secrets["SERPAPI_KEY"])
                if rank_check.get("decisive"):
                    if rank_check.get("result"):
                        st.markdown("<h2 style='color:green;text-align:center'>✅ TRUE</h2>", unsafe_allow_html=True)
                        st.write(f"Reason: Authoritative list ({rank_check.get('wiki_url')}) shows **{rank_check.get('matched_name')}** as the {ordinal}th {role} of {country or 'the country'}.")
                    else:
                        st.markdown("<h2 style='color:red;text-align:center'>🚨 FAKE</h2>", unsafe_allow_html=True)
                        st.write(f"Reason: Authoritative list ({rank_check.get('wiki_url')}) shows **{rank_check.get('matched_name')}** as the {rank_check.get('rank')}th {role}, not the {ordinal}th.")
                    st.write("Source (for verification):", rank_check.get("wiki_url"))
                    st.stop()  # done

            # 2) SerpAPI fetch
            try:
                params = {"engine":"google", "q": claim, "tbm":"nws", "num": NUM_RESULTS, "api_key": st.secrets["SERPAPI_KEY"]}
                search = GoogleSearch(params)
                data = search.get_dict()
                results = data.get("news_results") or data.get("organic_results") or []
            except Exception as e:
                st.error("Search failed: " + str(e))
                results = []

            if not results:
                st.markdown("<h2 style='color:red;text-align:center'>🚨 FAKE</h2>", unsafe_allow_html=True)
                st.write("Reason: No relevant news results returned by the live search API. Try adding context (date/event).")
            else:
                normalized = []
                for r in results:
                    title = r.get("title") or r.get("title_raw") or r.get("title_original") or ""
                    snippet = r.get("snippet") or r.get("snippet_highlighted") or r.get("excerpt") or ""
                    link = r.get("link") or r.get("source", {}).get("url") or r.get("source_link") or ""
                    normalized.append({"title": title, "snippet": snippet, "link": link})

                # compute decision via new intelligence module
                final_score, metrics, analyzed = analyze_top_articles(normalized, claim, top_k=TOP_K_FOR_VERDICT)

                # Strict Thresholds and Output
                # TRUE only if (Net Support is positive AND Semantic Similarity is high)
                if final_score >= 0.20 and metrics["avg_sim"] >= 0.60:
                    st.markdown("<h2 style='color:green;text-align:center'>✅ TRUE</h2>", unsafe_allow_html=True)
                    st.write("Reason: Multiple sources provide strong semantic and net logical support for this claim.")
                    st.write(f"Details — net support (entail-contra): {metrics['net_support']:.2f}, avg semantic sim: {metrics['avg_sim']:.2f}, avg credibility boost: {metrics['avg_cred']:.2f}")
                else:
                    st.markdown("<h2 style='color:red;text-align:center'>🚨 FAKE</h2>", unsafe_allow_html=True)
                    st.write("Reason: Weak semantic or net logical support found from credible sources. The claim is likely refuted or neutral/unverifiable.")
                    st.write(f"Closest evidence metrics — net support (entail-contra): {metrics['net_support']:.2f}, avg semantic sim: {metrics['avg_sim']:.2f}, avg credibility boost: {metrics['avg_cred']:.2f}")

                # show short synthesized reason
                if final_score >= 0.20:
                    # create short example excerpts from top analyzed
                    ex = []
                    for r in analyzed[:3]:
                        # Only show snippets that have high entailment and low contradiction
                        if r.get("sem_sim", 0.0) > 0.4 and r.get("entail_p", 0.0) > r.get("contra_p", 0.0):
                            ex.append(textwrap.shorten(r.get("best_sent") or r.get("snippet",""), width=160, placeholder="..."))
                    if ex:
                        st.info("Example supporting excerpts: " + " | ".join(ex))
                else:
                    best = analyzed[0] if analyzed else None
                    if best and best.get("best_sent"):
                        st.info("Closest (but weak) excerpt: " + textwrap.shorten(best.get("best_sent") or best.get("snippet",""), width=220, placeholder="..."))

                # transparency
                with st.expander("Show analyzed top sources and scores"):
                    for idx, r in enumerate(analyzed):
                        st.markdown(f"**{idx+1}. {r.get('title') or r.get('link','(no title)')}**")
                        st.write(f"- Domain: {domain_from_url(r.get('link',''))}")
                        st.write(f"- Semantic similarity (sentence-level): {pretty_pct(r.get('sem_sim',0.0))}")
                        st.write(f"- **Net Support (Entail-Contra)**: {r.get('entail_p',0.0) - r.get('contra_p',0.0):.2f}")
                        st.write(f"  (E: {pretty_pct(r.get('entail_p',0.0))} | C: {pretty_pct(r.get('contra_p',0.0))})")
                        st.write(f"- Credibility boost: {r.get('cred',0.0):.2f}")
                        st.write(f"- Link: {r.get('link')}")
                        st.markdown("---")


# Footer
st.markdown("---")
st.caption("Made with ❤️ — SerpAPI + SentenceTransformers + NLI. Use responsibly and verify critical claims with official sources.")
