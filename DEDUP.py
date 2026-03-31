import streamlit as st
import pandas as pd
import re
import rispy
from rapidfuzz import fuzz
import numpy as np
import io
import google.generativeai as genai
import time
import concurrent.futures
import requests

# --- KEYBOARD SHORTCUTS IMPORT ---
try:
    from streamlit_shortcuts import add_keyboard_shortcuts
    HAS_SHORTCUTS = True
except ImportError:
    HAS_SHORTCUTS = False

# --- AI / ML IMPORTS ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Try to import bibtexparser
try:
    import bibtexparser
    HAS_BIBTEXPARSER = True
except ImportError:
    HAS_BIBTEXPARSER = False

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SRMA Deduplicator & Screener", page_icon="🧬", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #ff4b4b; }
    .metric-card { background-color: #f9f9f9; padding: 15px; border-radius: 8px; border: 1px solid #eee; text-align: center; }
    
    /* Screener specific styles */
    .paper-card { border: 1px solid #ddd; padding: 20px; border-radius: 10px; background-color: white; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .paper-title { font-size: 20px; font-weight: bold; color: #333; margin-bottom: 10px; }
    .paper-meta { color: #666; font-size: 14px; margin-bottom: 15px; font-style: italic;}
    .paper-abstract { font-size: 16px; line-height: 1.6; color: #222; background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #ff4b4b; }
    
    /* AI Badge */
    .ai-badge { background-color: #e3f2fd; color: #1565c0; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; display: inline-block; margin-bottom: 8px; }
    
    /* Keyword Tags */
    .kw-tag-inc { background-color: #e8f5e9; color: #2e7d32; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 4px; border: 1px solid #c8e6c9; }
    .kw-tag-exc { background-color: #ffebee; color: #c62828; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 4px; border: 1px solid #ffcdd2; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'active_model_index' not in st.session_state:
    st.session_state.active_model_index = 0
if 'master_df' not in st.session_state:
    st.session_state.master_df = pd.DataFrame()
if 'duplicates' not in st.session_state:
    st.session_state.duplicates = []
if 'screener_index' not in st.session_state:
    st.session_state.screener_index = 0
if 'screener_filter' not in st.session_state:
    st.session_state.screener_filter = "Unscreened"
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'is_auto_screening' not in st.session_state:
    st.session_state.is_auto_screening = False

# AI States
if 'ai_sort_enabled' not in st.session_state:
    st.session_state.ai_sort_enabled = False
if 'predicted_order' not in st.session_state:
    st.session_state.predicted_order = []
if 'decision_counter' not in st.session_state:
    st.session_state.decision_counter = 0
if 'top_inc_keywords' not in st.session_state:
    st.session_state.top_inc_keywords = []
if 'top_exc_keywords' not in st.session_state:
    st.session_state.top_exc_keywords = []

# --- PARSING & NORMALIZATION ---

def force_string(val):
    if val is None or pd.isna(val): return ""
    if isinstance(val, list): return " ".join([str(v) for v in val])
    return str(val).strip()

def normalize_string(text):
    text = force_string(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return re.sub(r'\s+', ' ', text).strip()

def normalize_doi(doi):
    doi = force_string(doi).lower()
    doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "").replace("doi:", "")
    return re.sub(r'[\[\]\(\)\s]', '', doi)

def parse_ris(file):
    try:
        content = file.getvalue().decode("utf-8", errors='ignore')
        entries = rispy.loads(content)
        data = []
        for entry in entries:
            authors = entry.get('authors', [])
            if isinstance(authors, list): authors = ", ".join(authors)
            data.append({
                'Title': force_string(entry.get('primary_title') or entry.get('title') or entry.get('ti')),
                'Year': force_string(entry.get('year') or entry.get('py') or entry.get('date')),
                'DOI': force_string(entry.get('doi') or entry.get('do')),
                'Abstract': force_string(entry.get('abstract') or entry.get('ab') or entry.get('notes')),
                'Authors': force_string(authors),
                'Journal': force_string(entry.get('journal_name') or entry.get('secondary_title') or entry.get('t2') or entry.get('jo')),
                'Pages': force_string(entry.get('start_page') or entry.get('sp')),
                'Source': file.name
            })
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame()

def parse_pubmed(file):
    content = file.getvalue().decode("utf-8", errors='ignore')
    records = re.split(r'\nPMID-', content)
    data = []
    for rec in records:
        if not rec.strip(): continue
        text = "PMID-" + rec if not rec.startswith("PMID-") else rec
        ti = re.search(r'TI\s+-\s+(.*?)(?=\n[A-Z]{2,4}\s+-|\n\n)', text, re.DOTALL)
        ab = re.search(r'AB\s+-\s+(.*?)(?=\n[A-Z]{2,4}\s+-|\n\n)', text, re.DOTALL)
        doi = re.search(r'(?:LID|AID)\s+-\s+(.*?)\s+\[doi\]', text)
        dp = re.search(r'DP\s+-\s+(\d{4})', text)
        pg = re.search(r'PG\s+-\s+(.*?)(?=\n)', text)
        jt = re.search(r'JT\s+-\s+(.*?)(?=\n)', text)
        authors = re.findall(r'FAU\s+-\s+(.*?)(?=\n)', text)
        if ti:
            data.append({
                'Title': force_string(ti.group(1).replace('\n      ', ' ')),
                'Year': force_string(dp.group(1)) if dp else "",
                'DOI': force_string(doi.group(1)) if doi else "",
                'Abstract': force_string(ab.group(1).replace('\n      ', ' ')) if ab else "",
                'Authors': ", ".join(authors) if authors else "",
                'Journal': force_string(jt.group(1)) if jt else "",
                'Pages': force_string(pg.group(1)) if pg else "",
                'Source': file.name
            })
    return pd.DataFrame(data)

def parse_csv(file):
    try:
        df = pd.read_csv(file)
        rename_map = {'Publication Year': 'Year', 'Journal/Book': 'Journal', 'Abstract Note': 'Abstract', 'Author': 'Authors', 'Url': 'DOI'}
        df = df.rename(columns=rename_map)
        expected = ['Title', 'Year', 'DOI', 'Abstract', 'Authors', 'Journal', 'Pages']
        for col in expected:
            if col not in df.columns: df[col] = ""
        df = df.fillna("")
        for col in expected: df[col] = df[col].apply(force_string)
        if 'Source' not in df.columns: df['Source'] = file.name
        return df
    except Exception as e:
        return pd.DataFrame()

def parse_bib(file):
    content = file.getvalue().decode("utf-8", errors='ignore')
    data = []
    if HAS_BIBTEXPARSER:
        try:
            parser = bibtexparser.bparser.BibTexParser(common_strings=True)
            bib_database = bibtexparser.loads(content, parser=parser)
            for entry in bib_database.entries:
                data.append({
                    'Title': force_string(entry.get('title', '').replace('\n', ' ')),
                    'Year': force_string(entry.get('year', '')),
                    'DOI': force_string(entry.get('doi', '')),
                    'Abstract': force_string(entry.get('abstract', '').replace('\n', ' ')),
                    'Authors': force_string(entry.get('author', '').replace('\n', ' ')),
                    'Journal': force_string(entry.get('journal', '')),
                    'Pages': force_string(entry.get('pages', '')),
                    'Source': file.name
                })
            return pd.DataFrame(data)
        except Exception: pass
    raw_entries = re.split(r'\n@', content)
    for raw in raw_entries:
        if not raw.strip(): continue
        entry = {'Source': file.name}
        def extract(key):
            match = re.search(key + r'\s*=\s*\{(.*?)\},', raw, re.DOTALL | re.IGNORECASE)
            return match.group(1).replace('\n', ' ').strip() if match else ""
        entry['Title'] = extract('title')
        if not entry['Title']: continue
        entry['Year'] = extract('year')
        entry['DOI'] = extract('doi')
        entry['Abstract'] = extract('abstract')
        entry['Authors'] = extract('author')
        entry['Journal'] = extract('journal')
        entry['Pages'] = extract('pages')
        for k in entry: entry[k] = force_string(entry[k])
        data.append(entry)
    return pd.DataFrame(data)

# --- REPLACED: AI AUTOSCREEN WITH REST API FOR THREAD SAFETY ---
def auto_screen_thread_safe(idx, title, abstract, api_key, protocol):
    """Thread-safe REST API caller that isolates API keys."""
    prompt = f"""
    You are an expert academic researcher conducting a Systematic Review and Meta-Analysis (SRMA).
    Evaluate the following paper against the user-defined screening protocol.
    
    ### SCREENING PROTOCOL:
    {protocol}
    
    ### PAPER TO EVALUATE:
    Title: {title}
    Abstract: {abstract}
    
    ### INSTRUCTIONS:
    1. First, write a detailed <reasoning> block where you explicitly analyze the paper against EACH criteria.
    2. Final Decision: provide your final decision inside a <decision> block. Must be exactly: Include, Exclude, or Maybe.
    """
    
    models_to_try = ['gemini-3.1-flash-lite-preview', 'gemini-3-flash-preview']
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    
                    dec_match = re.search(r'<decision>(.*?)</decision>', text, re.IGNORECASE | re.DOTALL)
                    decision = dec_match.group(1).strip().capitalize() if dec_match else "Maybe"
                    decision = ''.join(e for e in decision if e.isalnum())
                    
                    res_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.IGNORECASE | re.DOTALL)
                    reasoning = res_match.group(1).strip() if res_match else "No reasoning provided."
                    
                    if "Include" in decision: return idx, "Include", reasoning
                    elif "Exclude" in decision: return idx, "Exclude", reasoning
                    else: return idx, "Maybe", reasoning
                    
                elif response.status_code == 429: # Rate Limit
                    time.sleep(20) # Wait 20 seconds before retrying
                else:
                    break # Break retry loop on other errors (like 400 Bad Request) and try next model
                    
            except Exception as e:
                time.sleep(5)
                
    return idx, "Maybe", "All models and retries exhausted for this thread."

# --- DEDUPLICATION LOGIC ---
def calculate_similarity(row1, row2):
    t1 = normalize_string(row1['Title'])
    t2 = normalize_string(row2['Title'])
    if not t1 or not t2: return 0
    title_score = fuzz.token_set_ratio(t1, t2)
    if title_score < 75: return title_score
    
    a1 = normalize_string(row1['Authors'])
    a2 = normalize_string(row2['Authors'])
    if not a1 or not a2: auth_score = title_score 
    else: auth_score = fuzz.token_set_ratio(a1, a2)
        
    y1_str = str(row1['Year']).strip()
    y2_str = str(row2['Year']).strip()
    
    # Safely extract the 4-digit year using regex to avoid ValueError
    y1_match = re.search(r'\d{4}', y1_str)
    y2_match = re.search(r'\d{4}', y2_str)
    
    weighted_score = (title_score * 0.75) + (auth_score * 0.25)
    
    if y1_match and y2_match:
        y1_int = int(y1_match.group())
        y2_int = int(y2_match.group())
        if y1_int == y2_int: 
            weighted_score += 3
        elif abs(y1_int - y2_int) > 1: 
            weighted_score -= 10
        
    return min(100, int(weighted_score))

def find_duplicates(df, threshold):
    duplicates = []
    df['norm_title'] = df['Title'].apply(normalize_string)
    df['norm_doi'] = df['DOI'].apply(normalize_doi)
    
    doi_groups = df[df['norm_doi'] != ""].groupby('norm_doi').groups
    processed_ids = set()
    for doi, valid_indices in doi_groups.items():
        ids = list(valid_indices)
        if len(ids) > 1:
            duplicates.append({'type': 'Exact DOI', 'ids': ids, 'score': 100})
            processed_ids.update(ids)
            
    candidates = df[~df.index.isin(processed_ids)].copy()
    candidates = candidates[candidates['norm_title'].astype(str).str.len() > 10]
    if len(candidates) < 2: 
        # Sort before returning just in case there are exact DOI matches
        duplicates.sort(key=lambda x: x['score'], reverse=True)
        return duplicates
        
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(candidates['norm_title'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    indices = candidates.index.tolist()
    upper_tri_indices = np.triu_indices_from(cosine_sim, k=1)
    
    blocking_threshold = 0.5
    valid_pairs = [(i, j) for i, j in zip(*upper_tri_indices) if cosine_sim[i, j] > blocking_threshold]
    
    for i, j in valid_pairs:
        idx1 = indices[i]
        idx2 = indices[j]
        score = calculate_similarity(df.loc[idx1], df.loc[idx2])
        if score >= threshold:
            duplicates.append({'type': 'Similarity', 'ids': [idx1, idx2], 'score': score})
            
    # Sort from most similar (highest score) to least similar
    duplicates.sort(key=lambda x: x['score'], reverse=True)
    return duplicates

def auto_resolve(duplicates, df, criteria, min_similarity):
    resolved_count = 0
    for group in duplicates:
        ids = group['ids']
        active_ids = [i for i in ids if df.at[i, 'decision'] != 'Exclude']
        if len(active_ids) < 2: continue
        if group['score'] < min_similarity: continue
        base_id = active_ids[0]
        match_failed = False
        for other_id in active_ids[1:]:
            for crit in criteria:
                val1 = normalize_string(df.at[base_id, crit])
                val2 = normalize_string(df.at[other_id, crit])
                if val1 == "" or val2 == "" or val1 != val2:
                    match_failed = True
                    break
            if match_failed: break
        if not match_failed:
            best_id = max(active_ids, key=lambda x: len(str(df.at[x, 'Abstract'])))
            for pid in active_ids:
                if pid != best_id:
                    df.at[pid, 'decision'] = 'Exclude'
                    df.at[pid, 'auto_reason'] = f"Auto-Resolved ({group['score']}%)"
            resolved_count += 1
    return resolved_count

# --- AI TRAINING LOGIC (AUTO) ---
def train_ai_model(df):
    if not HAS_SKLEARN: return None, [], []
    train_df = df[df['screening_status'].isin(['Include', 'Exclude'])].copy()
    
    if len(train_df[train_df['screening_status']=='Include']) < 3 or \
       len(train_df[train_df['screening_status']=='Exclude']) < 3:
        return None, [], []
        
    train_df['target'] = train_df['screening_status'].apply(lambda x: 1 if x == 'Include' else 0)
    # Use .fillna("") to replace any hidden NaNs with empty strings
    train_text = train_df['Title'].fillna("") + " " + train_df['Abstract'].fillna("")
    y_train = train_df['target']
    
    model = make_pipeline(TfidfVectorizer(stop_words='english', max_features=1000), MultinomialNB())
    model.fit(train_text, y_train)
    
    # Do the same for the prediction dataset
    all_text = df['Title'].fillna("") + " " + df['Abstract'].fillna("")
    probs = model.predict_proba(all_text)[:, 1]
    vectorizer = model.named_steps['tfidfvectorizer']
    classifier = model.named_steps['multinomialnb']
    feature_names = vectorizer.get_feature_names_out()
    log_prob_diff = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    top_inc_indices = np.argsort(log_prob_diff)[-10:][::-1]
    inc_keywords = [feature_names[i] for i in top_inc_indices]
    top_exc_indices = np.argsort(log_prob_diff)[:10]
    exc_keywords = [feature_names[i] for i in top_exc_indices]
    return pd.Series(probs, index=df.index), inc_keywords, exc_keywords

# --- GLOBAL SIDEBAR (Workspace & AI Stats) ---
with st.sidebar:
    st.markdown("### 💾 Workspace Backup")
    if not st.session_state.master_df.empty:
        csv_buffer = st.session_state.master_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Progress File",
            data=csv_buffer,
            file_name="srma_workspace_backup.csv",
            mime="text/csv",
            help="Download your current state to avoid losing progress. Upload this in Tab 1 to resume."
        )
    else:
        st.info("Upload data to enable saving.")
        
    st.divider()
    st.markdown("### 🧠 LLM Auto-Screener")
    
    # Let user select number of threads
    st.session_state.num_threads = st.number_input("Number of AI Threads", min_value=1, max_value=10, value=1, help="More threads = faster screening, but requires more API keys.")
    
    # Generate dynamic API key inputs
    st.session_state.api_keys = []
    for i in range(st.session_state.num_threads):
        key = st.text_input(f"Gemini API Key {i+1}", type="password", key=f"api_key_{i}")
        if key.strip():
            st.session_state.api_keys.append(key.strip())
            
    st.session_state.srma_protocol = st.text_area(
        "Screening Protocol", 
        height=150, 
        placeholder="1. Include RCTs and observational studies.\n2. Exclude animal models..."
    )
    
    # --- NEW: Reset Button ---
    if st.button("🔄 Reset AI Quota & Models", help="Click this if you changed your API key or if your daily quota has reset."):
        st.session_state.active_model_index = 0
        st.session_state.is_auto_screening = False
        st.success("AI models reset! You can start auto-screening again.")
        time.sleep(1.5)
        st.rerun()
    
    st.markdown("### 🤖 Live AI Learner")
    if not st.session_state.master_df.empty:
        if not HAS_SKLEARN:
            st.error("Install `scikit-learn` to enable AI.")
        else:
            inc_count = len(st.session_state.master_df[st.session_state.master_df['screening_status'] == 'Include'])
            exc_count = len(st.session_state.master_df[st.session_state.master_df['screening_status'] == 'Exclude'])
            
            st.caption(f"Learned from: {inc_count} Inc / {exc_count} Exc")
            
            if st.session_state.ai_sort_enabled and st.session_state.top_inc_keywords:
                st.markdown("**✅ AI Likes:**")
                kw_html = "".join([f'<span class="kw-tag-inc">{kw}</span>' for kw in st.session_state.top_inc_keywords[:8]])
                st.markdown(kw_html, unsafe_allow_html=True)
                st.markdown("**⛔ AI Dislikes:**")
                kw_html_ex = "".join([f'<span class="kw-tag-exc">{kw}</span>' for kw in st.session_state.top_exc_keywords[:8]])
                st.markdown(kw_html_ex, unsafe_allow_html=True)
                st.caption("_Updates every 5 decisions_")
            elif inc_count < 3 or exc_count < 3:
                 st.info(f"Screen {max(0, 3-inc_count)} more Includes and {max(0, 3-exc_count)} more Excludes to start AI.")
    else:
        st.caption("Awaiting data...")

# --- UI LAYOUT TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📂 1. Upload", "⚡ 2. Deduplicate", "📥 3. Intermediate Export", "👁️ 4. Screener", "📊 5. Final Report", "ℹ️ About"])

# TAB 1: UPLOAD
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🆕 Start New Review")
        uploaded_files = st.file_uploader("Upload raw files (RIS, PubMed txt, CSV, BIB)", accept_multiple_files=True)
        if st.button("Process New Files", type="primary"):
            if uploaded_files:
                all_data = []
                for file in uploaded_files:
                    ext = file.name.split('.')[-1].lower()
                    if ext == 'ris': df = parse_ris(file)
                    elif ext == 'txt': df = parse_pubmed(file)
                    elif ext == 'bib': df = parse_bib(file)
                    else: df = parse_csv(file)
                    if not df.empty: all_data.append(df)
                if all_data:
                    full_df = pd.concat(all_data, ignore_index=True)
                    full_df['id'] = full_df.index
                    full_df['decision'] = 'Include'
                    full_df['screening_status'] = 'Unscreened'
                    full_df['ai_score'] = 0.0
                    for col in full_df.columns:
                        if full_df[col].dtype == 'object': full_df[col] = full_df[col].apply(force_string)
                    for c in ['Authors', 'Journal', 'Pages', 'DOI', 'Year', 'Abstract', 'ai_reasoning']:
                        if c not in full_df.columns: full_df[c] = ""
                    st.session_state.master_df = full_df
                    st.session_state.duplicates = []
                    st.session_state.is_auto_screening = False
                    st.success(f"Loaded {len(full_df)} references.")
                else: st.warning("No valid references found.")
    
    with c2:
        st.markdown("### 🔄 Resume Previous Session")
        resume_file = st.file_uploader("Upload a Workspace Backup (CSV)", type=['csv'])
        if st.button("Resume Session"):
            if resume_file:
                df = pd.read_csv(resume_file)
                for c in ['Authors', 'Journal', 'Pages', 'DOI', 'Year', 'Abstract', 'screening_status', 'decision', 'ai_score', 'ai_reasoning']:
                    if c not in df.columns:
                        df[c] = "" if c not in ['ai_score'] else 0.0
                st.session_state.master_df = df
                st.session_state.is_auto_screening = False
                st.success(f"Resumed session with {len(df)} references!")
                st.rerun()
            else:
                st.warning("Please upload a backup CSV file first.")

# TAB 2: DEDUPLICATE
with tab2:
    df = st.session_state.master_df
    if df.empty: st.info("Upload data first.")
    else:
        with st.expander("🕵️ Step 1: Find Duplicates", expanded=not bool(st.session_state.duplicates)):
            search_threshold = st.slider("Fuzzy Search Threshold", 50, 100, 80)
            if st.button("Find Duplicates", type="primary"):
                with st.spinner("Analyzing..."):
                    st.session_state.duplicates = find_duplicates(df, search_threshold)

        if st.session_state.duplicates:
            st.divider()
            with st.container():
                st.markdown("### 🤖 Auto-Resolver")
                with st.expander("Configure Auto-Resolve", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    check_authors = c1.checkbox("Authors", value=False)
                    check_journal = c2.checkbox("Journal", value=False)
                    check_pages = c3.checkbox("Pages", value=False)
                    check_title = c1.checkbox("Title", value=True)
                    check_year = c2.checkbox("Year", value=True)
                    check_doi = c3.checkbox("DOI", value=True)
                    st.markdown("---")
                    col_slide, col_val = st.columns([0.85, 0.15])
                    with col_slide: auto_thresh = st.slider("Similarity %", 0, 100, 97)
                    with col_val: st.markdown(f"**{auto_thresh}%**")
                    if st.button("Resolve Duplicates", type="primary", use_container_width=True):
                        criteria = []
                        if check_title: criteria.append('Title')
                        if check_year: criteria.append('Year')
                        if check_doi: criteria.append('DOI')
                        if check_authors: criteria.append('Authors')
                        if check_journal: criteria.append('Journal')
                        if check_pages: criteria.append('Pages')
                        count = auto_resolve(st.session_state.duplicates, df, criteria, auto_thresh)
                        if count > 0: st.rerun()

            resolved_dupes = len([d for d in st.session_state.duplicates if len([i for i in d['ids'] if df.at[i, 'decision'] != 'Exclude']) < 2])
            st.progress(resolved_dupes / len(st.session_state.duplicates))
            st.caption(f"Resolved {resolved_dupes} / {len(st.session_state.duplicates)} duplicate sets")
            st.divider()
            st.subheader("📝 Manual Review")
            unresolved = [d for d in st.session_state.duplicates if len([i for i in d['ids'] if df.at[i, 'decision'] != 'Exclude']) > 1]
            for i, group in enumerate(unresolved[:50]):
                with st.container(border=True):
                    head_col1, head_col2 = st.columns([0.8, 0.2])
                    head_col1.markdown(f"**Similarity: {group['score']}%**")
                    cols = st.columns(len(group['ids']))
                    
                    for idx, col in zip(group['ids'], cols):
                        row = df.loc[idx]
                        with col:
                            st.markdown(f"**ID: {idx}**")
                            st.markdown(f"_{row['Title']}_")
                            st.caption(f"👤 **Authors:** {str(row['Authors'])[:120]}{'...' if len(str(row['Authors'])) > 120 else ''}")
                            st.caption(f"📖 **Journal:** {row['Journal']} | 📅 **Year:** {row['Year']}")
                            if st.button("Exclude", key=f"ex_{i}_{idx}"):
                                df.at[idx, 'decision'] = 'Exclude'
                                st.rerun()

# TAB 3: EXPORT
with tab3:
    if not st.session_state.master_df.empty:
        df_final = st.session_state.master_df
        deduped = df_final[df_final['decision'] == 'Include']
        st.info("You can export the deduplicated list here as a backup, or proceed directly to the Screener tab.")
        st.metric("Deduplicated Papers", len(deduped))
        st.download_button("Download CSV (For Screening)", deduped.to_csv(index=False).encode('utf-8'), "deduplicated_refs.csv", "text/csv")
    else: st.warning("No data to export.")

# TAB 4: SCREENER
with tab4:
    st.subheader("👁️ Abstract Screener")
    df_master = st.session_state.master_df
    
    if df_master.empty:
        st.info("No data loaded. Please upload your files in Tab 1 to begin.")
    else:
        # --- Progress Bar & Live Stats ---
        total_valid = len(df_master[df_master['decision'] == 'Include'])
        screened_df = df_master[(df_master['decision'] == 'Include') & (df_master['screening_status'] != 'Unscreened')]
        screened_count = len(screened_df)
        
        if total_valid > 0:
            st.progress(screened_count / total_valid)
            st.caption(f"**Overall Screening Progress:** {screened_count} / {total_valid} ({int((screened_count/total_valid)*100)}%)")
            
            # --- NEW: Live Stats Columns ---
            st.markdown("**Live Screening Stats:**")
            stat_c1, stat_c2, stat_c3 = st.columns(3)
            stat_c1.metric("🟢 Included", len(screened_df[screened_df['screening_status'] == 'Include']))
            stat_c2.metric("🔴 Excluded", len(screened_df[screened_df['screening_status'] == 'Exclude']))
            stat_c3.metric("🟡 Maybe", len(screened_df[screened_df['screening_status'] == 'Maybe']))
            st.divider()

        # --- 🤖 AI MULTI-THREADED BATCH AUTO-SCREENER ---
        st.markdown("### 🤖 Multi-Threaded Batch Auto-Screener")
        
        if len(st.session_state.get('api_keys', [])) > 0 and st.session_state.get('srma_protocol'):
            unscreened_df = st.session_state.master_df[(st.session_state.master_df['decision'] == 'Include') & (st.session_state.master_df['screening_status'] == 'Unscreened')]
            
            if not unscreened_df.empty:
                # Check if the auto-screening loop is currently active
                if st.session_state.get('is_auto_screening', False):
                    st.warning("🔄 Auto-screening in progress...")
                    
                    # --- NEW: Functional Pause Button ---
                    if st.button("⏹️ Pause Auto-Screening", type="secondary"):
                        st.session_state.is_auto_screening = False
                        st.rerun()
                    
                    # Grab a small batch equal to the number of active threads
                    num_threads = len(st.session_state.api_keys)
                    batch_df = unscreened_df.head(num_threads)
                    
                    with st.spinner(f"Screening batch of {len(batch_df)} papers..."):
                        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                            futures = []
                            for i, (idx, row) in enumerate(batch_df.iterrows()):
                                assigned_key = st.session_state.api_keys[i % num_threads]
                                futures.append(
                                    executor.submit(
                                        auto_screen_thread_safe, 
                                        idx, row['Title'], row['Abstract'], assigned_key, st.session_state.srma_protocol
                                    )
                                )
                            
                            # Wait for this specific batch to finish
                            for future in concurrent.futures.as_completed(futures):
                                idx, decision, reasoning = future.result()
                                # Update the master dataframe
                                st.session_state.master_df.at[idx, 'screening_status'] = decision
                                st.session_state.master_df.at[idx, 'ai_reasoning'] = reasoning
                    
                    # Rerun the app to update the UI stats and process the next batch!
                    time.sleep(0.5) 
                    st.rerun()
                    
                else:
                    st.info(f"**{len(unscreened_df)}** papers waiting to be screened using **{len(st.session_state.api_keys)}** active threads.")
                    if st.button("🚀 Start Multi-Threaded Screening", type="primary"):
                        st.session_state.is_auto_screening = True
                        st.rerun()
            else:
                st.success("✅ No unscreened papers left to auto-screen!")
                st.session_state.is_auto_screening = False
        else:
            st.info("👈 Enter your Gemini API Key(s) and Screening Protocol in the sidebar to enable batch AI screening.")
        
        st.divider()

        # --- FILTER & SEARCH LOGIC ---
        col_filter, col_search = st.columns([0.4, 0.6])
        
        with col_filter:
            prev_filter = st.session_state.screener_filter
            new_filter = st.radio("Status Filter:", ["Unscreened", "Maybe", "Duplicate", "All", "Include", "Exclude"], 
                                  index=["Unscreened", "Maybe", "Duplicate", "All", "Include", "Exclude"].index(prev_filter),
                                  horizontal=True, key="screener_filter_widget")
            if new_filter != prev_filter:
                st.session_state.screener_filter = new_filter
                st.session_state.screener_index = 0
                st.rerun()

        with col_search:
            search_input = st.text_input("🔍 Search (Title, Abstract, Author):", 
                                         value=st.session_state.search_query, 
                                         placeholder="Type keywords to filter results...")
            if search_input != st.session_state.search_query:
                st.session_state.search_query = search_input
                st.session_state.screener_index = 0
                st.rerun()

        # --- APPLY FILTERS ---
        base_df = st.session_state.master_df[st.session_state.master_df['decision'] == 'Include']
        
        if new_filter == "All": 
            screen_df = base_df.copy()
        else: 
            screen_df = base_df[base_df['screening_status'] == new_filter].copy()
            
        if st.session_state.search_query:
            q = st.session_state.search_query.lower()
            mask = (
                screen_df['Title'].str.lower().str.contains(q, na=False) |
                screen_df['Abstract'].str.lower().str.contains(q, na=False) |
                screen_df['Authors'].str.lower().str.contains(q, na=False)
            )
            screen_df = screen_df[mask]

        # --- GET INDICES ---
        if st.session_state.ai_sort_enabled and new_filter == "Unscreened" and not st.session_state.search_query:
            if st.session_state.predicted_order:
                current_indices = screen_df.index
                available_indices = [idx for idx in st.session_state.predicted_order if idx in current_indices]
                extras = [idx for idx in current_indices if idx not in set(available_indices)]
                available_indices.extend(extras)
            else:
                available_indices = screen_df.index.tolist()
        elif st.session_state.ai_sort_enabled and new_filter == "Unscreened" and st.session_state.search_query:
            available_indices = screen_df.sort_values('ai_score', ascending=False).index.tolist()
        else:
            available_indices = screen_df.index.tolist()

        # Stats
        total_in_view = len(screen_df)
        st.caption(f"Showing {total_in_view} papers matching filters.")
        
        # Index Safety
        if st.session_state.screener_index >= len(available_indices) and len(available_indices) > 0:
            st.session_state.screener_index = len(available_indices) - 1
        
        if not available_indices:
            st.info(f"No papers found matching your filters.")
            if new_filter == "Unscreened" and not st.session_state.search_query: 
                st.success("You have screened all papers! Check 'Maybe' items or proceed to Final Report.")
        elif st.session_state.screener_index >= len(available_indices): 
             st.success("🎉 End of this list!")
             if st.button("Start Over for this View"):
                st.session_state.screener_index = 0
                st.rerun()
        else:
            current_idx = available_indices[st.session_state.screener_index]
            row = st.session_state.master_df.loc[current_idx]
            
            # --- CARD VIEW ---
            with st.container():
                if st.session_state.ai_sort_enabled and 'ai_score' in row:
                    score_pct = int(row['ai_score'] * 100)
                    if score_pct > 70:
                        st.markdown(f'<span class="ai-badge">🤖 AI Match: {score_pct}%</span>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="paper-card">
                    <div class="paper-title">{row['Title']}</div>
                    <div class="paper-meta">
                        👤 {str(row['Authors'])[:100]}... <br>
                        📅 {row['Year']} | 📖 {row['Journal']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if row['DOI']:
                    clean_doi = normalize_doi(row['DOI'])
                    if clean_doi: st.markdown(f"🔗 **DOI:** [{row['DOI']}](https://doi.org/{clean_doi}) (Opens in new tab)")
                
                st.markdown("**Abstract:**")
                abstract_text = row['Abstract'] if row['Abstract'] else "No abstract available."
                
                # Dynamic Search Highlighting
                if st.session_state.search_query:
                    q = st.session_state.search_query
                    abstract_text = re.sub(f"({re.escape(q)})", r'<mark>\1</mark>', abstract_text, flags=re.IGNORECASE)
                
                # Dynamic AI Keyword Highlighting
                if st.session_state.ai_sort_enabled:
                    for kw in st.session_state.top_inc_keywords[:5]:
                        abstract_text = re.sub(f"(?i)\\b({re.escape(kw)})\\b", r'<span class="kw-tag-inc">\1</span>', abstract_text)
                    for kw in st.session_state.top_exc_keywords[:5]:
                        abstract_text = re.sub(f"(?i)\\b({re.escape(kw)})\\b", r'<span class="kw-tag-exc">\1</span>', abstract_text)
                        
                st.markdown(f"""<div class="paper-abstract">{abstract_text}</div>""", unsafe_allow_html=True)

            st.divider()
            
            # --- BUTTON LOGIC ---
            def set_status(status):
                st.session_state.master_df.at[current_idx, 'screening_status'] = status
                filter_mode = st.session_state.screener_filter
                stays_visible = True
                if filter_mode != "All" and filter_mode != status: stays_visible = False
                
                if stays_visible: 
                    st.session_state.screener_index += 1
                
                if HAS_SKLEARN:
                    st.session_state.decision_counter += 1
                    if st.session_state.decision_counter >= 5:
                        st.session_state.decision_counter = 0
                        probs, inc_kw, exc_kw = train_ai_model(st.session_state.master_df)
                        if probs is not None:
                            st.session_state.master_df['ai_score'] = probs
                            st.session_state.ai_sort_enabled = True
                            st.session_state.top_inc_keywords = inc_kw
                            st.session_state.top_exc_keywords = exc_kw
                            unscreened = st.session_state.master_df[
                                (st.session_state.master_df['decision'] == 'Include') & 
                                (st.session_state.master_df['screening_status'] == 'Unscreened')
                            ]
                            st.session_state.predicted_order = unscreened.sort_values('ai_score', ascending=False).index.tolist()
                st.rerun()

            c_back, c_dup, c_ex, c_maybe, c_inc, c_next = st.columns([1, 2, 2, 2, 2, 1])
            with c_back:
                if st.button("⬅️ Prev", key="btn_prev"):
                    st.session_state.screener_index = max(0, st.session_state.screener_index - 1)
                    st.rerun()
            with c_dup:
                if st.button("👯 Duplicate", key="btn_dup", use_container_width=True): set_status('Duplicate')
            with c_ex:
                if st.button("⛔ Exclude", key="btn_ex", use_container_width=True, type="secondary"): set_status('Exclude')
            with c_maybe:
                if st.button("🤔 Maybe", key="btn_maybe", use_container_width=True): set_status('Maybe')
            with c_inc:
                if st.button("✅ Include", key="btn_inc", use_container_width=True, type="primary"): set_status('Include')
            with c_next:
                if st.button("Next ➡️", key="btn_next"):
                    st.session_state.screener_index += 1
                    st.rerun()
                    
            # --- KEYBOARD SHORTCUTS ---
            if HAS_SHORTCUTS:
                add_keyboard_shortcuts({
                    'ArrowRight': 'btn_inc',
                    'ArrowLeft': 'btn_ex',
                    'ArrowDown': 'btn_maybe',
                    'ArrowUp': 'btn_dup',
                    'Space': 'btn_next'
                })
                st.caption("⌨️ **Shortcuts enabled:** `Right Arrow` (Include) | `Left Arrow` (Exclude) | `Down Arrow` (Maybe) | `Up Arrow` (Duplicate) | `Space` (Next)")
                    
            curr_status = st.session_state.master_df.at[current_idx, 'screening_status']
            if curr_status != 'Unscreened': st.info(f"Current Status: **{curr_status}**")

    st.divider()
    with st.expander("📝 Review & Edit Screening Decisions"):
        if not st.session_state.master_df.empty:
            review_df = st.session_state.master_df[st.session_state.master_df['decision'] == 'Include'][['Title', 'screening_status', 'DOI', 'Authors']]
            edited_df = st.data_editor(
                review_df,
                column_config={"screening_status": st.column_config.SelectboxColumn("Status", options=["Unscreened", "Include", "Exclude", "Maybe", "Duplicate"], required=True)},
                disabled=["Title", "DOI", "Authors"], key="screener_editor"
            )
            if st.button("Save Changes from Table"):
                st.session_state.master_df.update(edited_df)
                st.success("Updated decisions!")
                st.rerun()

# TAB 5: FINAL REPORT
with tab5:
    st.subheader("📊 Final Report & PRISMA Flow")
    if st.session_state.master_df.empty: st.info("Process data first.")
    else:
        mdf = st.session_state.master_df
        total_raw = len(mdf)
        dedup_step_1_removed = len(mdf[mdf['decision'] == 'Exclude']) 
        dedup_step_2_removed = len(mdf[(mdf['decision'] == 'Include') & (mdf['screening_status'] == 'Duplicate')])
        total_duplicates_removed = dedup_step_1_removed + dedup_step_2_removed
        records_screened = total_raw - total_duplicates_removed
        
        included_df = mdf[(mdf['decision'] == 'Include') & (mdf['screening_status'] == 'Include')]
        excluded_df = mdf[(mdf['decision'] == 'Include') & (mdf['screening_status'] == 'Exclude')]
        maybe_df = mdf[(mdf['decision'] == 'Include') & (mdf['screening_status'] == 'Maybe')]
        unscreened_df = mdf[(mdf['decision'] == 'Include') & (mdf['screening_status'] == 'Unscreened')]
        
        final_included = len(included_df)
        final_excluded = len(excluded_df)
        maybe_count = len(maybe_df)
        unscreened_count = len(unscreened_df)
        
        # --- NEW: Calculate source breakdown ---
        source_counts = mdf['Source'].value_counts()
        # Format the breakdown for Graphviz (using \n for newlines)
        source_breakdown = "\\n".join([f"• {src}: {count}" for src, count in source_counts.items()])
        
        st.markdown("### PRISMA Flow Diagram")
        prisma_dot = f"""
        digraph PRISMA {{
            rankdir=TB;
            node [shape=box, style="filled,rounded", fillcolor="white", fontname="Arial"];
            edge [fontname="Arial"];
            id [label="Records identified from:\\n{source_breakdown}\\n\\nTotal (n = {total_raw})"];
            dedup_node [label="Records removed before screening:\\nDuplicate records removed\\n(n = {total_duplicates_removed})", fillcolor="#ffebee"];
            screen [label="Records screened\\n(n = {records_screened})"];
            excluded [label="Records excluded\\n(n = {final_excluded})", fillcolor="#ffebee"];
            included [label="Studies included in review\\n(n = {final_included})", fillcolor="#e8f5e9"];
            id -> screen;
            id -> dedup_node [style=dotted, constraint=false, dir=none]; 
            {{rank=same; id; dedup_node}}
            screen -> included;
            screen -> excluded;
        }}
        """
        st.graphviz_chart(prisma_dot)
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1. Raw Imported", total_raw)
        col2.metric("2. Duplicates Removed", total_duplicates_removed)
        col3.metric("3. Valid Screened", records_screened)
        col4.metric("4. Final Included", final_included)
        
        st.divider()
        if maybe_count > 0 or unscreened_count > 0:
            st.error("⚠️ Finalization Blocked")
            st.markdown(f"**You cannot export the final report yet.** Please resolve: **{maybe_count}** 'Maybe' and **{unscreened_count}** 'Unscreened'.")
        else:
            st.success("✅ Review Complete!")
            c_down1, c_down2 = st.columns(2)
            with c_down1: st.download_button("📥 Download INCLUDED Studies (CSV)", included_df.to_csv(index=False).encode('utf-8'), "final_included_studies.csv", "text/csv", type="primary")
            with st.expander("View Full Data Table"): st.dataframe(included_df)

# TAB 6: ABOUT
with tab6:
    st.markdown("## ℹ️ About")
    col_info, col_cite = st.columns(2)
    with col_info:
        st.markdown("### Developer Information")
        st.markdown("**Name:** Azza Fithra Alhanifa  \n**Affiliation:** Faculty of Medicine, Universitas Udayana  \n**Location:** Denpasar, Bali, Indonesia  \n**Year:** 2026")
    with col_cite:
        st.markdown("### 📚 How to Cite (APA)")
        st.code("""Alhanifa, A. F. (2026). SRMA Deduplicator & Screener [Software]. Faculty of Medicine, Universitas Udayana.""", language="text")
    st.divider()
    st.markdown("### 🛠️ Built With")
    st.markdown("* **Streamlit**: Web framework.\n* **Pandas**: Data handling.\n* **RapidFuzz**: Deduplication.\n* **Scikit-learn**: AI/Machine Learning.\n* **Rispy/Graphviz**: Parsing & Visualization.")
    st.caption("© 2026 Azza Fithra Alhanifa. All Rights Reserved.")
