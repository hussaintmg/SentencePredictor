from dotenv import load_dotenv
import streamlit as st
import os
import time
from pathlib import Path

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VectorSearch — Mini Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY", "")
index_name = os.getenv("PINECONE_INDEX_NAME", "vectorsearch-demo")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

/* ── Root palette ── */
:root {
    --ink:       #0d0d0d;
    --paper:     #f5f2eb;
    --cream:     #ede9e0;
    --gold:      #c8a84b;
    --gold-pale: #f0e4bb;
    --rust:      #b5441a;
    --muted:     #6b6456;
    --border:    #d4cfc4;
    --card-bg:   #fffef9;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--paper) !important;
    color: var(--ink);
}

/* hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 1100px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3.5rem 2rem 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 39px,
        var(--border) 39px,
        var(--border) 40px
    );
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    line-height: 1.05;
    color: var(--ink);
    margin: 0 0 0.5rem;
}
.hero-title em { color: var(--rust); font-style: italic; }
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    margin-top: 0.5rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--cream) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: var(--ink);
    border-bottom: 2px solid var(--gold);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.5rem;
}

/* ── Search bar wrapper ── */
.search-wrap {
    background: var(--card-bg);
    border: 1.5px solid var(--border);
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 4px 4px 0 var(--gold-pale);
}

/* ── Result card ── */
.result-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-left: 4px solid var(--gold);
    border-radius: 2px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.25rem;
    position: relative;
    transition: box-shadow 0.2s;
}
.result-card:hover {
    box-shadow: 5px 5px 0 var(--gold-pale);
}
.result-rank {
    position: absolute;
    top: -1px; right: 1.2rem;
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: var(--gold-pale);
    line-height: 1;
    pointer-events: none;
    user-select: none;
}
.result-meta {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
    flex-wrap: wrap;
}
.result-filename {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    background: var(--ink);
    color: var(--paper);
    padding: 0.15rem 0.55rem;
    border-radius: 2px;
    letter-spacing: 0.04em;
}
.result-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--rust);
    font-weight: 600;
    border: 1px solid var(--rust);
    padding: 0.1rem 0.45rem;
    border-radius: 2px;
}
.result-chunk {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 0.6rem;
    font-family: 'DM Mono', monospace;
}
.result-text {
    font-size: 0.95rem;
    line-height: 1.75;
    color: var(--ink);
    font-weight: 300;
}

/* ── Status pills ── */
.pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    margin: 0.15rem;
}
.pill-green  { background: #d4edda; color: #1a5c2a; border: 1px solid #a3cfa9; }
.pill-yellow { background: #fff3cd; color: #7a5c00; border: 1px solid #ffd97a; }
.pill-red    { background: #fde8e4; color: #7a1c0a; border: 1px solid #f0a898; }

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--card-bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 3px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    color: var(--ink) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-pale) !important;
}
.stButton > button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.8rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: var(--rust) !important;
}
div[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 4px !important;
    background: var(--card-bg) !important;
    padding: 0.5rem !important;
}
.stSlider > div { padding-top: 0.3rem; }
.stNumberInput > div > div > input {
    background: var(--card-bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 3px !important;
    font-family: 'DM Mono', monospace !important;
}
.stAlert { border-radius: 3px !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Semantic Document Intelligence</div>
    <div class="hero-title">Vector<em>Search</em></div>
    <div class="hero-sub">Upload PDFs · Generate Embeddings · Search with Natural Language</div>
</div>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_pinecone_index(api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    return pc.Index(index_name)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def index_pdfs(uploaded_files, index, model, chunk_size: int, overlap: int):
    all_ids = []
    progress = st.progress(0, text="Preparing…")
    total = len(uploaded_files)

    for fi, uf in enumerate(uploaded_files):
        progress.progress((fi) / total, text=f"Processing **{uf.name}** ({fi+1}/{total})…")
        text = extract_text_from_pdf(uf.read())
        chunks = chunk_text(text, chunk_size, overlap)
        embeddings = model.encode(chunks, show_progress_bar=False)

        vectors = []
        for ci, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vid = f"{Path(uf.name).stem}__chunk_{ci}"
            vectors.append({
                "id": vid,
                "values": emb.tolist(),
                "metadata": {
                    "filename": uf.name,
                    "chunk_index": ci,
                    "text": chunk[:1000],
                },
            })
            all_ids.append(vid)

        # upsert in batches of 100
        for batch_start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[batch_start : batch_start + 100])

    progress.progress(1.0, text="✅ All documents indexed!")
    time.sleep(0.8)
    progress.empty()
    return all_ids

def search(query: str, index, model, top_k: int):
    emb = model.encode([query])[0].tolist()
    results = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return results.matches

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.divider()
    st.markdown("**Chunking settings**")
    chunk_size = st.slider("Chunk size (words)", 100, 1000, 500, 50)
    overlap = st.slider("Overlap (words)", 0, 200, 50, 10)

    st.divider()
    st.markdown("**Search settings**")
    top_k = st.slider("Top-K results", 1, 20, 5)

    st.divider()
    st.markdown("**Model**")
    st.markdown(
        '<span class="pill pill-green">all-MiniLM-L6-v2</span>'
        '<span class="pill pill-yellow">384-dim</span>',
        unsafe_allow_html=True,
    )
    st.caption("Sentence-BERT model loaded locally. Embeddings are generated on-device.")

# ── Main layout ───────────────────────────────────────────────────────────────
col_upload, col_search = st.columns([1, 1], gap="large")

# ── Upload column ─────────────────────────────────────────────────────────────
with col_upload:
    st.markdown('<div class="section-label">Step 1 — Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF files (minimum 5)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for uf in uploaded_files:
            size_kb = len(uf.getvalue()) / 1024
            icon = "✅" if size_kb > 1 else "⚠️"
            st.markdown(f"{icon} `{uf.name}` — {size_kb:.1f} KB")

    if len(uploaded_files) > 0 and len(uploaded_files) < 5:
        st.warning(f"Please upload at least 5 PDFs. ({5 - len(uploaded_files)} more needed)")

    index_btn = st.button(
        "⚡ Index Documents",
        disabled=(len(uploaded_files) < 5 or not api_key or not index_name),
        use_container_width=True,
    )

    if index_btn:
        if not api_key:
            st.error("Enter your Pinecone API key in the sidebar.")
        else:
            try:
                with st.spinner("Connecting to Pinecone…"):
                    index = get_pinecone_index(api_key, index_name)
                model = load_model()
                # Reset file pointers
                for uf in uploaded_files:
                    uf.seek(0)
                ids = index_pdfs(uploaded_files, index, model, chunk_size, overlap)
                st.session_state["indexed"] = True
                st.session_state["index_obj"] = index
                st.success(f"Indexed **{len(ids)} chunks** from {len(uploaded_files)} PDFs.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

# ── Search column ─────────────────────────────────────────────────────────────
with col_search:
    st.markdown('<div class="section-label">Step 2 — Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="search-wrap">', unsafe_allow_html=True)

    query = st.text_input(
        "Natural language query",
        placeholder="e.g. What are the main findings of the study?",
        label_visibility="collapsed",
    )

    search_btn = st.button("🔍 Search", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if search_btn:
        if not query.strip():
            st.warning("Enter a search query.")
        elif not api_key or not index_name:
            st.warning("Configure Pinecone credentials in the sidebar.")
        else:
            try:
                if "index_obj" not in st.session_state:
                    with st.spinner("Connecting to Pinecone…"):
                        st.session_state["index_obj"] = get_pinecone_index(api_key, index_name)

                model = load_model()
                index = st.session_state["index_obj"]

                with st.spinner("Searching…"):
                    matches = search(query, index, model, top_k)

                if not matches:
                    st.info("No results found. Make sure documents are indexed.")
                else:
                    st.markdown(f"### Results for *\"{query}\"*")
                    st.caption(f"{len(matches)} result(s) retrieved · cosine similarity")

                    for rank, match in enumerate(matches, 1):
                        meta = match.metadata or {}
                        filename = meta.get("filename", "unknown.pdf")
                        chunk_idx = meta.get("chunk_index", "?")
                        text = meta.get("text", "")
                        score = match.score

                        score_pct = f"{score * 100:.1f}%"

                        st.markdown(f"""
<div class="result-card">
    <div class="result-rank">{rank}</div>
    <div class="result-meta">
        <span class="result-filename">📄 {filename}</span>
        <span class="result-score">similarity {score_pct}</span>
    </div>
    <div class="result-chunk">chunk #{chunk_idx}</div>
    <div class="result-text">{text}</div>
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Search failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:var(--muted,#888);font-size:0.78rem;font-family:DM Mono,monospace;'>"
    "VectorSearch · Streamlit + Pinecone + Sentence-BERT · Semantic PDF Search"
    "</center>",
    unsafe_allow_html=True,
)
