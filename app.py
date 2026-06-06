import streamlit as st
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# ─── Load secrets from .env ───────────────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME = "mini-search-engine"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mini Search Engine",
    page_icon="🔍",
    layout="centered",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hero section */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 2px dashed rgba(167,139,250,0.4);
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(167,139,250,0.8);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.25s ease;
    width: 100%;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124,58,237,0.55);
}

/* Text input */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.07);
    border: 1.5px solid rgba(167,139,250,0.35);
    border-radius: 12px;
    color: #e2e8f0;
    font-size: 1rem;
    padding: 0.65rem 1rem;
    transition: border-color 0.25s;
}
.stTextInput > div > div > input:focus {
    border-color: #a78bfa;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.2);
}

/* Result cards */
.result-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(124,58,237,0.25);
}
.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
}
.result-filename {
    font-weight: 600;
    font-size: 1rem;
    color: #a78bfa;
}
.result-score {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.82rem;
    font-weight: 600;
}
.result-text {
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.65;
}

/* Section labels */
.section-label {
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08);
    margin: 1.8rem 0;
}

/* Hide sidebar toggle & default streamlit elements */
[data-testid="collapsedControl"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load embedding model (cached) ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def extract_text_from_pdf(file) -> str:
    """Extract plain text from an uploaded PDF file object."""
    try:
        file.seek(0)
        file_bytes = file.read()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def get_pinecone_index():
    """Connect to Pinecone and return the index (create if needed)."""
    if not PINECONE_API_KEY:
        st.error("❌ PINECONE_API_KEY not found in .env file.")
        st.stop()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)


# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 Mini Search Engine</h1>
    <p>Upload your PDFs, index them, then search using natural language</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Step 1: Upload PDFs ────────────────────────────────────────────────────────
st.markdown('<p class="section-label">📂 Step 1 — Upload PDFs (minimum 5)</p>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    label="Drag & drop PDF files here",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    remaining = max(0, 5 - len(uploaded_files))
    if remaining > 0:
        st.warning(f"⚠️ Please upload {remaining} more PDF(s) to meet the minimum requirement.")
    else:
        st.success(f"✅ {len(uploaded_files)} PDF(s) ready to index.")

st.markdown("")

# ── Step 2: Index Button ───────────────────────────────────────────────────────
st.markdown('<p class="section-label">⚙️ Step 2 — Index Documents</p>', unsafe_allow_html=True)
if st.button("🚀 Index Documents", disabled=not uploaded_files):
    if not uploaded_files:
        st.error("Please upload at least one PDF first.")
    else:
        index = get_pinecone_index()
        progress_bar = st.progress(0, text="Starting indexing…")
        total = len(uploaded_files)
        all_vectors = []

        for file_idx, uf in enumerate(uploaded_files):
            progress_bar.progress(
                int((file_idx / total) * 80),
                text=f"Processing: {uf.name}",
            )
            text = extract_text_from_pdf(uf)
            chunks = chunk_text(text)
            if chunks:
                embeddings = model.encode(chunks, show_progress_bar=False)
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    all_vectors.append({
                        "id": f"{uf.name}__chunk_{i}",
                        "values": emb.tolist(),
                        "metadata": {
                            "filename": uf.name,
                            "text": chunk[:1000],
                        },
                    })

        # Upsert in batches of 100
        progress_bar.progress(85, text="Uploading to Pinecone…")
        batch_size = 100
        for b in range(0, len(all_vectors), batch_size):
            index.upsert(vectors=all_vectors[b : b + batch_size])

        progress_bar.progress(100, text="Done!")
        st.success(f"✅ Indexed **{len(all_vectors)}** chunks from **{total}** PDF(s) into Pinecone.")

st.markdown("---")

# ── Step 3: Search ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">🔎 Step 3 — Search</p>', unsafe_allow_html=True)
query = st.text_input("", placeholder="Type your query here and press Enter…", key="search_query")

if query and query.strip():
    with st.spinner("Searching…"):
        try:
            index = get_pinecone_index()
            query_emb = model.encode(query.strip()).tolist()
            results = index.query(vector=query_emb, top_k=5, include_metadata=True)
        except Exception as e:
            st.error(f"Search error: {e}")
            results = None

    if results and results.matches:
        st.markdown(f"### 📋 Top {len(results.matches)} Results")
        for match in results.matches:
            filename = match.metadata.get("filename", "Unknown")
            score = match.score
            text = match.metadata.get("text", "")
            st.markdown(f"""
<div class="result-card">
    <div class="result-header">
        <span class="result-filename">📄 {filename}</span>
        <span class="result-score">Score: {score:.4f}</span>
    </div>
    <div class="result-text">{text}</div>
</div>
""", unsafe_allow_html=True)
    elif results:
        st.info("🔍 No matching results found. Try a different query or index more documents.")
