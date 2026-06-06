import streamlit as st
import fitz
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Mini Search Engine", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file):
    try:
        file.seek(0)
        file_bytes = file.read()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    except:
        return ""

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

model = load_model()

with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Pinecone API Key", value=os.getenv("PINECONE_API_KEY", ""), type="password")
    index_name = st.text_input("Index Name", value="mini-search-engine")

st.title("🔍 Mini Search Engine")

uploaded_files = st.file_uploader("Upload at least 5 PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) < 5:
        st.warning(f"Please upload {5 - len(uploaded_files)} more PDFs.")

    if st.button("Index Documents"):
        if not api_key:
            st.error("Please provide a Pinecone API Key.")
        else:
            try:
                pc = Pinecone(api_key=api_key)
                if index_name not in [idx.name for idx in pc.list_indexes()]:
                    pc.create_index(name=index_name, dimension=384, metric='cosine',
                                    spec=ServerlessSpec(cloud='aws', region='us-east-1'))

                index = pc.Index(index_name)

                with st.spinner("Indexing..."):
                    for uf in uploaded_files:
                        text = extract_text_from_pdf(uf)
                        chunks = chunk_text(text)
                        if chunks:
                            embeddings = model.encode(chunks)
                            vectors = []
                            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                                vectors.append({
                                    "id": f"{uf.name}_{i}",
                                    "values": emb.tolist(),
                                    "metadata": {"filename": uf.name, "text": chunk[:1000]}
                                })
                            index.upsert(vectors=vectors)
                st.success("Indexing complete!")
            except Exception as e:
                st.error(f"Error: {e}")

query = st.text_input("Search query:")
if query:
    if not api_key:
        st.error("Please provide a Pinecone API Key.")
    else:
        try:
            pc = Pinecone(api_key=api_key)
            index = pc.Index(index_name)
            query_emb = model.encode(query).tolist()
            results = index.query(vector=query_emb, top_k=5, include_metadata=True)

            for match in results.matches:
                with st.expander(f"📄 {match.metadata['filename']} (Score: {match.score:.4f})"):
                    st.write(match.metadata['text'])
        except Exception as e:
            st.error(f"Search error: {e}")
