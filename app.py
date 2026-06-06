import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(page_title="Mini Search Engine", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

model = load_model()

st.title("🔍 Mini Search Engine")

uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if "index" not in st.session_state:
        all_chunks = []
        file_names = []

        with st.spinner("Indexing documents..."):
            for uploaded_file in uploaded_files:
                text = extract_text_from_pdf(uploaded_file)
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                file_names.extend([uploaded_file.name] * len(chunks))

            embeddings = model.encode(all_chunks, convert_to_tensor=True)
            st.session_state["index"] = {"chunks": all_chunks, "embeddings": embeddings, "files": file_names}
            st.success("Indexing complete!")

    query = st.text_input("Enter your search query:")

    if query:
        query_embedding = model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, st.session_state["index"]["embeddings"])[0]
        top_results = torch.topk(cos_scores, k=min(5, len(cos_scores)))

        st.subheader("Search Results:")
        for score, idx in zip(top_results[0], top_results[1]):
            idx = idx.item()
            with st.container():
                st.markdown(f"**Source:** {st.session_state['index']['files'][idx]}")
                st.markdown(f"**Score:** {score:.4f}")
                st.write(st.session_state["index"]["chunks"][idx])
                st.divider()
