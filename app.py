import os

from sentence_transformers import SentenceTransformer
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import faiss
import numpy as np
from dotenv import load_dotenv
import os
# ------------------- Configuration ------------------- #
load_dotenv()
GEMINI_API_KEY = os.getenv("google_key")  # 🔁 Replace this
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

embedding_model = load_embedding_model()

# ------------------- PDF Processing ------------------- #
@st.cache_data(show_spinner="📖 Extracting text...")
def extract_text_cached(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_chars=500):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

@st.cache_data(show_spinner="📐 Creating embeddings...")
def embed_chunks(chunks):
    vectors = embedding_model.encode(chunks)
    return np.array(vectors).astype("float32")

def search_chunks(query, chunks, chunk_vectors, k=3):
    query_vec = embedding_model.encode([query]).astype("float32")
    index = faiss.IndexFlatL2(chunk_vectors.shape[1])
    index.add(chunk_vectors)
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

def ask_gemini_stream(history, context_chunks, new_question):
    context = "\n\n".join(context_chunks)
    chat = "\n".join(history)
    prompt = f"""You are a helpful assistant answering questions about the uploaded PDF. Use the context and prior conversation.

PDF Context:
\"\"\"
{context}
\"\"\"

Conversation:
{chat}

Q: {new_question}
A:"""
    model = genai.GenerativeModel("gemini-2.0-flash")  # Or "gemini-pro"
    stream = model.generate_content(prompt, stream=True)
    return stream

# ------------------- Streamlit App ------------------- #
st.set_page_config(page_title="Chat With Pdf", layout="wide")

# Sidebar – Upload, Clear Chat
st.sidebar.title("📁 Upload & Analyze")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# Detect PDF change using a hash
if uploaded_file is not None:
    file_hash = hash(uploaded_file.getvalue())  # Hash file content

    if st.session_state.get("last_file_hash") != file_hash:
        # New file uploaded — reset state
        st.session_state.chat_history = []
        st.session_state.pdf_chunks = None
        st.session_state.chunk_vectors = None
        st.session_state.last_file_hash = file_hash
        st.rerun()  # Re-run app with clean state

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = None

if "chunk_vectors" not in st.session_state:
    st.session_state.chunk_vectors = None

# Process PDF
if uploaded_file and st.session_state.pdf_chunks is None:
    file_bytes = uploaded_file.read()
    text = extract_text_cached(file_bytes)
    if not text.strip():
        st.warning("❌ The PDF has no readable text.")
        st.stop()
    chunks = chunk_text(text)
    vectors = embed_chunks(chunks)
    st.session_state.pdf_chunks = chunks
    st.session_state.chunk_vectors = vectors
    st.sidebar.success("✅ PDF processed!")

# Main Chat Area
st.title("🧠 Chat with your PDF")

# Display previous messages
for entry in st.session_state.chat_history:
    if entry.startswith("Q:"):
        st.chat_message("user").write(entry[2:].strip())
    elif entry.startswith("A:"):
        st.chat_message("assistant").write(entry[2:].strip())

# Chat input and handling
if st.session_state.pdf_chunks:
    user_input = st.chat_input("Ask something about the PDF...")

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.chat_history.append(f"Q: {user_input}")

        top_chunks = search_chunks(user_input, st.session_state.pdf_chunks, st.session_state.chunk_vectors)

        with st.chat_message("assistant"):
            response_container = st.empty()
            stream = ask_gemini_stream(st.session_state.chat_history, top_chunks, user_input)
            full_response = ""
            for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    response_container.markdown(full_response)
            st.session_state.chat_history.append(f"A: {full_response}")
else:
    st.info("📄 Upload a PDF from the sidebar to begin.")
