import os
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import faiss
import numpy as np
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# ------------------- Configuration ------------------- #
load_dotenv()

# Try to get API key from Streamlit secrets first, then from environment
try:
    GEMINI_API_KEY = st.secrets["google_key"]
except:
    GEMINI_API_KEY = os.getenv("google_key")

# If no API key found, ask user to input it
if not GEMINI_API_KEY:
    st.sidebar.header("🔑 API Configuration")
    GEMINI_API_KEY = st.sidebar.text_input(
        "Enter your Google API Key:", 
        type="password",
        help="Get your API key from Google AI Studio"
    )
    
    if not GEMINI_API_KEY:
        st.error("🔑 Please enter your Google API key in the sidebar to continue")
        st.info("💡 Get your free API key from: https://makersuite.google.com/app/apikey")
        st.stop()

genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_embedding_model():
    """Load the embedding model and tokenizer"""
    model_name = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_texts(texts, tokenizer, model, max_length=512):
    """Encode texts to embeddings"""
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings.numpy()

# Load models once
tokenizer, embedding_model = load_embedding_model()

# ------------------- PDF Processing ------------------- #
@st.cache_data(show_spinner="📖 Extracting text...")
def extract_text_cached(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    total_pages = len(doc)
    
    # Extract text from each page
    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text()
        
        # Add page separator and text
        if page_text.strip():  # Only add if page has content
            full_text += f"\n--- PAGE {page_num + 1} ---\n"
            full_text += page_text + "\n"
    
    doc.close()
    return full_text, total_pages

def chunk_text(text, max_chars=1000, overlap=100):
    """
    Create overlapping chunks to ensure no information is lost
    """
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed max_chars
        if len(current_chunk) + len(para) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap: keep last part of current chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n" + para
            else:
                # If single paragraph is too long, split it
                if len(para) > max_chars:
                    words = para.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= max_chars:
                            temp_chunk += word + " "
                        else:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word + " "
                    current_chunk = temp_chunk
                else:
                    current_chunk = para
        else:
            current_chunk += "\n" + para if current_chunk else para
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks

@st.cache_data(show_spinner="📐 Creating embeddings...")
def embed_chunks(chunks):
    vectors = encode_texts(chunks, tokenizer, embedding_model)
    return vectors.astype("float32")

def search_chunks(query, chunks, chunk_vectors, k=5):
    """
    Search for relevant chunks - increased k to get more context
    """
    query_vec = encode_texts([query], tokenizer, embedding_model).astype("float32")
    index = faiss.IndexFlatL2(chunk_vectors.shape[1])
    index.add(chunk_vectors)
    _, indices = index.search(query_vec, k)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def ask_gemini_stream(history, context_chunks, new_question):
    context = "\n\n".join(context_chunks)
    chat = "\n".join(history[-10:])  # Keep last 10 messages for context
    
    prompt = f"""You are a helpful AI assistant answering questions about the uploaded PDF document. Use the provided context from the document and the conversation history to give accurate, detailed answers.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{chat}

INSTRUCTIONS:
- Base your answer primarily on the document content provided
- If the answer isn't in the provided context, say so clearly
- Be specific and cite relevant parts when possible
- If asked about the entire document, use your knowledge from all the chunks processed

QUESTION: {new_question}

ANSWER:"""

    model = genai.GenerativeModel("gemini-2.0-flash")
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

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

# Process PDF
if uploaded_file and st.session_state.pdf_chunks is None:
    file_bytes = uploaded_file.read()
    text, total_pages = extract_text_cached(file_bytes)
    
    if not text.strip():
        st.warning("❌ The PDF has no readable text.")
        st.stop()
    
    # Show document statistics
    st.sidebar.info(f"📄 **Document Info:**\n- Pages: {total_pages}\n- Characters: {len(text):,}\n- Words: ~{len(text.split()):,}")
    
    chunks = chunk_text(text)
    st.sidebar.info(f"📝 Created {len(chunks)} text chunks")
    
    vectors = embed_chunks(chunks)
    st.session_state.pdf_chunks = chunks
    st.session_state.chunk_vectors = vectors
    st.session_state.total_pages = total_pages
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
    # Show document stats in main area
    if st.session_state.total_pages > 0:
        st.info(f"📚 Document loaded: {st.session_state.total_pages} pages, {len(st.session_state.pdf_chunks)} chunks available for search")
    
    user_input = st.chat_input("Ask something about the PDF...")

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.chat_history.append(f"Q: {user_input}")

        # Get more relevant chunks for better coverage
        top_chunks = search_chunks(user_input, st.session_state.pdf_chunks, st.session_state.chunk_vectors, k=5)

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