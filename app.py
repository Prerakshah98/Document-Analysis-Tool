import streamlit as st
import os

# Import backend logic
from rag_logic import load_and_process_pdf, ask_question, summarize_document

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #1a202c;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #4a5568;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Chat Message Styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    /* User Message (Right/Blue) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #e6f3ff;
        border: 1px solid #cce5ff;
    }
    /* AI Message (Left/White) */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #3182ce;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #2b6cb0;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* File Uploader Clean Look */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e0;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "docs_list" not in st.session_state:
    st.session_state.docs_list = None 

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2814/2814666.png", width=60)
    st.title("DocuMind Control")
    st.markdown("---")
    
    st.subheader("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    
    # Logic: Handle File Removal
    if uploaded_file is None and st.session_state.vector_db is not None:
        st.warning("⚠️ File removed.")
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.vector_db = None
            st.session_state.docs_list = None
            st.session_state.messages = []
            st.rerun()

    # Logic: Process File
    if uploaded_file:
        temp_path = "temp_uploaded.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 Process Document", use_container_width=True):
            with st.spinner("⚙️ Analyzing document structure..."):
                try:
                    # Release old DB ref
                    st.session_state.vector_db = None 
                    
                    # Trigger backend (Creates a NEW unique DB folder)
                    db, docs = load_and_process_pdf(temp_path)
                    
                    # Store results
                    st.session_state.vector_db = db
                    st.session_state.docs_list = docs
                    st.session_state.messages = []
                    st.session_state.messages.append({"role": "assistant", "content": "Hello! I've read your document. Ask me anything or request a summary!"})
                    
                    st.toast("Document processed successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    
    # Summarization
    if st.session_state.docs_list:
        st.subheader("📝 Tools")
        if st.button("📄 Generate Summary", use_container_width=True):
            with st.spinner("✍️ Writing summary..."):
                try:
                    summary = summarize_document(st.session_state.docs_list)
                    st.session_state.messages.append({"role": "assistant", "content": f"### 📝 Document Summary\n\n{summary}"})
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Upload a PDF to unlock tools.")

# --- MAIN PAGE ---
st.markdown('<div class="main-header">DocuMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Document Analysis & Q&A Engine</div>', unsafe_allow_html=True)

# Chat Area
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input Area
if prompt := st.chat_input("Type your question here..."):
    if not st.session_state.vector_db:
        st.warning("Please upload a document first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ask_question(st.session_state.vector_db, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")