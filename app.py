import streamlit as st
import requests
import os
import uuid  # NEW: For generating unique IDs

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

# --- CSS STYLING (Keep your existing CSS here) ---
st.markdown("""<style>.stApp { background-color: #f8f9fa; }</style>""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
# Assign a unique ID to this browser tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) 
    
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2814/2814666.png", width=60)
    st.title("DocuMind Control")
    # Show the user their unique ID (Good for debugging)
    st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if st.session_state.file_processed:
        if st.button("🔄 Reset Session", use_container_width=True):
            # Send the ID to the API to delete the correct database
            requests.post(f"{API_URL}/reset", json={"session_id": st.session_state.session_id})
            st.session_state.file_processed = False
            st.session_state.messages = []
            # Generate a NEW ID for the next document
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    if uploaded_file and not st.session_state.file_processed:
        if st.button("🚀 Process Document", use_container_width=True):
            with st.spinner("⚙️ Processing Document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    # Pass the session_id directly in the URL as a query parameter
                    response = requests.post(f"{API_URL}/upload?session_id={st.session_state.session_id}", files=files)
                    
                    if response.status_code == 200:
                        st.session_state.file_processed = True
                        st.session_state.messages = [{"role": "assistant", "content": "Ready! Ask me anything."}]
                        st.rerun()
                    else:
                        st.error(f"Backend Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    if st.session_state.file_processed:
        st.subheader("📝 Tools")
        if st.button("📄 Generate Summary", use_container_width=True):
            with st.spinner("✍️ Summarizing..."):
                # Pass the session_id in the JSON body
                resp = requests.post(f"{API_URL}/summarize", json={"session_id": st.session_state.session_id})
                if resp.status_code == 200:
                    summary = resp.json().get("summary")
                    st.session_state.messages.append({"role": "assistant", "content": f"### 📝 Summary\n\n{summary}"})
                else:
                    st.error("Could not generate summary.")

# --- MAIN PAGE ---
st.markdown('<div class="main-header">DocuMind AI</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.file_processed:
        st.warning("Please upload and process a document first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            try:
                # Pass the session_id in the JSON body
                payload = {
                    "question": prompt,
                    "session_id": st.session_state.session_id
                }
                response = requests.post(f"{API_URL}/ask", json=payload)
                
                if response.status_code == 200:
                    answer = response.json().get("answer")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Backend failed to respond.")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")