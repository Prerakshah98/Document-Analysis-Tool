# DocuMind: AI Document Analysis Microservice

An advanced AI-powered application that transforms static PDF documents into interactive knowledge bases. Rebuilt as a **decoupled client-server microservice**, it safely handles multiple concurrent users and uses a **Dual-Architecture** approach: **RAG (Retrieval Augmented Generation)** for precise Q&A and **Context Stuffing** for full document summarization.

---

## 🚀 Key Features

- **Decoupled Architecture:** A stateless FastAPI backend handles all heavy AI processing, keeping the Streamlit frontend fast and responsive.
- **Multi-User Isolation:** Generates unique UUIDs for every user session, mapping them to isolated ChromaDB instances to prevent data leaks between concurrent users.
- **Cloud-Optimized Memory:** Uses Google's Embedding APIs to offload heavy tensor computations, allowing the backend to run flawlessly on memory-constrained cloud tiers (like Render's 512MB limit).
- **Automated Garbage Collection:** Employs FastAPI Background Tasks to silently sweep and delete orphaned database folders when users abandon their sessions.
- **Dual Intelligence Engine:** Uses RAG (`k=5` chunks) for accurate question answering and Context Stuffing for broad, bullet-point document summaries.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Backend API** | FastAPI, Uvicorn |
| **Frontend UI** | Streamlit |
| **Orchestration** | LangChain |
| **Vector Database** | ChromaDB (Local) |
| **LLM** | Google Gemma-3-27b-it (via Google Gemini API) |
| **Embeddings** | Google Gemini Embedding API (`models/gemini-embedding-001`) |

---

## ⚙️ Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Prerakshah98/Document-Analysis-Tool.git
cd Document-Analysis-Tool
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root folder and add your Google API key:
```plaintext
GOOGLE_API_KEY=your_actual_api_key_here
```

### 4. Run the Backend (Terminal 1)

Start the FastAPI server:
```bash
uvicorn api:app --reload --port 8000
```

### 5. Run the Frontend (Terminal 2)

> **Note:** Ensure `API_URL` in `app.py` is set to `http://localhost:8000` for local testing.
```bash
streamlit run app.py
```

---

## 🧠 System Design Highlights

### 1. Stateless HTTP & UUID Memory

Because HTTP is stateless, the backend doesn't remember users between clicks. This is solved by generating a **UUID4** on the frontend and passing it as a query parameter in every REST API call. The backend uses this UUID as a key in a runtime dictionary to route queries to the correct local database folder.

### 2. Chunking Strategy

LangChain's `RecursiveCharacterTextSplitter` is used with a `chunk_size` of `1000` and a `chunk_overlap` of `200`. This creates a **"sliding window"** effect, ensuring sentences cut off at the end of one chunk are repeated at the start of the next, preventing context loss at chunk boundaries.

### 3. API Offloading

Instead of using local HuggingFace embedding models (which require ~800MB of RAM for PyTorch), raw text chunks are routed to **Google's Embedding API**. This drastically reduces the server's memory footprint to under **150MB**, making it perfectly suited for free-tier cloud deployments.

---

## 📁 Project Structure
```
Document-Analysis-Tool/
├── api.py              # FastAPI backend — handles uploads, querying, and cleanup
├── app.py              # Streamlit frontend — user interface
├── rag_logic.py        # core RAG code
├── requirements.txt    # Python dependencies
└── .env                # Environment variables (not committed)
```

---

## 🌐 Deployment

This project is designed to be deployed as two separate services:

- **Backend (FastAPI):** Deploy on [Render](https://render.com), [Railway](https://railway.app), or any platform supporting Python.
- **Frontend (Streamlit):** Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud).

After deploying the backend, update `API_URL` in `app.py` to point to your live backend URL before deploying the frontend.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
