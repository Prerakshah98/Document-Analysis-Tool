# DocuMind: AI Document Analysis Tool

An advanced AI-powered application that transforms static PDF documents into interactive knowledge bases. It uses a Dual-Architecture approach to handle different user needs: **RAG (Retrieval Augmented Generation)** for precise questions and **Context Stuffing** for full document summarization.

## 🚀 Key Features

* **Dual Intelligence Engine:**
    * **Q&A Mode:** Uses RAG with `RecursiveCharacterTextSplitter` (Overlap: 200) to answer specific questions without hallucinations.
    * **Summarization Mode:** Uses "Context Stuffing" to analyze global document context for accurate, bullet-point summaries.
* **Modern UI:** A professional, clean interface built with Streamlit, featuring session state management and a custom "Tech Blue" theme.
* **Robust Architecture:** Implements a "Side-Step" database strategy to handle file locking issues on Windows, ensuring 100% isolation between different document uploads.
* **Smart Caching:** Uses a local Vector Database (`ChromaDB`) to persist embeddings for the active session.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Framework:** LangChain (Classic & Core)
* **Frontend:** Streamlit
* **Vector Database:** ChromaDB (Local Embedded)
* **LLM:** Google Gemma-3-27b-it (via Google Gemini API)
* **Embeddings:** Google Text-Embedding-004

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YourUsername/Document-Analysis-Tool.git](https://github.com/YourUsername/Document-Analysis-Tool.git)
    cd "Document Analysis Tool"
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    * Create a `.env` file in the root folder.
    * Add your Google API key:
        ```text
        GOOGLE_API_KEY=your_actual_api_key_here
        ```

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

## 🧠 System Design Highlights

### 1. Ingestion Pipeline
We use `RecursiveCharacterTextSplitter` with a `chunk_overlap` of **200**. This creates a "sliding window" effect, ensuring that sentences cut off at the end of one chunk are repeated at the start of the next. This prevents the AI from losing context at chunk boundaries.

### 2. RAG vs. Stuffing
* **For Q&A (RAG):** We retrieve only the top 5 relevant chunks (`k=5`). This minimizes token usage and focuses the AI on specific facts.
* **For Summarization (Stuffing):** We bypass the vector search and concatenate the raw text chunks directly. This gives the AI the "Big Picture" needed to summarize the entire document flow.

### 3. File Locking Solution
To prevent `[WinError 32]` when switching files, the system creates a unique, timestamped database folder for every new upload (`./chroma_db_{timestamp}`). The app then switches the "pointer" to the new folder, leaving the old one to be cleaned up safely in the background.

