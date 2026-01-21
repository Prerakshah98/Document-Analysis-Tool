# AI Document Analysis Tool

## Overview
A local RAG (Retrieval Augmented Generation) engine that allows users to:
1. **Chat with PDF:** Ask questions and get answers based strictly on the document content.
2. **Summarize:** Generate concise bullet-point summaries of the entire document context.

## Tech Stack
- **LangChain:** For orchestration and RAG logic.
- **Google Gemini (gemma-3-27b-it):** For high-quality generation and reasoning.
- **ChromaDB:** For local vector storage and retrieval.
- **Google Generative AI Embeddings:** For semantic search (text-embedding-004).

## Features
- **Smart Caching:** Automatically detects if a PDF has been processed to save API costs.
- **Context Stuffing:** Uses full-document context for summarization.
- **Hybrid Architecture:** Switches between RAG (for queries) and Stuffing (for summaries).

## Setup
1. Clone the repo.
2. Install requirements: `pip install langchain langchain-google-genai chromadb`
3. Add your `GOOGLE_API_KEY` to a `.env` file.
4. Run the logic: `python rag_logic.py`