import os
import shutil
import time
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document  

# 1. Load Secrets
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2. Configure Models
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", api_key=GOOGLE_API_KEY)

# --- HELPER: ROBUST CLEANUP ---
def cleanup_old_dbs():
    """
    Tries to remove old database folders. 
    If a folder is locked by Windows, we SKIP it instead of crashing.
    """
    # Find all folders starting with 'chroma_db_'
    db_folders = glob.glob("./chroma_db_*")
    
    for folder in db_folders:
        try:
            shutil.rmtree(folder)
            print(f"🧹 Cleaned up old DB: {folder}")
        except PermissionError:
            print(f"⚠️ Could not delete {folder} (Locked by Windows). Skipping...")
        except Exception as e:
            print(f"⚠️ Error cleaning {folder}: {e}")

# --- CORE FUNCTIONS ---
def load_and_process_pdf(pdf_path):
    """
    Ingests the PDF with a UNIQUE database path to avoid file locks.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1. Try to clean up old mess (but don't crash if we can't)
    cleanup_old_dbs()

    # 2. Create a UNIQUE folder name for this session
    # This ensures we never collide with a locked file
    unique_db_path = f"./chroma_db_{int(time.time())}"
    
    print(f"--- 1. Loading PDF to {unique_db_path}... ---")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("--- 2. Splitting Text... ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)
    
    print("--- 3. Creating Vector Database... ---")
    vector_db = Chroma.from_documents(
        documents=final_documents,
        embedding=embeddings,
        persist_directory=unique_db_path
    )
    print("✅ Vector Database Created!")
    return vector_db, final_documents

def ask_question(vector_db, question):
    print(f"--- Searching for: '{question}' ---")
    
    prompt_template = """
    You are a helpful assistant. Answer the question strictly based on the provided context.
    If the answer is not in the context, just say "I cannot find the answer in the document."
    Context: {context}
    Question: {question}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    result = qa_chain.invoke({"query": question})
    return result['result']

def summarize_document(docs_list):
    print(f"--- Generating Summary... ---")
    full_text = "\n\n".join([doc.page_content for doc in docs_list[:10]])
    
    prompt_template = """
    You are an expert summarizer. 
    Below is the full text of a document. 
    Create a concise, bullet-point summary of the main ideas.
    DOCUMENT TEXT: {text}
    SUMMARY:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm
    
    result = chain.invoke({"text": full_text})
    return result.content