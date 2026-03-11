import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# --- IMPORT GOOGLE MODELS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 1. Load Secrets
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2. Configure Models
# We use text-embedding-004 for fast, free, and memory-light embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=GOOGLE_API_KEY
)

# Keep your Gemma model for answering questions
llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it", 
    api_key=GOOGLE_API_KEY
)

# --- CORE FUNCTIONS ---

def load_and_process_pdf(pdf_path, session_id):
    """
    Ingests the PDF and saves the vector database to a session-specific folder.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Ensure the database folder is strictly named after the unique session ID
    unique_db_path = f"./chroma_db_{session_id}"
    
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
    print("✅ Vector Database Created Successfully!")
    
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