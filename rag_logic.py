import os
import shutil
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
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", api_key=GOOGLE_API_KEY)

# --- CORE FUNCTIONS ---
def load_and_process_pdf(pdf_path):
    """
    Ingests the PDF: Reads -> Splits -> Embeds -> Stores in DB.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # SYSTEM DESIGN: "Fresh Ingestion"
    # If a DB already exists, we delete it to ensure we don't mix old and new data.
    if os.path.exists("./chroma_db"):
        print(f"--- 🧹 Clearing old database to load {pdf_path}... ---")
        shutil.rmtree("./chroma_db") 

    print("--- 1. Loading PDF... ---")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("--- 2. Splitting Text... ---")
    # chunk_overlap=200 prevents sentences from being cut in half at boundaries
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)
    
    print("--- 3. Creating Vector Database... ---")
    vector_db = Chroma.from_documents(
        documents=final_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Vector Database Created!")
    return vector_db, final_documents

def ask_question(vector_db, question):
    """
    Retrieves relevant chunks and asks Gemini to answer based on them.
    """
    print(f"--- Searching for: '{question}' ---")
    
    prompt_template = """
    You are a helpful assistant. Answer the question strictly based on the provided context.
    If the answer is not in the context, just say "I cannot find the answer in the document."
    Context: {context}
    Question: {question}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # RetrievalQA coordinates the "Search -> Prompt -> Answer" flow
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
    """
    Sends the raw text of the chunks to the LLM for summarization.
    """
    print(f"--- Generating Summary for {len(docs_list)} chunks... ---")
    # We join all chunk text into one giant string
    full_text = "\n\n".join([doc.page_content for doc in docs_list])
    
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

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # SYSTEM DESIGN CONTROL:
    # True = Delete DB and rebuild (use when PDF changes)
    # False = Use existing DB (use when asking new questions to same PDF)
    FORCE_REBUILD = True 

    if FORCE_REBUILD or not os.path.exists("./chroma_db"):
        print("--- ⚠️ Rebuild Mode Active ---")
        db, docs_list = load_and_process_pdf("sample.pdf")
    else:
        print("--- 📂 Loading existing DB (Cache Mode) ---")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        # Pull text out of DB so we can summarize it without reading the file
        print("   🔄 Fetching text from DB...")
        existing_data = db.get()
        docs_list = [Document(page_content=text) for text in existing_data['documents']]

    # 2. Run Test Questions
    print("\n--- Test 1: Irrelevant Question ---")
    print(f"AI Answer: {ask_question(db, 'How is the weather?')}")

    print("\n--- Test 2: Real Question ---")
    print(f"AI Answer: {ask_question(db, 'How do I get out of Jail?')}")

    # 3. Run Summary
    if docs_list:
        print("\n--- Test 3: Summarize Document ---")
        print(f"Summary: \n{summarize_document(docs_list)}")
    else:
        print("\n⚠️ Summary skipped (No documents found).")