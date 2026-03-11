import os
import shutil
import gc
import time
import glob
from fastapi import FastAPI, Query, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from rag_logic import load_and_process_pdf, ask_question, summarize_document

# 1. Initialize the FastAPI app
app = FastAPI(title="Documind API", description="API for Document Analysis Tool", version="1.0")

# 2. Global state to hold the vector database and documents
sessions = {}

# 3. Pydantic Schemas (The "Gatekeepers")
class QuestionRequest(BaseModel):
    question: str
    session_id: str

class SessionRequest(BaseModel):
    session_id: str

# --- BACKGROUND TASK: Cleanup Orphaned Sessions ---
def cleanup_orphaned_sessions():
    """Background task to delete folders abandoned by closed tabs."""
    current_time = time.time()
    
    # Find all folders starting with 'chroma_db_'
    for folder in glob.glob("./chroma_db_*"):
        try:
            # Check when the folder was last modified
            folder_age_seconds = current_time - os.path.getmtime(folder)
            
            # If the folder is older than 1 hour (3600 seconds), nuke it
            if folder_age_seconds > 3600:
                shutil.rmtree(folder)
                print(f"🧹 Sweeper deleted abandoned session: {folder}")
        except Exception:
            # If it fails (locked file), ignore it. We will catch it on the next sweep.
            pass

# 4. ENDPOINT: Health Check (Simple GET request)
@app.get("/")
async def root():
    return {"message": "Welcome to the Documind API! Use /upload to analyze a PDF and /ask to ask questions."}  

@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks, # NEW: Inject background task manager
    session_id: str = Query(...), 
    file: UploadFile = File(...)
):
    # 1. Trigger the sweeper to run in the background
    background_tasks.add_task(cleanup_orphaned_sessions)
    
    temp_path = f"temp_{session_id}_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        db, docs = load_and_process_pdf(temp_path, session_id)
        sessions[session_id] = {
            "vector_db": db,
            "documents": docs
        }
        return {"message": f"Successfully processed for session {session_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask")
async def chat_with_pdf(request: QuestionRequest):
    user_session = sessions.get(request.session_id)
    if not user_session or user_session["vector_db"] is None:
        raise HTTPException(status_code=400, detail="Session expired or PDF not uploaded.")
    
    answer = ask_question(user_session["vector_db"], request.question)
    return {"answer": answer}

@app.post("/summarize")
async def get_summary(request: SessionRequest):
    user_session = sessions.get(request.session_id)
    if not user_session or user_session["documents"] is None:
        raise HTTPException(status_code=400, detail="Session expired or PDF not uploaded.")
    
    summary = summarize_document(user_session["documents"])
    return {"summary": summary}

@app.post("/reset")
async def reset_session(request: SessionRequest):
    sid = request.session_id
    
    if sid in sessions:
        # 1. Fetch the database object
        db = sessions[sid].get("vector_db")
        
        # 2. THE INSIDE-OUT WIPE: Tell Chroma to delete the data internally first
        if db is not None:
            try:
                db.delete_collection()
            except Exception as e:
                print(f"Notice: Collection already empty or locked: {e}")
        
        # 3. Clear RAM
        sessions[sid]["vector_db"] = None
        sessions[sid]["documents"] = None
        del sessions[sid]
        
        # Force Python to dump memory
        gc.collect() 
        
        # 4. Try to delete the physical folder (with a small delay for Windows)
        db_path = f"./chroma_db_{sid}"
        if os.path.exists(db_path):
            import time
            time.sleep(1.5) # Give Windows 1.5 seconds to release the SQLite lock
            try:
                shutil.rmtree(db_path)
                print(f"🗑️ Successfully deleted {db_path}")
            except Exception as e:
                print(f"⚠️ Windows locked {db_path}. The Background Sweeper will get it later.")
                
        return {"message": f"Session {sid} wiped successfully."}
        
    return {"message": "No active session found."}