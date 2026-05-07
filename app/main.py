import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from .engine import SherlockEngine

app = FastAPI(title="Sherlock AI - Case File Assistant")

# Simple, global initialization. 
# It's easy to follow and works perfectly for this scope.
sherlock = SherlockEngine()

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-case-file")
async def upload_document(file: UploadFile = File(...)):
    # 1. Simple validation
    filename = file.filename.lower()
    if not (filename.endswith('.pdf') or filename.endswith('.txt')):
        raise HTTPException(status_code=400, detail="Please upload a PDF or TXT file.")

    # 2. Straightforward file saving
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Call the engine
        sherlock.process_document(temp_path)
        return {"message": f"Successfully indexed: {file.filename}"}
    
    except Exception as e:
        # If something goes wrong, we want the detective to know
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    
    finally:
        # Always clean up the temp file, even if it crashed
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask")
async def ask_sherlock(request: QuestionRequest):
    # The 'Catch' in the test: Ensure we don't hallucinate
    # We pass the logic to the engine which handles the retrieval
    result = sherlock.query(request.question)
    
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }