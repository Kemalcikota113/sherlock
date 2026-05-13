import os
import shutil
import tempfile
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .engine import SherlockEngine

app = FastAPI(title="Sherlock - Case File Assistant")


@lru_cache
def get_engine() -> SherlockEngine:
    return SherlockEngine()


class QuestionRequest(BaseModel):
    question: str


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), engine: SherlockEngine = Depends(get_engine)) -> dict:

    name = (file.filename or "").lower()
    if not (name.endswith(".pdf") or name.endswith(".txt")):
        raise HTTPException(400, "Only .pdf and .txt files are supported.")

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(name)[1], delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = engine.ingest(tmp_path, file.filename)
    except Exception as exc:
        raise HTTPException(500, f"Failed to process file: {exc}") from exc
    finally:
        os.remove(tmp_path)

    return {"filename": file.filename, "chunks": chunks}


@app.post("/api/ask")
def ask(request: QuestionRequest, engine: SherlockEngine = Depends(get_engine)) -> dict:
    return engine.query(request.question)


@app.get("/api/documents")
def list_documents(engine: SherlockEngine = Depends(get_engine)) -> dict:
    return {"documents": engine.list_documents()}


@app.delete("/api/documents")
def clear_documents(engine: SherlockEngine = Depends(get_engine)) -> dict:
    engine.clear()
    return {"status": "cleared"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse("app/static/index.html")
