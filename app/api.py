import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.ingest import load_and_chunk_pdf, build_and_save_index, load_index, FAISS_INDEX_DIR
from app.chain import build_chain, ask
from dotenv import load_dotenv

load_dotenv()

# Global state 
chain_state = {"chain": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the chain at startup if a FAISS index already exists
    if os.path.exists(FAISS_INDEX_DIR):
        chain_state["chain"] = build_chain()
        print("[api] Chain loaded at startup.")
    else:
        print("[api] No FAISS index found. Upload a document first.")
    yield


app = FastAPI(
    title="Financial Document Q&A API",
    description="RAG-powered question answering over SEC 10-K filings",
    version="1.0.0",
    lifespan=lifespan
)


# Request / Response models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[int]


# Endpoints
@app.get("/")
def health_check():
    """Health check endpoint — confirms API is running."""
    return {"status": "ok", "index_loaded": chain_state["chain"] is not None}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF to index. Rebuilds the FAISS index from the new document.
    Accepts: multipart/form-data with a 'file' field containing a PDF.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs("data", exist_ok=True)
    save_path = f"data/{file.filename}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = load_and_chunk_pdf(save_path)
    build_and_save_index(chunks)
    chain_state["chain"] = build_chain()

    return {"message": "Document indexed successfully", "chunks": len(chunks)}


@app.post("/ask", response_model=AnswerResponse)
def answer_question(request: QuestionRequest):
    """
    Ask a question about the indexed document.
    Returns the answer and the page numbers of source chunks.
    """
    if chain_state["chain"] is None:
        raise HTTPException(
            status_code=503,
            detail="No document indexed yet. Call /upload first."
        )
    result = ask(chain_state["chain"], request.question)
    return AnswerResponse(answer=result["answer"], sources=result["sources"])


@app.delete("/index")
def clear_index():
    """Clear the FAISS index to start fresh with a new document."""
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
    chain_state["chain"] = None
    return {"message": "Index cleared."}