# app.py

import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from rag_pipeline.query_pipeline import (
    load_chunks,
    load_embedding_model,
    load_faiss_index,
    load_llm,
    load_tokenizer,
    query_rag_pipeline
)

# --- App Setup ---
app = FastAPI()
ROOT_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Load RAG Components ---
print("🔧 Initialising RAG pipeline...")

chunks = load_chunks()
embedding_model = load_embedding_model()
faiss_index = load_faiss_index()
llm_pipeline = load_llm(mode="cloud")  
tokenizer = load_tokenizer()

print(f"✅ Loaded {len(chunks)} chunks.")
print("✅ Components ready.")

# --- Serve Frontend ---
@app.get("/")
def serve_frontend():
    return FileResponse(ROOT_DIR / "static" / "index.html")

# --- Query Endpoint ---
@app.post("/query")
async def handle_query(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()

    if not question:
        return JSONResponse(content={"answer": "⚠️ Please provide a valid question."})

    answer = query_rag_pipeline(
        question,
        embedding_model,
        faiss_index,
        chunks,
        llm_pipeline,
        tokenizer
    )
    return JSONResponse(content={"answer": answer})


# --- Start the App ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# http://localhost:8000/