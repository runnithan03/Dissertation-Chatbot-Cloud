import sys
from pathlib import Path

# Add the root project directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_pipeline.embeddings_store import load_all
from rag_pipeline.query_pipeline import query_rag_pipeline, load_llm

# Load everything
embedding_model, faiss_index, chunks = load_all()
llm_pipeline = load_llm()

# Test a sample question
question = "What is Multiple-Response Regression?"
answer = query_rag_pipeline(question, embedding_model, faiss_index, chunks, llm_pipeline)

print("🧠 Question:", question)
print("💬 Answer:", answer)
