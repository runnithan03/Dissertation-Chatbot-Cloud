import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Path Setup ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # Goes up from /test
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.pkl")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")

# --- Load Components ---
print("🔧 Loading RAG components...")
try:
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    print(f"✅ Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
except Exception as e:
    print(f"❌ Failed to load chunks.pkl: {e}")
    exit(1)

try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    print("✅ Loaded FAISS index and embedding model.")
except Exception as e:
    print(f"❌ Failed to load FAISS index or model: {e}")
    exit(1)

try:
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm_pipeline = pipeline("text2text-generation", model=llm_model, tokenizer=tokenizer)
    print("✅ Loaded FLAN-T5 model and tokenizer.")
except Exception as e:
    print(f"❌ Failed to load FLAN-T5 pipeline: {e}")
    exit(1)

# --- Query Function ---
def retrieve_relevant_chunks(question, k=3):
    query_embedding = embedding_model.encode([question], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k)
    return [chunks[idx] for idx in indices[0]]

def query_rag_pipeline(question, k=3, max_tokens=150):
    retrieved_chunks = retrieve_relevant_chunks(question, k)
    context = "\n".join(chunk["text"] for chunk in retrieved_chunks)
    prompt = (
        f"Based on the context, answer the question in 3 sentences or less.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer only (do not repeat the context):"
    )
    response = llm_pipeline(prompt, max_new_tokens=max_tokens)
    return response[0]['generated_text'].strip()

# --- Test Run ---
if __name__ == "__main__":
    test_question = "What is the key finding of the equity fund analysis?"
    print("\n💬 Testing RAG pipeline...")
    answer = query_rag_pipeline(test_question)
    print(f"\nQ: {test_question}\nA: {answer}")
