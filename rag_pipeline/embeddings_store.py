import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# --- Path setup ---
ROOT_DIR = Path(__file__).resolve().parents[1]  # .../Code
DATA_DIR = ROOT_DIR / "data"

CHUNKS_FILE = DATA_DIR / "chunks.pkl"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.bin"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

# --- Load Chunks ---
def load_chunks():
    print(f"🔍 Loading chunks from {CHUNKS_FILE}")
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"❌ chunks.pkl not found at: {CHUNKS_FILE}")
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    print(f"✅ Loaded {len(chunks)} chunks.")
    return chunks

# --- Create Embeddings ---
def create_embeddings(chunks):
    print("⚙️  Encoding chunks with embedding model (BAAI/bge-small-en-v1.5)...")
    embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    texts = [chunk['text'] if isinstance(chunk, dict) else str(chunk) for chunk in chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    print(f"✅ Embeddings shape: {embeddings.shape}")  # (num_chunks, 384)
    return embedding_model, embeddings

# --- Save FAISS Index ---
def save_faiss_index(embeddings):
    print("📦 Building FAISS index...")
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, str(FAISS_INDEX_FILE))
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"✅ Saved FAISS index to {FAISS_INDEX_FILE} and embeddings to {EMBEDDINGS_FILE}")
    return faiss_index

# --- Main ---
if __name__ == "__main__":
    chunks = load_chunks()
    model, embeddings = create_embeddings(chunks)
    save_faiss_index(embeddings)

def load_all():
    chunks = load_chunks()

    print(f"📦 Loading FAISS index from {FAISS_INDEX_FILE}")
    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(f"❌ FAISS index not found at: {FAISS_INDEX_FILE}")
    faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
    print(f"✅ FAISS index loaded with {faiss_index.ntotal} vectors.")

    embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    return embedding_model, faiss_index, chunks

