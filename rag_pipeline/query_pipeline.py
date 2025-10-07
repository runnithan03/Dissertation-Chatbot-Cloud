import pickle
import faiss
import numpy as np
import ollama
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# --- Path Setup ---
ROOT_DIR = Path(__file__).resolve().parents[1]  # .../Code
DATA_DIR = ROOT_DIR / "data"

# --- File Paths ---
CHUNKS_FILE = DATA_DIR / "chunks.pkl"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.bin"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

# --- Loaders ---
def load_chunks():
    with open(CHUNKS_FILE, "rb") as f:
        return pickle.load(f)

def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

def load_faiss_index():
    return faiss.read_index(str(FAISS_INDEX_FILE))

def load_llm(mode="local"):
    """
    Load LLM pipeline based on mode:
    - 'local': uses Ollama (for dev)
    - 'cloud': uses Groq (for deployment)
    """
    if mode == "local":
        import ollama
        def local_mistral_pipeline(prompt, max_new_tokens=150):
            response = ollama.chat(model="mistral", messages=[
                {"role": "user", "content": prompt}
            ])
            return [{"generated_text": response["message"]["content"]}]
        return local_mistral_pipeline

    elif mode == "cloud":
        from call_llm import llm_pipeline
        return llm_pipeline

    else:
        raise ValueError(f"Unknown mode: {mode}")

def load_tokenizer():
    return AutoTokenizer.from_pretrained("google/flan-t5-large")

# --- Retrieval ---
def retrieve_relevant_chunks(question, embedding_model, faiss_index, chunks, k=8, distance_threshold=0.85):
    query_embedding = embedding_model.encode([question], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k)

    print(f"🔍 FAISS distances: {distances[0]}")

    filtered_chunks = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i] < distance_threshold:
            filtered_chunks.append(chunks[idx])

    if not filtered_chunks:
        print("⚠️ No chunks passed distance threshold. Using top-1 fallback.")
        return [chunks[indices[0][0]]]

    return filtered_chunks

# --- Main RAG Pipeline ---
def query_rag_pipeline(question, embedding_model, faiss_index, chunks, llm_pipeline, tokenizer, k=3, max_tokens=150):
    print(f"🧠 Question: {question}")

    # 1. Retrieve top-k chunks
    retrieved_chunks = retrieve_relevant_chunks(question, embedding_model, faiss_index, chunks, k)
    texts = [chunk["text"] if isinstance(chunk, dict) else str(chunk) for chunk in retrieved_chunks]

    # 2. Preview
    print("📚 Context preview:")
    for i, txt in enumerate(texts, 1):
        print(f"[{i}] {txt[:150]}...\n")

    # 3. Truncate context to avoid overflow
    context = "\n".join(texts)
    context_tokens = tokenizer.encode(context, truncation=True, max_length=350)
    context = tokenizer.decode(context_tokens)

    # 4. Prompt template
    prompt = (
        "You are an expert dissertation assistant.\n"
        "Based on the context, answer the user's question in 2–3 clear sentences.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer only:"
    )

    # 5. Debug Prompt
    print("📝 Prompt preview:\n", prompt[:800], "\n...")
    print("🧪 Sending to LLM...")

    # 6. Run through local LLM
    response = llm_pipeline(prompt, max_new_tokens=max_tokens)
    return response[0]["generated_text"].strip()

# --- Test Run ---
if __name__ == "__main__":
    print("🔧 Loading components...")
    chunks = load_chunks()
    embedding_model = load_embedding_model()
    faiss_index = load_faiss_index()
    tokenizer = load_tokenizer()
    llm_pipeline = load_llm(mode="local")  # 👈 Local dev

    test_question = "What is Multiple-Response Regression?"
    answer = query_rag_pipeline(test_question, embedding_model, faiss_index, chunks, llm_pipeline, tokenizer)
    print("💬 Answer:\n", answer)

