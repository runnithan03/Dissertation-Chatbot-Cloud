from .query_pipeline import (
    query_rag_pipeline,
    load_chunks,
    load_faiss_index,
    load_embedding_model,
    load_tokenizer,
    load_llm
)

chunks = load_chunks()
faiss_index = load_faiss_index()
embedding_model = load_embedding_model()
tokenizer = load_tokenizer()
llm_pipeline = load_llm(mode="cloud")

def run_query(prompt: str) -> str:
    return query_rag_pipeline(
        prompt,
        embedding_model,
        faiss_index,
        chunks,
        llm_pipeline,
        tokenizer
    )
