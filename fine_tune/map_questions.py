# Code/fine_tune/map_questions.py
import sys, csv, json, pickle, re, numpy as np
from pathlib import Path

# --- Paths (stable regardless of VSCode CWD) ---
ROOT = Path(__file__).resolve().parents[1]      # .../Code
RAG_DIR = ROOT / "rag_pipeline"
FINE_DIR = ROOT / "fine_tune"
OUT_DIR = ROOT / "outputs"

CHUNKS_PKL = RAG_DIR / "chunks.pkl"             # primary
CHUNKS_JSON = FINE_DIR / "chunks.json"          # fallback (optional)
AUTHORING_CSV = FINE_DIR / "authoring.csv"
OUT_CSV = OUT_DIR / "authoring_mapped.csv"

OUT_DIR.mkdir(exist_ok=True)

# --- Retrieval knobs ---
EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"       # set to SAME model as your RAG retriever
TOP_K = 3
SCORE_FLOOR = 0.55                              # raise to filter weak matches
DISSERTATION_BOOST = 0.03                       # small nudge so diss wins ties


# ---------- Utils ----------
def norm(s: str) -> str:
    s = (s or "")
    s = s.lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preview(txt: str, n: int = 160) -> str:
    s = re.sub(r"\s+", " ", txt or "").strip()
    return (s[:n] + "…") if len(s) > n else s


def load_chunks():
    # Prefer PKL, fallback to JSON
    if CHUNKS_PKL.exists():
        raw = pickle.load(open(CHUNKS_PKL, "rb"))
        hint = str(CHUNKS_PKL)
    elif CHUNKS_JSON.exists():
        raw = json.load(open(CHUNKS_JSON, "r", encoding="utf-8"))
        hint = str(CHUNKS_JSON)
    else:
        sys.exit(f"No chunks at {CHUNKS_PKL} or {CHUNKS_JSON}.")

    normed = []
    if isinstance(raw, dict):
        it = raw.items()
    else:
        it = enumerate(raw)
    for k, v in it:
        if isinstance(v, dict):
            cid = v.get("id", v.get("chunk_id", v.get("uuid", k)))
            txt = v.get("text") or v.get("content") or ""
            src = v.get("source", "")
            src_type = v.get("source_type", "reference")
        else:
            cid, txt, src, src_type = k, str(v), "", "reference"
        if txt.strip():
            normed.append({"id": cid, "text": txt, "source": src, "source_type": src_type})
    if not normed:
        sys.exit(f"Parsed 0 chunks from {hint}.")
    print(f"[map] Loaded {len(normed)} chunks from {hint}")
    return normed


def embed_texts(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def main():
    if not AUTHORING_CSV.exists():
        sys.exit(f"Missing {AUTHORING_CSV} (save your sheet as CSV).")

    # Load corpus
    chunks = load_chunks()
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    sources = [c["source"] for c in chunks]
    src_types = [c.get("source_type", "reference") for c in chunks]
    texts_norm = [norm(t) for t in texts]

    # Load authoring
    with AUTHORING_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows or "question" not in reader.fieldnames:
        sys.exit("authoring.csv must contain a 'question' column.")

    # Embed chunks once
    C = embed_texts(texts)  # normalized
    out_fields = list(rows[0].keys())
    for col in ["chunk_ids", "scores", "chunk_sources", "chunk_preview", "needs_review"]:
        if col not in out_fields:
            out_fields.append(col)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()

        processed, flagged = 0, 0
        for r in rows:
            q = r["question"]
            a = r.get("answer", "")

            # 1) Exact answer anchor (prefer dissertation if multiple hits)
            ans_norm = norm(a)
            exact_idxs = []
            if ans_norm and len(ans_norm) >= 12:  # skip tiny answers
                for i, t in enumerate(texts_norm):
                    if ans_norm in t:
                        exact_idxs.append(i)
            if exact_idxs:
                # stable dissertation-first ordering
                exact_idxs.sort(key=lambda i: 0 if src_types[i] == "dissertation" else 1)
                pick = exact_idxs[:TOP_K]
                r["chunk_ids"] = ";".join(str(ids[i]) for i in pick)
                r["scores"] = ";".join(["1.000"] * len(pick))
                r["chunk_sources"] = ";".join(sources[i] for i in pick)
                r["chunk_preview"] = " || ".join(f"[{ids[i]}] {preview(texts[i])}" for i in pick)
                r["needs_review"] = "no"
                w.writerow(r)
                processed += 1
                continue

            # 2) Embedding fallback with dissertation boost
            q_emb = embed_texts([q])[0]
            sims = C @ q_emb  # cosine (normalized)
            order = np.argsort(-sims)[:TOP_K * 4]
            boosted = []
            for i in order:
                s = float(sims[i])
                if src_types[i] == "dissertation":
                    s += DISSERTATION_BOOST
                boosted.append((s, i))
            boosted.sort(reverse=True)
            final_idx = [i for _, i in boosted[:TOP_K]]

            if not final_idx:
                r.update({"chunk_ids": "", "scores": "", "chunk_sources": "", "chunk_preview": "", "needs_review": "yes"})
                flagged += 1
            else:
                best = max(float(sims[i]) for i in final_idx)
                r["chunk_ids"] = ";".join(str(ids[i]) for i in final_idx)
                r["scores"] = ";".join(f"{float(sims[i]):.3f}" for i in final_idx)
                r["chunk_sources"] = ";".join(sources[i] for i in final_idx)
                r["chunk_preview"] = " || ".join(f"[{ids[i]}] {preview(texts[i])}" for i in final_idx)
                r["needs_review"] = "yes" if best < SCORE_FLOOR else "no"
                if r["needs_review"] == "yes":
                    flagged += 1

            w.writerow(r)
            processed += 1

    print(f"[map] Wrote: {OUT_CSV}")
    print(f"[map] Rows processed: {processed} | needs_review=yes: {flagged}")


if __name__ == "__main__":
    main()
