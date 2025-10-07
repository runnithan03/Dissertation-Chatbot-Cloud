# Code/rag_pipeline/data_ingestion.py
import os, re, pickle

from pathlib import Path
from typing import List, Dict
import pickle, re

# --- Paths (works no matter where you run from) ---
ROOT = Path(__file__).resolve().parents[1]     # .../Code
DOCS_DIR = ROOT / "docs"                       # dissertation + references
OUT_PATH = ROOT / "data" / "chunks.pkl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Chunking knobs ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
MIN_CHARS = 40

# Try LangChain splitter (nice) else fallback
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    USE_LANGCHAIN = True
except Exception:
    USE_LANGCHAIN = False


def source_type_from_name(name: str) -> str:
    n = name.lower()
    # Tweak the heuristic if your filename uses a specific pattern
    return "dissertation" if ("dissertation" in n or "thesis" in n) else "reference"


def extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    empty_pages = 0
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if not txt.strip():
            empty_pages += 1
            continue
        # normalise newlines, de-hyphenate, collapse spaces (keep newlines)
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
        txt = re.sub(r"[ \t]+", " ", txt)
        parts.append(txt.strip())
    if empty_pages:
        print(f"  [warn] {pdf_path.name}: {empty_pages} empty/graphics-only pages (consider OCR if critical)")
    return "\n\n".join(parts)


def split_text(text: str) -> List[str]:
    if not text.strip():
        return []
    if USE_LANGCHAIN:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
    else:
        # Simple fixed-width splitter with overlap
        chunks, i, n = [], 0, len(text)
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        while i < n:
            ch = text[i:i + CHUNK_SIZE].strip()
            if len(ch) >= MIN_CHARS:
                chunks.append(ch)
            i += step
    return [c for c in chunks if len(c) >= MIN_CHARS]


def rebuild() -> List[Dict]:
    assert DOCS_DIR.exists(), f"{DOCS_DIR} not found"
    items: List[Dict] = []
    cid = 0
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[ingest] No PDFs found in {DOCS_DIR}")
    for pdf in pdfs:
        print(f"[ingest] {pdf.name}")
        text = extract_pdf_text(pdf)
        if not text.strip():
            print("  [skip] no extractable text")
            continue
        for ch in split_text(text):
            items.append({
                "id": cid,
                "source": pdf.name,
                "source_type": source_type_from_name(pdf.name),
                "text": ch
            })
            cid += 1
        print(f"  [ok] chunks so far: {cid}")
    print(f"[ingest] total chunks: {len(items)}")
    with open(OUT_PATH, "wb") as f:
        pickle.dump(items, f)
    print(f"[ingest] wrote: {OUT_PATH.resolve()}")
    return items


if __name__ == "__main__":
    rebuild()
