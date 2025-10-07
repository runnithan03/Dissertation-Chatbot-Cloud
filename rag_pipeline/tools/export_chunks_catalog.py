import os, csv, json, pickle
from typing import Any, Tuple

IN = os.environ.get("CHUNKS_PATH", "chunks.pkl")
OUT_DIR = "fine_tune"
CATALOG_CSV = os.path.join(OUT_DIR, "chunks_catalog.csv")
CHUNKS_JSON = os.path.join(OUT_DIR, "chunks.json")

os.makedirs(OUT_DIR, exist_ok=True)

def extract_text_source(obj: Any) -> Tuple[str, str]:
    """
    Return (text, source). Be permissive about shapes.
    """
    # 1) plain string
    if isinstance(obj, str):
        return obj, "unknown"

    # 2) dict-like
    if isinstance(obj, dict):
        # common keys
        text = obj.get("text") or obj.get("page_content") or obj.get("content") or ""
        # look for source in several places
        src = (
            obj.get("source")
            or (obj.get("metadata") or {}).get("source")
            or (obj.get("metadata") or {}).get("file")
            or (obj.get("meta") or {}).get("source")
            or "unknown"
        )
        return text, src

    # 3) LangChain Document
    # (avoid importing langchain just to do isinstance; use duck typing)
    if hasattr(obj, "page_content"):
        text = getattr(obj, "page_content", "") or ""
        meta = getattr(obj, "metadata", {}) or {}
        src = (
            meta.get("source")
            or meta.get("file")
            or meta.get("path")
            or "unknown"
        )
        return text, src

    # 4) tuple/list variants
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        first = obj[0]
        # (text, metadata/source)
        if isinstance(first, str):
            text = first
            src = "unknown"
            if len(obj) > 1:
                second = obj[1]
                if isinstance(second, dict):
                    src = second.get("source") or second.get("file") or second.get("path") or "unknown"
                elif isinstance(second, str):
                    src = second
            return text, src
        # (dict, ...)
        if isinstance(first, dict):
            return extract_text_source(first)

    # Fallback
    return str(obj), "unknown"

# Load file
with open(IN, "rb") as f:
    raw = pickle.load(f)

# Sometimes people store {"chunks":[...]}
if isinstance(raw, dict) and "chunks" in raw:
    items = raw["chunks"]
else:
    items = raw

norm = []
rows = []
count = 0

for i, ch in enumerate(items):
    text, src = extract_text_source(ch)
    # normalise whitespace, keep a preview
    preview = " ".join((text or "").split())[:400]
    if not text:
        continue  # skip empty fragments
    norm.append({"id": i, "source": src or "unknown", "text": text})
    rows.append([i, src or "unknown", preview])
    count += 1

with open(CATALOG_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["chunk_id", "source", "preview_400chars"])
    w.writerows(rows)

with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
    json.dump(norm, f, ensure_ascii=False)

print(f"[export] Loaded from: {IN}")
print(f"[export] Normalised chunks: {count}")
print(f"[export] Wrote: {CATALOG_CSV}")
print(f"[export] Wrote: {CHUNKS_JSON}")
# Show a couple of samples
for s in norm[:3]:
    print(f"ID {s['id']} | source={s['source']}\n{(' '.join(s['text'].split())[:160])}...\n")
