FROM python:3.10-slim

# --- Env Vars ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers

# --- Set working directory ---
WORKDIR /app

# --- Install system deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# --- Install Python deps ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- Copy source code ---
COPY . .

# --- Expose Render port ---
EXPOSE 10000

# --- Launch FastAPI app ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
