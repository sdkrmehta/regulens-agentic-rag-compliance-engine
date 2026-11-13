# Multi-stage production container with pre-cached ONNX weights
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache ONNX embedding & reranker model weights during build
ENV HF_HUB_ENABLE_HF_TRANSFER=0
RUN python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding;\
from flashrank import Ranker;\
TextEmbedding(model_name='BAAI/bge-small-en-v1.5');\
SparseTextEmbedding(model_name='Qdrant/bm25');\
Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/app/.model_cache');\
print('Pre-cached all ONNX model weights successfully!')\
"

# Copy source code, data, and scripts
COPY src/ ./src/
COPY data/ ./data/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
